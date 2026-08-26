/**
 * @brief VoxCPM/ggml 性能诊断工具
 *
 * 定位推理性能问题：
 *   1. CPU 拓扑与亲和性诊断（Android 上 adb shell 进程常被限制到小核 cpuset）
 *   2. Q4_K x Q8_0 matmul 微基准（贴近 VoxCPM decode 的真实负载形状）
 *   3. 线程数扫描，找最优线程配置
 *
 * 用法:
 *   voxcpm_perf                    # 系统诊断 + 默认基准 + 线程扫描
 *   voxcpm_perf --matmul-only      # 只跑基准
 *   voxcpm_perf --pin 0,1,2,3,4    # 绑定指定核心再测
 *   voxcpm_perf --threads 6        # 指定线程数
 *   voxcpm_perf --m 8 --n 4096 --k 4096   # 自定义 matmul 形状
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sched.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    bool sys_info = true;
    bool matmul = true;
    bool sweep = true;
    int threads = 0;          // 0 = 使用全部可用核
    int max_sweep_threads = 8;
    int64_t m = 4;            // matmul batch (decode 步长)
    int64_t n = 4096;         // 输出行数
    int64_t k = 4096;         // 内积维度
    int reps = 5;
    std::vector<int> pin_cpus;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--matmul-only") { opt.sys_info = false; opt.sweep = false; }
        else if (arg == "--sys-only") { opt.matmul = false; opt.sweep = false; }
        else if (arg == "--threads") { opt.threads = std::atoi(next_value("--threads").c_str()); opt.sweep = false; }
        else if (arg == "--m") { opt.m = std::atoll(next_value("--m").c_str()); }
        else if (arg == "--n") { opt.n = std::atoll(next_value("--n").c_str()); }
        else if (arg == "--k") { opt.k = std::atoll(next_value("--k").c_str()); }
        else if (arg == "--reps") { opt.reps = std::atoi(next_value("--reps").c_str()); }
        else if (arg == "--pin") {
            std::stringstream ss(next_value("--pin"));
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) opt.pin_cpus.push_back(std::atoi(item.c_str()));
            }
            opt.sweep = false;
        }
        else if (arg == "--help" || arg == "-h") {
            std::printf(
                "voxcpm_perf — VoxCPM/ggml 性能诊断\n"
                "  (default)          系统诊断 + matmul 基准 + 线程扫描\n"
                "  --matmul-only      只跑 matmul 基准\n"
                "  --sys-only         只打印系统诊断\n"
                "  --threads N        指定线程数（禁用扫描）\n"
                "  --pin 0,1,2        绑定指定 CPU（禁用扫描）\n"
                "  --m N --n N --k N  matmul 形状（默认 m=4 n=4096 k=4096）\n"
                "  --reps N           重复次数（默认 5，取最小值）\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s (try --help)\n", arg.c_str());
            std::exit(1);
        }
    }
    return opt;
}

// ---------------------------------------------------------------------------
// 系统诊断
// ---------------------------------------------------------------------------

std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (f.is_open()) {
        std::getline(f, line);
    }
    return line;
}

int online_cpu_count() {
    const std::string line = read_first_line("/sys/devices/system/cpu/online");
    int count = 0;
    // 格式如 "0-7" 或 "0-3,5-7"
    std::stringstream ss(line);
    std::string part;
    while (std::getline(ss, part, ',')) {
        const auto dash = part.find('-');
        if (dash != std::string::npos) {
            count += std::atoi(part.substr(dash + 1).c_str()) - std::atoi(part.c_str()) + 1;
        } else if (!part.empty()) {
            count += 1;
        }
    }
    return count > 0 ? count : 1;
}

void print_system_info() {
    std::printf("==================== 系统诊断 ====================\n");

#ifdef __ANDROID__
    char value[92] = {0};
    if (__system_property_get("ro.product.model", value) > 0) {
        std::printf("设备型号    : %s\n", value);
    }
    if (__system_property_get("ro.build.version.release", value) > 0) {
        std::printf("Android     : %s\n", value);
    }
#endif

    // CPU 拓扑：按最大频率分簇
    const int n_cpu = online_cpu_count();
    std::printf("在线 CPU    : %d 个\n", n_cpu);

    std::map<long long, std::vector<int>> clusters;  // max_freq_khz -> cpus
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        char path[128];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        const std::string freq_str = read_first_line(path);
        const long long freq = freq_str.empty() ? -1 : std::atoll(freq_str.c_str());
        clusters[freq].push_back(cpu);
    }

    int cluster_idx = 0;
    for (const auto& [freq, cpus] : clusters) {
        std::string cpu_list;
        for (size_t i = 0; i < cpus.size(); ++i) {
            if (i) cpu_list += ",";
            cpu_list += std::to_string(cpus[i]);
        }
        std::string freq_desc = freq < 0 ? "未知" : std::to_string(freq / 1000) + " MHz";
        std::printf("CPU 簇 %d    : [%s]  max=%s%s\n", cluster_idx++, cpu_list.c_str(),
                    freq_desc.c_str(),
                    cluster_idx == 1 && clusters.size() > 1 ? "" : "");
    }

    // 找出大核（最高频簇）
    std::vector<int> big_cpus;
    if (!clusters.empty()) {
        big_cpus = clusters.rbegin()->second;
        std::string list;
        for (size_t i = 0; i < big_cpus.size(); ++i) {
            if (i) list += ",";
            list += std::to_string(big_cpus[i]);
        }
        std::printf("大核        : [%s] (%zu 个)\n", list.c_str(), big_cpus.size());
    }

    // 当前亲和性（cpuset 限制会直接体现在这里）
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        std::vector<int> usable;
        for (int cpu = 0; cpu < n_cpu; ++cpu) {
            if (CPU_ISSET(cpu, &allowed)) usable.push_back(cpu);
        }
        std::string list;
        for (size_t i = 0; i < usable.size(); ++i) {
            if (i) list += ",";
            list += std::to_string(usable[i]);
        }
        std::printf("可用亲和性  : [%s] (%zu/%d)\n", list.c_str(), usable.size(), n_cpu);

        if ((int)usable.size() < n_cpu) {
            std::printf("\n*** 警告: 本进程只能使用 %zu/%d 个核！***\n", usable.size(), n_cpu);
            std::printf("Android 常把 adb shell 进程限制在小核 cpuset，性能可差 3-5 倍。\n");
            std::printf("修复方法（root）: echo %d > /dev/cpuset/top-app/tasks\n", getpid());
            std::printf("或重启程序前执行: echo $$ > /dev/cpuset/top-app/tasks\n");
        }
    }

    const std::string cgroup = read_first_line("/proc/self/cgroup");
    if (!cgroup.empty()) {
        std::printf("cgroup      : %s\n", cgroup.c_str());
    }

    std::printf("==================================================\n\n");
}

// ---------------------------------------------------------------------------
// matmul 微基准: C[N,M] = A[N,K](Q4_K) x B[K,M](Q8_0)
// ---------------------------------------------------------------------------

struct MatmulBench {
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    struct ggml_context* ctx = nullptr;       // 张量
    struct ggml_context* graph_ctx = nullptr; // 计算图
    struct ggml_tensor* a = nullptr;
    struct ggml_tensor* b = nullptr;
    struct ggml_cgraph* graph = nullptr;
    size_t bytes_touched = 0;
    double flops = 0;
};

MatmulBench setup_matmul(int64_t m, int64_t n, int64_t k) {
    MatmulBench mb;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        std::fprintf(stderr, "no CPU device found\n");
        std::exit(1);
    }
    mb.backend = ggml_backend_dev_init(dev, nullptr);
    if (!mb.backend) {
        std::fprintf(stderr, "failed to init CPU backend\n");
        std::exit(1);
    }

    // 张量元数据
    size_t ctx_size = 3 * ggml_tensor_overhead();
    struct ggml_init_params ip = {ctx_size, nullptr, true};
    mb.ctx = ggml_init(ip);

    mb.a = ggml_new_tensor_2d(mb.ctx, GGML_TYPE_Q4_K, k, n);
    mb.b = ggml_new_tensor_2d(mb.ctx, GGML_TYPE_F32, k, m);  // 激活侧 F32，kernel 内部量化
    mb.bytes_touched = ggml_nbytes(mb.a) + ggml_nbytes(mb.b);
    mb.flops = 2.0 * (double)n * (double)m * (double)k;

    // 权重 buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(mb.ctx, mb.backend);

    // 用固定 seed 的伪随机数据填充后量化（避免全零数据的病态路径）
    {
        const int64_t n_per_row = k;
        std::vector<float> src(n_per_row);
        unsigned int seed = 1337;
        auto next_rand = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)((int)(seed >> 8) % 2000 - 1000) / 1000.0f;
        };

        for (int64_t row = 0; row < n; ++row) {
            for (int64_t i = 0; i < n_per_row; ++i) src[i] = next_rand() * 0.1f;
            ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(),
                                (char*)mb.a->data + row * ggml_row_size(GGML_TYPE_Q4_K, n_per_row),
                                0, 1, n_per_row, nullptr);
        }
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t i = 0; i < n_per_row; ++i) src[i] = next_rand() * 0.1f;
            std::memcpy((char*)mb.b->data + row * n_per_row * sizeof(float), src.data(),
                        n_per_row * sizeof(float));
        }
    }

    // 计算图
    size_t graph_ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    struct ggml_init_params gp = {graph_ctx_size, nullptr, true};
    mb.graph_ctx = ggml_init(gp);
    mb.graph = ggml_new_graph(mb.graph_ctx);
    struct ggml_tensor* result = ggml_mul_mat(mb.ctx, mb.a, mb.b);
    ggml_build_forward_expand(mb.graph, result);

    mb.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mb.backend));
    if (!ggml_gallocr_alloc_graph(mb.gallocr, mb.graph)) {
        std::fprintf(stderr, "failed to alloc graph\n");
        std::exit(1);
    }
    return mb;
}

double run_matmul_once(MatmulBench& mb) {
    const auto t0 = Clock::now();
    if (ggml_backend_graph_compute(mb.backend, mb.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        std::exit(1);
    }
    const auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void run_benchmark(MatmulBench& mb, int threads, int reps) {
    ggml_backend_cpu_set_n_threads(mb.backend, threads);
    run_matmul_once(mb);  // warmup
    run_matmul_once(mb);

    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        best = std::min(best, run_matmul_once(mb));
    }

    const double gbps = mb.bytes_touched / best / 1e9;
    const double gops = mb.flops / best / 1e9;
    std::printf("threads=%-2d  %8.2f ms  %7.2f GB/s  %7.2f GFLOPS\n",
                threads, best * 1e3, gbps, gops);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    if (opt.sys_info) {
        print_system_info();
    }

    // 应用 --pin
    if (!opt.pin_cpus.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int cpu : opt.pin_cpus) CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            std::fprintf(stderr, "sched_setaffinity failed (需要 root 或有效 CPU 列表)\n");
            return 1;
        }
        std::printf("已绑定 CPU: ");
        for (size_t i = 0; i < opt.pin_cpus.size(); ++i) {
            std::printf("%d%s", opt.pin_cpus[i], i + 1 < opt.pin_cpus.size() ? "," : "");
        }
        std::printf("\n");
    }

    if (!opt.matmul) return 0;

    std::printf("========== matmul 基准: A[%lld,%lld] Q4_K 权重 x B[%lld,%lld] F32 激活 ==========\n",
                (long long)opt.n, (long long)opt.k, (long long)opt.k, (long long)opt.m);

    MatmulBench mb = setup_matmul(opt.m, opt.n, opt.k);

    if (opt.sweep) {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        int max_threads = 8;
        if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
            int n = CPU_COUNT(&allowed);
            max_threads = std::min(opt.max_sweep_threads, n);
        }
        for (int t = 1; t <= max_threads; ++t) {
            run_benchmark(mb, t, opt.reps);
        }
    } else {
        int threads = opt.threads;
        if (threads <= 0) {
            cpu_set_t allowed;
            CPU_ZERO(&allowed);
            if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
                threads = CPU_COUNT(&allowed);
            } else {
                threads = 4;
            }
        }
        run_benchmark(mb, threads, opt.reps);
    }

    return 0;
}
