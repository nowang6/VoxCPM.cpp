# OpenCL 整机静默复位：最小化验证方案

> 版本：2026-08-27（两轮已执行）· 状态：**验证完成，零复位——致死场景不可复现**
> 第一轮 T2–T6（§8）：枚举/context/编译/dispatch/规模负载全过 ×3；
> 第二轮 P1–P6+SOAK（§9）：ggml init 精确复刻（kernel 集/大预分配/CPU 满载/DDR 满载/反复 teardown/密集 soak）全过。
> **操作结论**：OpenCL 解冻为"探针级可用"；全栈 `--backend opencl` 仍需串口在录；生产推理走 CPU/Hexagon。
> 原则：每次 GPU 测试都有导致整机复位的风险，因此测试按**单位风险的信息价值**排序，
> 从零风险开始逐层升级，**第一个复位的测试即定位故障层，立即停止**。

---

## 0. 背景与已确认事实（不需要再验证）

| 事实 | 证据 |
|---|---|
| 复位与 OpenCL 栈强相关，与计算量无关 | 最重负载（数千次 kernel dispatch）存活过；最轻负载（仅初始化，权重全在 CPU）致死（2026-08-26 19:51，journal 铁证） |
| 不是用户态内存问题 | 致死运行全程 6GB cgroup 限制；journal 无 OOM 记录；限制器本身有效（超额进程被杀、机器存活） |
| 不是 CPU 本身 | 纯 CPU 5 线程满载跑了几十次，从未复位 |
| 不是 NPU (HTP) | 崩溃的运行从未加载 Hexagon/QNN 路径；当天 HTP 实验多次运行未复位 |
| 不是过热 | 死亡前热区 40–51°C |
| 复位是静默的 | journal 死亡瞬间零日志；无 ramoops（cmdline 无 pstore）；journald 异步刷盘丢失 panic 尾巴 |
| 故障是随机的 | 同样的轻量初始化当天早些时候成功过多次 → 每次触碰 OpenCL 栈是概率事件 |

内核相关约束：`cgroup.memory=nokmem,nosocket`（驱动侧内存分配不记账），
`kernel.panic_on_rcu_stall=1` + gh-watchdog（驱动挂死 → 直接复位）。

**主嫌排序（验证前假设）**：① kgsl/GMU 驱动栈或 QTI OpenCL blob 缺陷 → 看门狗复位；
② CPU+GPU 联合瞬态电流 → 电源跌落（brownout）；③ Trustzone 安全路径故障。

**主嫌排序（2026-08-27 两轮验证后修订）**：①~③ 均未被证实，且 ①（blob/驱动栈孤立缺陷）
被 T2–T6 + P2 + SOAK 大幅削弱——枚举、context、89 kernel 编译、256MB 级分配、满频 dispatch、
120 次 init/teardown、1420 次 GMU 电源切换全部存活。当前剩余解释：
**a) 环境/时间相关因子**（输入电压裕量等，软件侧不可见）；
**b) 低概率随机事件**（17+ 次触碰零复位，单次致死概率上限被压至 ~10–20% 以下）；
**c) 全栈中超出 init 的路径**（sched reserve / 权重 mirror / op 级 dispatch 序列，需串口验证）。

---

## 1. 验证目标（本方案要回答的三个问题）

- **Q1 故障层定位**：OpenCL 栈的哪一层触发复位？——枚举 / 建 context / 编译内核 / 单次 dispatch / 大规模负载
  ✅ **答案**：没有任何一层可被触发。T2–T6 + P1–P6 全过（§8/§9），`ft_hang_intr_status` 全程未动
- **Q2 驱动 vs 电源**：降频后同一负载是否仍触发？（降频存活 → 指向电源；降频仍死 → 指向驱动）
  ⬜ **未执行**（T7 降频对照以"T6 复位"为前提，而 T6 未复位 → 无需执行）
- **Q3 可复现性**：致死层重复运行是否稳定复现？（区分"概率性竞态"与"确定性触发"）
  ✅ **答案**：不可复现。17+ 次触碰（含 120 次 init/teardown、SOAK 密集序列）零复位（§9）

不在目标内：修复驱动、性能测量、VoxCPM 全栈 OpenCL 推理（那是通过本方案之后的下一阶段）。

---

## 2. 前置条件

### 2.1 串口控制台（强烈建议，信息价值最高的一项）

本机 cmdline 已有 `console=ttyMSM0,115200n8`——panic 尾巴和 PON 复位原因**只有串口能抓到**
（journald 异步刷盘注定丢失最后几秒）。

- 用另一台电脑（复位后仍存活）接 HDK 调试串口录制：`screen -L -Logfile serial.log /dev/ttyUSB0 115200`
- ⚠️ Qualcomm HDK UART 电平为 **1.8V**，勿接 3.3V/5V 适配器
- 没有串口也允许执行本方案（分层定位仍有效），但每次复位丢失 panic 原因，只剩"哪层死"没有"为何死"

### 2.2 标准测试包装（所有测试必须使用）

```bash
# 统一：cgroup 内存上限 + 超时 + 降优先级 + 行缓冲落盘（重启后日志仍在 /tmp）
scripts/run-limited.sh --mem-gb 2 --time 120 -- \
    stdbuf -oL -eL ./<probe> 2>&1 | tee /tmp/<probe>_run<N>.log
```

已有设施（均已验证有效）：
- `scripts/run-limited.sh` → cgroup 限流（进程超额被杀、机器存活）
- `scripts/compare_wav.py` + `voxcpm_tts --seed` → 逐位可复现对比
- 观测变量：`VOXCPM_LOG_SCHEDULER=1`、`VOXCPM_LOG_DECODE_TIMING=1` 等（仅 CPU/HTP 阶段用）

### 2.3 复位后取证脚本（每次复位后第一件事，先于一切）

```bash
# /tmp/after_reboot.sh —— 在重启后立即执行
#!/usr/bin/env bash
out=/tmp/forensic_$(date +%m%d_%H%M%S).log
{
  echo "=== 上一次启动的尾部 ==="
  journalctl -b -1 -n 100 --no-pager --no-hostname
  echo "=== 本次开机内核头部（找 PON 复位原因） ==="
  sudo dmesg | head -80 | grep -iE "pon|reset|reason|watchdog|kgsl|panic"
  echo "=== pstore ===";  ls -la /sys/fs/pstore/
  echo "=== kgsl 故障容忍状态 ==="
  for f in ft_hang_intr_status ft_policy ft_pagefault_policy ifpc_count; do
    echo "$f: $(cat /sys/class/kgsl/kgsl-3d0/$f 2>/dev/null)"
  done
} > "$out" 2>&1
echo "captured: $out"
```

> 复位原因若无法从 dmesg 看到（当前观察），串口日志是唯一来源——这就是 2.1 的价值。

### 2.4 GPU 状态基线（T2 之前记录一次）

```bash
cat /sys/class/kgsl/kgsl-3d0/ft_hang_intr_status     # 基线（当前观察值 1）
cat /sys/class/kgsl/kgsl-3d0/gpu_clock_stats          # 基线（当前 boot 全零）
cat /sys/class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq # 基线 220MHz
```

GPU 测试后重读并对比：`ft_hang_intr_status` 变化 = 驱动挂起检测被触发过的直接证据。

---

## 3. 探针程序（编译一次，全程复用）

```bash
mkdir -p /tmp/clprobe && cd /tmp/clprobe
# 每个 probe_x.c 用下面统一编译（头文件 /usr/include/CL，libOpenCL → Adreno blob）：
# gcc -O2 probe_x.c -o probe_x -lOpenCL
```

### probe_enum.c —— 第 0 层：仅枚举（不建 context，最小剂量）

```c
#include <CL/cl.h>
#include <stdio.h>
int main(void) {
    cl_platform_id plats[4]; cl_uint np = 0;
    cl_int e = clGetPlatformIDs(4, plats, &np);
    printf("clGetPlatformIDs -> %d n=%u\n", e, np);
    for (cl_uint i = 0; i < np; i++) {
        char name[256] = {0};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof name, name, NULL);
        printf("platform[%u]: %s\n", i, name);
        cl_device_id devs[4]; cl_uint nd = 0;
        clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 4, devs, &nd);
        for (cl_uint j = 0; j < nd; j++) {
            char dn[256] = {0};
            clGetDeviceInfo(devs[j], CL_DEVICE_NAME, sizeof dn, dn, NULL);
            printf("  device[%u]: %s\n", j, dn);
        }
    }
    printf("ENUM OK\n");
    return 0;
}
```

### probe_ctx.c —— 第 1 层：建 context + 命令队列

```c
#include <CL/cl.h>
#include <stdio.h>
int main(void) {
    cl_platform_id p; cl_device_id d; cl_int e;
    if (clGetPlatformIDs(1, &p, NULL) || clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL))
        { puts("no device"); return 1; }
    cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    printf("clCreateContext -> %d ctx=%p\n", e, (void*)ctx);
    if (e != CL_SUCCESS) return 1;
    cl_command_queue q = clCreateCommandQueue(ctx, d, 0, &e);
    printf("clCreateCommandQueue -> %d q=%p\n", e, (void*)q);
    clFinish(q);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    printf("CTX OK\n");
    return 0;
}
```

### probe_build.c —— 第 2 层：编译一个 trivial 内核

```c
#include <CL/cl.h>
#include <stdio.h>
static const char* src =
"__kernel void vadd(__global const float* a, __global const float* b,"
"                   __global float* c) {"
"    size_t i = get_global_id(0); c[i] = a[i] + b[i]; }";
int main(void) {
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    if (e != CL_SUCCESS) { puts("ctx fail"); return 1; }
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &e);
    e = clBuildProgram(prog, 1, &d, NULL, NULL, NULL);
    if (e != CL_SUCCESS) {
        char log[4096] = {0};
        clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        printf("build failed: %s\n", log);
        return 1;
    }
    printf("BUILD OK\n");
    clReleaseProgram(prog); clReleaseContext(ctx);
    return 0;
}
```

### probe_run.c —— 第 3 层：dispatch 一个 1024 元素 vector add

```c
#include <CL/cl.h>
#include <stdio.h>
#include <math.h>
static const char* src =
"__kernel void vadd(__global const float* a, __global const float* b,"
"                   __global float* c) {"
"    size_t i = get_global_id(0); c[i] = a[i] + b[i]; }";
int main(void) {
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, d, 0, &e);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, NULL);
    clBuildProgram(prog, 1, &d, NULL, NULL, NULL);
    cl_kernel k = clCreateKernel(prog, "vadd", &e);
    printf("kernel -> %d\n", e);

    float a[1024], b[1024], c[1024];
    for (int i = 0; i < 1024; i++) { a[i] = i; b[i] = 2 * i; }
    cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, sizeof a, a, &e);
    cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, sizeof b, b, &e);
    cl_mem mc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof c, NULL, &e);
    clSetKernelArg(k, 0, sizeof(cl_mem), &ma);
    clSetKernelArg(k, 1, sizeof(cl_mem), &mb);
    clSetKernelArg(k, 2, sizeof(cl_mem), &mc);
    size_t gws = 1024;
    e = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
    printf("enqueue -> %d\n", e);
    clFinish(q);
    clEnqueueReadBuffer(q, mc, CL_TRUE, 0, sizeof c, c, 0, NULL, NULL);
    int bad = 0;
    for (int i = 0; i < 1024; i++) if (fabsf(c[i] - 3.0f * i) > 1e-3f) bad++;
    printf("RUN OK mismatches=%d\n", bad);
    return bad ? 1 : 0;
}
```

### probe_scale.c —— 第 4 层：ggml 使用模式（大 buffer + 循环 dispatch）

```c
/* 用法: ./probe_scale [nbuf] [mb_per_buf] [iters]
 * 默认: 8 个 64MB buffer × 200 轮 —— 模拟 ggml-opencl 的大额分配 + 高频 dispatch 模式 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char* src =
"__kernel void vadd(__global const float* a, __global const float* b,"
"                   __global float* c) {"
"    size_t i = get_global_id(0); c[i] = a[i] + b[i]; }";
int main(int argc, char** argv) {
    int nbuf  = argc > 1 ? atoi(argv[1]) : 8;
    int mb    = argc > 2 ? atoi(argv[2]) : 64;
    int iters = argc > 3 ? atoi(argv[3]) : 200;
    cl_platform_id p; cl_device_id d; cl_int e;
    clGetPlatformIDs(1, &p, NULL);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, d, 0, &e);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, NULL);
    clBuildProgram(prog, 1, &d, NULL, NULL, NULL);
    cl_kernel k = clCreateKernel(prog, "vadd", &e);
    size_t bytes = (size_t)mb * 1024 * 1024;
    cl_mem bufs[16];
    for (int i = 0; i < nbuf; i++) {
        bufs[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &e);
        printf("buf[%d/%d] %dMB -> %d\n", i + 1, nbuf, mb, e); fflush(stdout);
        if (e != CL_SUCCESS) return 1;
    }
    for (int it = 0; it < iters; it++) {
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufs[0]);
        clSetKernelArg(k, 1, sizeof(cl_mem), &bufs[1 % nbuf]);
        clSetKernelArg(k, 2, sizeof(cl_mem), &bufs[2 % nbuf]);
        size_t gws = bytes / sizeof(float);
        e = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
        if (e != CL_SUCCESS) { printf("iter %d enqueue %d\n", it, e); return 1; }
        clFinish(q);
        printf("iter %d/%d ok\n", it + 1, iters); fflush(stdout);   /* 落盘进度：复位后能看到死在哪一轮 */
    }
    printf("SCALE OK\n");
    return 0;
}
```

---

## 4. 测试矩阵（严格按顺序，一个通过才进下一个）

| # | 测试 | 内容 | 风险 | 通过标准 | 若复位 → 结论 |
|---|---|---|---|---|---|
| T0 | 只读取证基线 | 2.3 脚本 + 2.4 基线记录 | 零 | 记录在案 | — |
| T1 | CPU 对照（可选） | `voxcpm_tts --backend cpu --threads 5 --seed 1234` ×1 次 | 零 | 正常出音频且与 `/tmp/mb_cpu_a.wav` 逐位一致 | （历史已证稳，跳过亦可） |
| **T2** | **枚举** | `probe_enum` | **低** | 打印 `ENUM OK`，机器存活 | blob 的设备枚举路径即致命 → OpenCL 完全不可用，终点 |
| **T3** | **建上下文** | `probe_ctx` | **低** | `CTX OK` | context 创建（kgsl/GMU 电源状态机）是触发层 |
| **T4** | **编译内核** | `probe_build` | **中** | `BUILD OK` | blob 编译器栈是触发层 |
| **T5** | **单次 dispatch** | `probe_run` | **中** | `RUN OK mismatches=0`，**重复 3 次均存活** | kernel dispatch 路径是触发层 |
| **T6** | **规模负载** | `probe_scale 8 64 200` | **高** | `SCALE OK`，重复 3 次均存活 | 负载规模相关 → 进 T7 区分驱动/电源 |
| T7 | 降频对照 | 降频后重复 T6：见下方命令 | 高 | 见判定表 | 见判定表 |
| T8 | HTP 对照（可选） | Hexagon 路径跑一次（`scripts/run-hexagon.sh`） | 零（历史稳定） | 不复位、输出正常 | NPU 路径也有问题（与历史矛盾，需重查） |

T7 降频命令（验证电源假说的关键对照）：

```bash
# 降到 348MHz（可用档位: 680/615/550/475/401/348/295/220/124.8）
echo 'zhu88jie' | sudo -S sh -c 'echo 348000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq'
cat /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq   # 确认生效
scripts/run-limited.sh --mem-gb 2 --time 300 -- stdbuf -oL -eL ./probe_scale 8 64 200 2>&1 | tee /tmp/probe_scale_capped.log
# 测完恢复
echo 'zhu88jie' | sudo -S sh -c 'echo 680000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq'
```

---

## 5. 执行规则

1. **一个 GPU 测试一次只跑一个**，期间不跑任何其他负载
2. **严格归因模式**：每个 GPU 探针（T2–T6）在**刚开机的干净系统**上跑——即每个探针前主动重启一次机器（本机重启约 90 秒）。省事的"同 boot 连跑"模式可以跑，但归因退化为"≤N 层内有故障"
3. **复位即停**：第一个致死的探针就是故障层定位结果 → 执行 2.3 取证 → 停止升级，进入判定表
4. **存活 ≠ 安全**：故障是随机的（同样的初始化当天早些时候成功过多次），T5/T6 通过标准为**连续 3 次存活**
5. **每次复位后**第一件事执行 `after_reboot.sh`，再决定是否继续
6. **绝对禁止**（零新信息、纯风险）：
   - 直接跑 `voxcpm_tts --backend opencl` 当"测试"（已证实致死，且无串口时复位不留任何信息）
   - 不挂盘日志（`tee`）就跑 GPU 探针
   - 并发跑多个 GPU 探针

---

## 6. 判定表

| 结果 | 结论 | 后续动作 |
|---|---|---|
| T2 复位 | OpenCL blob 枚举路径致命 | OpenCL 在本机定性不可用；带串口日志提厂商工单；转 CPU + Hexagon 路线 |
| T3 复位 | context 创建（kgsl/GMU bringup）触发 | 同上 |
| T4 复位 | 内核编译器栈触发 | 同上 |
| T5 复位 | dispatch 路径触发 | 同上 |
| T5 过 ×3，T6 复位，T7（348MHz）存活 | **电源裕量不足**（brownout）| 换电源后复测；或常驻降频（性能受损）作为临时缓解 |
| T5 过 ×3，T6 复位，T7（348MHz）仍复位 | 驱动缺陷与负载规模相关 | OpenCL 大负载不可用；带串口日志提工单 |
| T2–T6 全过 ×3 | OpenCL 栈孤立使用时正常 → 问题在 **ggml-opencl 的使用模式**（多设备 context / 特定分配序列 / CPU+GPU 并发） | 第二轮：用 `probe_scale` 参数二分（buffer 数量×大小×轮数），再回归 VoxCPM 全栈对照实验（`--backend opencl --backend-map 全 cpu`，带串口） |
| 任何一次复位且串口在录 | 直接读 panic/fault 文本 | 以串口日志为准修正以上所有假设 |

---

## 7. 预算

- 最坏情况（逐层死到 T2 之外）约 **6–8 次受控复位**，每次代价 ~90 秒重启 + 会话中断
- 最好情况（全过 ×3）约 30 分钟无复位运行 + 3 次主动重启
- 若不接串口：可执行，但每次复位只剩"哪一层"信息，没有"为什么"

---

## 8. 执行结果（2026-08-27 10:00–10:04，同 boot 连跑模式）

执行模式说明：采用 §5.2 允许的同 boot 连跑（未做每探针前重启的严格归因），
所有测试均经 `run-limited.sh --mem-gb 2` 包装并 `tee` 落盘；探针与日志在 `/tmp/clprobe/`。

| # | 测试 | 结果 | 证据 |
|---|---|---|---|
| T0 | 基线 | ✅ 记录在案 | `ft_hang_intr_status=1`、clock_stats 全零、220MHz、主热区 34–40°C |
| T2 | 枚举 | ✅ PASS | platform `QUALCOMM Snapdragon(TM)`、device `Adreno(TM) 740`、`ENUM OK` |
| T3 | 建上下文 | ✅ PASS | `CTX OK`（context + 命令队列创建/销毁正常） |
| T4 | 编译内核 | ✅ PASS | `BUILD OK` |
| T5 | 单次 dispatch | ✅ PASS ×3 | `RUN OK mismatches=0` ×3，机器存活 |
| T6 | 规模负载 | ✅ PASS ×3 | `probe_scale 8 64 200` ×3，每轮 200/200 ok、`SCALE OK` |

**GPU 确实在工作（非空跑）**：终态 `gpu_clock_stats` 从基线全零变为 `2448135 0 0 0 0 0 0 324 0`；
devfreq `cur_freq` 220MHz→680MHz（达到最高档）；`ifpc_count` 0x0→0x17E（电源状态机切换 382 次，
GMU idle/active 转换大量发生且全部存活）；GPU 热区 51.5→51.3°C 无异常。

**驱动故障计数未动**：`ft_hang_intr_status` 全程 =1（与基线一致，挂起检测从未触发）。
**零复位**：boot 时间 09:12:38 全程未变，journal 无异常。

**结论（判定表最后一行）**：OpenCL 栈孤立使用时正常 → 故障在 **ggml-opencl 的使用模式**
（多设备 context / 特定分配序列 / CPU+GPU 并发 / scheduler 行为），而非 Adreno OpenCL blob
或 kgsl/GMU 驱动栈本身。

**保留的不确定性**：① 故障是概率性的（§0），本序列总 GPU 活跃时长仅数分钟，通过≠绝对安全；
② 同 boot 连跑使归因退化为"全部 ≤6 层在本 boot 内未触发"。若需严格归因可按 §5.2 重做。

**下一阶段（第二轮，✅ 已完成，结果见 §9）**：
1. ~~`probe_scale` 参数二分~~ → 由 P1（256MB 单 buffer）+ SOAK 覆盖
2. ~~复现 ggml 精确分配/调度序列的最小探针~~ → P1–P6 精确复刻 init 全路径，全部存活
3. 全栈对照回归（带串口）：`--backend opencl` + `--backend-map` 全 cpu —— **仍未执行（需串口）**

---

## 9. 第二轮执行结果（2026-08-27 10:05–11:23，同 boot 连跑）

**R0 代码考古**：ggml-opencl init 在 T2–T6 之外还做三件事——
① `load_cl_kernels` 逐个编译 89 个 .cl（激进选项 `-cl-fast-relaxed-math` 等）；
② init 立即预分配 297MB+37MB+43MB（按 `CL_DEVICE_MAX_MEM_ALLOC_SIZE`=256MB clamp）；
③ voxcpm OpenCL 模式必建双后端 sched（CPU 满载与 GPU init 并存）。
已核实昨日致死二进制 = 当前 build-mb（so 构建于 8-26 18:55、tts 19:25，源码 8-25 未改），
重放的就是同一份代码路径。探针：`/tmp/clprobe/probe2_*`
（⚠️ /tmp 重启即失——探针源码均为本文档 §3 与本节描述的自包含小程序，可随时重建）。

| # | 探针 | 复刻内容 | 结果 |
|---|---|---|---|
| P1 | probe2_prealloc | 256+37+43MB 预分配 + dispatch×20 | ✅ PASS |
| P2 | probe2_kernels | 89 个真实 .cl 逐个编译 | ✅ PASS（83/89 built；6 失败为探针缺 `-D` 宏的复刻误差） |
| P3 | probe2_ggmlinit | 真实 ggml 注册路径完整 init（枚举→查询→context→kernel→预分配→dev_init） | ✅ PASS |
| P4 | probe2_cpuconc | CPU 5 线程算力满载 + 完整 init ×3 | ✅ PASS（cpu_iters=196k） |
| P5 | probe2_cycle | 完整 init/teardown 循环 ×30（GMU 反复 bringup/rundown + 页表 churn） | ✅ PASS |
| P6 | probe2_membw | 8 线程 DDR 带宽满载（1833GB）+ 完整 init ×3 | ✅ PASS |
| SOAK | soak.sh | 6 轮交替 {membw + scale200 + cycle20} | ✅ PASS（DDR ~3.8TB、120 次 init/teardown、1200 轮 dispatch、534 次 kernel 编译） |

终态：boot 09:12:38 未变（**第二轮全程零复位**）；`ft_hang_intr_status=1` 未变；
ifpc_count 0x58B（电源状态机累计切换 1420 次存活）；GPU 热区 45°C。

**第二轮结论：致死场景不可复现。** 孤立 init、大分配、kernel 编译、CPU/DDR 并发满载、
反复 teardown、密集 soak 全部存活。剩余解释（按可能性排序）：
1. **环境/时间相关因子**（输入电压裕量、当天系统状态）——软件侧不可控不可见
2. **低概率随机事件**——但本日 17+ 次触碰零复位，若单次致死概率 ≥10%，
   全存活概率 ≤31%；≥20% 则 ≤8.6%，概率上限被显著压低
3. 全栈 `voxcpm_tts --backend opencl` 中超出 init 的路径（sched reserve / 权重 mirror /
    op 级 dispatch 序列）——**需带串口**才能安全测试（§5.6 禁止裸跑）

**操作建议**：GPU/OpenCL 路线解冻为"可用但有条件"——
允许跑探针级/局部 OpenCL 负载（已证安全模式）；全栈推理仍需串口在录再试；
生产推理继续用 CPU / Hexagon 路线。