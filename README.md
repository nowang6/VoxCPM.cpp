# VoxCPM.cpp

基于 [ggml](third_party/ggml) 的 VoxCPM 语音合成（TTS）**纯 C++ 推理引擎**。无 Python、无 PyTorch，编译即得可独立分发的命令行工具与 HTTP 服务，支持 x86 / ARM Linux / **安卓原生** / CUDA / Vulkan。

## 特性

- **零依赖推理**：ggml 单栈实现 LocEnc / BaseLM / ResidualLM / FSQ / LocDiT / AudioVAE 全流水线
- **多后端**：CPU（含 NEON/dotprod/i8mm 向量化）、Vulkan
- **安卓原生运行**：交叉编译产出 bionic 二进制，adb root shell 直接执行，不需要 Termux / proot
- **OpenAI 兼容服务**：`/v1/audio/speech`、音色注册管理，支持 wav/mp3/opus/flac/pcm 与 SSE 流式
- **量化工具链**：`voxcpm_quantize`、`voxcpm_imatrix`（重要性矩阵校准）
- **性能诊断**：`voxcpm_perf` 一键输出 CPU 拓扑、亲和性、量化 matmul 吞吐、最优线程数

## 快速开始

```bash
# 1. 编译（x86_64 Linux / macOS）
cmake -B build
cmake --build build -j

# 2. 合成语音（带参考音色克隆）
./build/examples/voxcpm_tts \
  --model-path ./models/voxcpm-0.5b-q4_0.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，这是一段本地合成的语音。" \
  --output ./out.wav \
  --threads 3

# 3. 不带参考音色（默认音色）
./build/examples/voxcpm_tts \
  --model-path ./models/voxcpm-0.5b-q4_0.gguf \
  --text "你好，世界。" --output hello.wav --threads 3
```

依赖：CMake ≥ 3.14、C++17 编译器。`nlohmann/json` 在 `third_party/json` 缺失时由
FetchContent 自动拉取；离线环境可手动放置：

```bash
git clone --depth 1 -b v3.11.3 https://github.com/nlohmann/json third_party/json
```

## 模型量化

从 HuggingFace 的 PyTorch 原始权重（VoxCPM-0.5B）到量化 GGUF 共两步，转换依赖只需装一次：

```bash
# 0. 准备转换环境（仅步骤 1 需要，推理本身零 Python 依赖）
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# 1. PyTorch → F16 GGUF（中间产物，约 2.8 GB）
python scripts/convert_voxcpm_to_gguf.py /path/to/VoxCPM-0.5B \
    --output models/voxcpm-0.5b-f16.gguf

# 2. F16 → Q8_0 量化（最终产物约 476 MB，压缩约 5.9 倍）
./build/examples/voxcpm_quantize \
    --input models/voxcpm-0.5b-f16.gguf \
    --output models/voxcpm-0.5b-q8_0.gguf \
    --type Q8_0 --threads 8
```

## 性能调优

### big.LITTLE SoC 上线程数 ≠ 核心数

ARM 大小核架构上，线程数应设为**大核数量**。小核加入线程池会拖慢每个同步点，实测反而更慢。
Snapdragon 8 Gen 2（1×X3 + 4×A715/A710 + 3×A510）同一句话的实测：

| 线程数 | 端到端 | RTF（越低越好） |
|-------|--------|------|
| 4（默认） | 10.87s | 2.43 |
| **5** | **9.35s** | **2.12** |
| 6 | 11.64s | 2.65 |
| 8（全核） | 15.97s | 3.12 |

结论：8 Gen 2 用 `--threads 5`；8 Gen 3（1+5+2 结构）试 `--threads 6`。

Snapdragon 6 Gen 4（4×A520 @1.8 GHz + 3×A720 @2.2 GHz + 1×A720 @2.3 GHz，Android 16）
的 `voxcpm_perf` matmul 基准（A[4096,4096] Q4_K × B[4096,4] F32）则更为极端——悬崖就在
快核（A720，核 4–7）的规模上：

| 线程数 | 耗时 | 带宽 | 吞吐 |
|-------|------|------|------|
| 1 | 3.32 ms | 2.87 GB/s | 40.48 GFLOPS |
| 2 | 1.96 ms | 4.86 GB/s | 68.64 GFLOPS |
| **3（占满中核簇）** | **1.47 ms** | **6.47 GB/s** | **91.40 GFLOPS** |
| 4 | 20.81 ms | 0.46 GB/s | 6.45 GFLOPS |
| 5 | 31.85 ms | 0.30 GB/s | 4.21 GFLOPS |
| 6 | 20.40 ms | 0.47 GB/s | 6.58 GFLOPS |
| 7 | 20.15 ms | 0.47 GB/s | 6.66 GFLOPS |
| 8（全核） | 7.53 ms | 1.26 GB/s | 17.81 GFLOPS |

threads=4 起吞吐塌缩一个数量级：该机亲和性 8/8、cpuset 未受限，塌缩纯粹来自大小核调度——
不绑核时调度器不会把 4 个线程都留在 A720 上，只要有一个工作线程落到 A520（或跨簇迁移），
每个同步点都被最慢线程拖住；threads=8 全核也只恢复到最优值的 1/5。结论：6 Gen 4 用
`--threads 3`（恰好占满 2.2 GHz A720 中核簇，比单跑 2.3 GHz 大核快 2.3 倍）；换机型先跑
`voxcpm_perf` 实测，可用 `--matmul-only --pin 4,5,6,7` 验证绑核后的真实快核吞吐。

### 性能诊断工具 `voxcpm_perf`

```bash
./build/examples/voxcpm_perf              # 拓扑 + 亲和性 + matmul 基准 + 线程扫描
./build/examples/voxcpm_perf --sys-only   # 只看 CPU 簇 / 大核 / 可用亲和性 / cgroup
./build/examples/voxcpm_perf --matmul-only --pin 4,5,6,7   # 绑定指定核心测量
```

工具输出 Q4_K 权重 × F32 激活的 matmul 吞吐（GB/s、GFLOPS）——这是 decode 阶段的核心瓶颈，
可用于快速对比不同线程数 / 绑核策略下的内核吞吐。

## 构建指南

### x86_64（Linux / WSL / macOS）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### ARM64 Linux（含手机上本地编译）

⚠️ **GCC 11 等旧编译器不认识新核心（如 Cortex-X3），`-mcpu=native` 会静默退化为基础
armv8-a，量化矩阵乘法走慢速路径，性能损失数倍且无任何报错。** 已知目标机型时请显式指定架构：

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOXCPM_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="armv8.6-a+dotprod+i8mm+bf16+fp16"
cmake --build build -j
```

该架构串适用于骁龙 8 Gen 1 及更新的旗舰（X2/X3/X4 核心均支持全套特性）。先验证 CPU：

```bash
grep -o -E 'asimddp|i8mm|bf16|fphp' /proc/cpuinfo | sort -u
# 四项都有 → 可用；缺任一项会直接 SIGILL 崩溃，老旗舰（855/865）请降低架构串
```


### Android 原生（交叉编译）

产出直接运行于 adb root shell 的 bionic 二进制。构建机要求：clang ≥ 16（目标
`aarch64-linux-android29`）+ NDK sysroot + lld。踩坑记录：

| 坑 | 解法 |
|----|------|
| NDK 官方只发布 x86_64 宿主编译器 | 用任意宿主架构的 clang ≥ 16 指向 NDK sysroot（sysroot 是纯数据） |
| GNU ld 读不了 NDK 库的 zstd debug 段 | `-fuse-ld=lld` |
| NDK libc++ 的 ABI 命名空间是 `__ndk1`（非 `__1`）且带 ABI tag | 头文件与静态库必须同源于 NDK |
| clang < 16 解析不了新 libc++ 头的 namespace 属性语法 | 编译器必须 ≥ 16 |
| NDK 无 libgcc | 用 `libclang_rt.builtins-aarch64-android.a` 顶替 |
| CMake `SYSTEM_NAME=Android` 要求 standalone-gcc 布局 | 改设 `CMAKE_SYSTEM_NAME=Linux` + 交叉参数 |
| 官方 toolchain 未指定 ABI 时默认 **armeabi-v7a（32 位）**：`ggml-cpu-impl.h` 的 32 位 fallback `vcvtnq_s32_f32` 与 clang 自带 `arm_neon.h` 重定义冲突，且 armv8.6 架构串对 32 位目标无效 | 显式 `-DANDROID_ABI=arm64-v8a`；改 ABI 必须删净 build 目录重配，旧缓存上追加无效 |
| OpenMP 默认开启，`libggml-cpu.so` 依赖 `libomp.so`，设备上没有 → 运行时 `CANNOT LINK EXECUTABLE` | `-DGGML_OPENMP=OFF`（ggml 自带线程池，不影响多线程性能） |

链接配方：`-nostdlib++` + NDK 的 `libc++_static.a` + `libc++abi.a` + `libunwind.a` 全静态。

```bash
# NDK 指向本机安装路径（r25+ 均可）
NDK=$HOME/android-ndk-r27d

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DVOXCPM_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="armv8.6-a+dotprod+i8mm+bf16+fp16" \
  -DGGML_OPENMP=OFF \
  -DVOXCPM_BUILD_TESTS=OFF -DVOXCPM_BUILD_BENCHMARK=OFF
cmake --build build-android -j
```

`ANDROID_ABI=arm64-v8a` 必须显式指定（默认 32 位会构建失败，见上表）；架构串要求
`/proc/cpuinfo` 同时具备 `asimddp/i8mm/bf16/fphp` 四项特性——Cortex-A720/A520（如
Snapdragon 6 Gen 4、8 Gen 1 及更新旗舰）全系支持，老机型见"ARM64 Linux"一节的降级说明。

### 构建选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `VOXCPM_NATIVE` | ON | ggml 自动探测宿主 CPU；ARM 上建议关闭并显式指定架构（见上） |
| `VOXCPM_CUDA` / `VOXCPM_VULKAN` | OFF | GPU 后端 |
| `VOXCPM_BUILD_EXAMPLES` | ON | CLI 工具与服务器 |
| `VOXCPM_BUILD_BENCHMARK` / `VOXCPM_BUILD_TESTS` | ON | 基准套件 / Catch2 单测 |
| `VOXCPM_ENABLE_MP3` / `VOXCPM_ENABLE_OPUS` | ON | 服务器音频编码（Opus 依赖 ffmpeg） |

## `voxcpm_tts` 参数参考

```
--text, -t TEXT              必填，要合成的文本
--output, -o PATH            输出 wav 路径
--model-path PATH            GGUF 模型路径
--prompt-audio / --prompt-text    参考音频 + 转写文本（音色克隆）
--reference-audio PATH       另一种参考音频入口
--backend {cpu|cuda|vulkan|auto}  默认 cpu
--threads N                  推理线程数（默认 4，big.LITTLE 上建议大核数）
--inference-timesteps N      CFM 迭代步数（默认 10；降到 6 提速约 40%，音质略降）
--cfg-value FLOAT            CFG 强度（默认 2.0）
--stream --stream-dir DIR    流式分片输出
--retry-badcase[-max-times N / -ratio-threshold F]  劣质结果自动重试
--normalize                  文本归一化
```

## HTTP 服务器（OpenAI 兼容）

```bash
./build/examples/voxcpm-server \
  --host 127.0.0.1 --port 8080 \
  --model-path ./models/voxcpm-0.5b-q4_k.gguf \
  --model-name voxcpm-1.5 \
  --threads 5 --backend cpu \
  --voice-dir ./runtime/voices \
  --max-queue 8 --max-decode-steps 512 \
  --output-sample-rate 24000 \
  --disable-auth
```

| 端点 | 说明 |
|------|------|
| `GET /healthz` | 健康检查 |
| `POST /v1/voices` | 注册音色：multipart 字段 `id`、`text` + 音频文件 `audio` |
| `GET /v1/voices/{id}` | 查询音色元信息 |
| `DELETE /v1/voices/{id}` | 删除音色 |
| `POST /v1/audio/speech` | 合成语音 |

使用示例：

```bash
# 注册音色
curl -X POST http://127.0.0.1:8080/v1/voices \
  -F "id=taiyi" \
  -F "text=对，这就是我，万人敬仰的太乙真人。" \
  -F "audio=@./examples/tai_yi_xian_ren.wav"

# 合成
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm-1.5",
    "input": "大家好，我现在正在大可奇奇体验AI科技。",
    "voice": "taiyi",
    "response_format": "wav",
    "speed": 1.0,
    "stream_format": "audio"
  }' --output ./speech.wav
```

- `response_format`：`mp3`（内置编码器，ffmpeg 兜底）/ `opus`（需 ffmpeg，否则 501）/ `flac` / `wav` / `pcm`
- `stream_format`：`audio` 整段返回；`sse` 返回 `audio.delta` 分片事件 + 结束 `audio.completed`
- `input` 限 1–4096 字符；`speed` 0.25–4.0；`instructions` 兼容字段，非空报错
- 请求串行执行，队列上限 `--max-queue`，满了返回 503
- `--output-sample-rate 24000` 可重采样输出（OpenAI 风格 pcm 客户端常需要）

## 安卓设备部署

自包含目录结构（编译产物 + 模型）：

```
voxcpm-android/
├── run.sh                  # mksh 兼容启动脚本（自动设 LD_LIBRARY_PATH）
├── <model>.gguf
├── bin/                    # voxcpm_tts / voxcpm-server / voxcpm_quantize / voxcpm_imatrix / voxcpm_perf
└── lib/                    # libggml.so.0 / libggml-cpu.so.0 / libggml-base.so.0
```

```bash
adb push voxcpm-android /data/local/tmp/voxcpm-android
adb push voxcpm-0.5b-q8_0.gguf /data/local/tmp/voxcpm-android/

adb shell
cd /data/local/tmp/voxcpm-android
export LD_LIBRARY_PATH=/data/local/tmp/voxcpm-android/lib

chmod +x run.sh bin/*

./run.sh \
  --model-path ./voxcpm-0.5b-q4_k.gguf \
  --text "$(cat test.txt)" \
  --output out4k.wav \
  --threads 3

# 取回音频（电脑端）
adb pull /data/local/tmp/voxcpm-android/out8.wav
```


## 常见问题

| 现象 | 原因与解法 |
|------|-----------|
| 运行即崩溃 / SIGILL | CPU 缺编译时指定的特性（dotprod/i8mm 等），换更低架构串重编 |
| ARM 上性能远低于预期 | `-mcpu=native` 静默退化，见"ARM64 Linux"一节显式指定架构 |
| Android 上性能只有预期 1/3 | cpuset 限制到小核，用 `voxcpm_perf` 确认并修复 |
| `third_party/json` 拉取失败 | 网络问题，手动 clone nlohmann/json v3.11.3 到该目录 |
| glibc 二进制在安卓跑不了 | bionic 系统，需要按"Android 原生"章节交叉编译专用版本 |

## 项目结构

```
src/            运行时实现（weight-store、各模块、CFM、tokenizer、server 核心）
include/voxcpm/ 公共头文件
examples/       voxcpm_tts / voxcpm-server / voxcpm_quantize / voxcpm_imatrix / voxcpm_perf
benchmark/      完整基准套件（JSON 报告）
tests/          Catch2 单元测试
third_party/    vendored 的 ggml、miniaudio、cpp-httplib
docs/           设计文档与经验记录（GGML 最佳实践、量化、迁移指南等）
```

## 许可证

Apache 2.0，见 [LICENSE](LICENSE)。依赖 ggml（MIT）等第三方库，遵循各自许可。
