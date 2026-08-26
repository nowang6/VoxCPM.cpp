# VoxCPM.cpp

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

基于 [ggml](third_party/ggml) 的 VoxCPM 语音合成（TTS）**纯 C++ 推理引擎**。无 Python、无 PyTorch，编译即得可独立分发的命令行工具与 HTTP 服务，支持 x86 / ARM Linux / **安卓原生** / CUDA / Vulkan。

- 模型下载（GGUF）：https://huggingface.co/bluryar/VoxCPM-GGUF
- VoxCPM 官方仓库：https://github.com/OpenBMB/VoxCPM

## 特性

- **零依赖推理**：ggml 单栈实现 LocEnc / BaseLM / ResidualLM / FSQ / LocDiT / AudioVAE 全流水线
- **多后端**：CPU（含 NEON/dotprod/i8mm 向量化）、CUDA、Vulkan
- **安卓原生运行**：交叉编译产出 bionic 二进制，adb root shell 直接执行，不需要 Termux / proot
- **OpenAI 兼容服务**：`/v1/audio/speech`、音色注册管理，支持 wav/mp3/opus/flac/pcm 与 SSE 流式
- **量化工具链**：`voxcpm_quantize`、`voxcpm_imatrix`（重要性矩阵校准）
- **性能诊断**：`voxcpm_perf` 一键输出 CPU 拓扑、亲和性、量化 matmul 吞吐、最优线程数

## 快速开始

```bash
# 1. 获取代码与模型
git clone <本仓库> && cd VoxCPM.cpp
# 从 HuggingFace 下载 GGUF 模型到 models/ 目录

# 2. 编译（x86_64 Linux / macOS）
cmake -B build
cmake --build build -j

# 3. 合成语音（带参考音色克隆）
./build/examples/voxcpm_tts \
  --model-path ./models/voxcpm-0.5b-q4_k.gguf \
  --prompt-audio ./examples/tai_yi_xian_ren.wav \
  --prompt-text "对，这就是我，万人敬仰的太乙真人。" \
  --text "大家好，这是一段本地合成的语音。" \
  --output ./out.wav \
  --threads 8

# 4. 不带参考音色（默认音色）
./build/examples/voxcpm_tts \
  --model-path ./models/voxcpm-0.5b-q4_k.gguf \
  --text "你好，世界。" --output hello.wav --threads 8
```

依赖：CMake ≥ 3.14、C++17 编译器。`nlohmann/json` 在 `third_party/json` 缺失时由
FetchContent 自动拉取；离线环境可手动放置：

```bash
git clone --depth 1 -b v3.11.3 https://github.com/nlohmann/json third_party/json
```

## 性能调优（重要）

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

### 性能诊断工具 `voxcpm_perf`

```bash
./build/examples/voxcpm_perf              # 拓扑 + 亲和性 + matmul 基准 + 线程扫描
./build/examples/voxcpm_perf --sys-only   # 只看 CPU 簇 / 大核 / 可用亲和性 / cgroup
./build/examples/voxcpm_perf --matmul-only --pin 4,5,6,7   # 绑定指定核心测量
```

工具输出 Q4_K 权重 × F32 激活的 matmul 吞吐（GB/s、GFLOPS）——这是 decode 阶段的核心瓶颈，
可用于快速对比不同线程数 / 绑核策略下的内核吞吐。

### Android 上的 cpuset 陷阱

adb shell 启动的进程常被 Android 归入 background cpuset，**只能用小核**，性能损失 3~5 倍。
`voxcpm_perf` 的系统诊断会直接显示"可用亲和性 (3/8)"之类的警告。root 修复方法：

```bash
echo $$ > /dev/cpuset/top-app/tasks    # 把当前 shell 移到大核组，再重新运行程序
```

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

### CUDA

```bash
cmake -B build-cuda -DVOXCPM_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \    # 30系=86，40系=89，按实际显卡填
  -DVOXCPM_BUILD_BENCHMARK=OFF -DVOXCPM_BUILD_TESTS=OFF
cmake --build build-cuda -j
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

链接配方：`-nostdlib++` + NDK 的 `libc++_static.a` + `libc++abi.a` + `libunwind.a` 全静态。

```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=<aarch64-android 工具链文件> \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOXCPM_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="armv8.6-a+dotprod+i8mm+bf16+fp16" \
  -DVOXCPM_BUILD_TESTS=OFF -DVOXCPM_BUILD_BENCHMARK=OFF
cmake --build build-android -j
```

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

adb shell                          # 进入设备（root shell）
cd /data/local/tmp/voxcpm-android
chmod +x run.sh bin/*
./run.sh "你好，这是原生安卓环境运行的语音合成。" out.wav

# 取回音频（电脑端）
adb pull /data/local/tmp/voxcpm-android/out.wav
```

> 新设备先跑 `LD_LIBRARY_PATH=lib bin/voxcpm_perf` 检查亲和性，若被 cpuset 限制到小核，
> 先执行上面的修复再测性能。

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
