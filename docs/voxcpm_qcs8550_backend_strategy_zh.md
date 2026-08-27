# VoxCPM 在 Qualcomm QCS8550 上的后端部署策略

## 目标平台

Qualcomm Dragonwing QCS8550（面向高端 IoT/边缘 AI 应用）：

| 计算单元 | 规格 | 峰值算力 |
|---------|------|---------|
| CPU | 8 核 Kryo（1×Cortex-X3 3.2GHz + 2×A715 2.8GHz + 2×A710 2.8GHz + 3×A510 2.0GHz）| - |
| GPU | Adreno 740 | FP16 ≈ 12 TOPS |
| NPU/HTP | Hexagon DSP/HTP | INT8 ≈ 48 TOPS |

> 核心结论（2026-08-27 修订）：**硬性约束 = 权重保持 GGUF Q8_0 原样部署，不做任何重新量化**。
> 在此约束下，唯一可用的加速路线是 GGML 原生后端：**大模型（BaseLM/ResidualLM/LocEnc/LocDiT）与 AudioVAE 主线全部走 CPU（Q8_0/保留精度层），GPU（Vulkan/OpenCL）作为有条件的补充，QNN/HTP NPU 路线整体搁置**。

## 前提约束：不重新量化

本文档所有路线均基于以下前提，与前一版本（2026-08-26）的根本区别在此：

1. **模型已定型为 `models/voxcpm-0.5b-q8_0.gguf`（766 MB）**，由 F16 GGUF 经 `voxcpm_quantize --type Q8_0` 产出。
2. **不做任何二次量化或权重格式转换**：不走向 QNN INT8/INT4 重新量化，不从原始 FP16/FP32 权重再出发，也不做 GGUF → DLC/ONNX 转换。
3. **推论**：可用后端 = GGML 中原生支持 Q8_0 GGUF 直接加载的后端，即 **CPU（ARM NEON/dotprod/i8mm）与 GPU（Vulkan；OpenCL 为实验分支）**。CUDA/Metal 与本平台无关；**QNN/HTP 因必然要求重量化而被排除** 。

## 当前代码支持情况

当前 `include/voxcpm/backend.h` 中 `BackendType` 仅支持：

```cpp
enum class BackendType { CPU, CUDA, Metal, Vulkan, Auto };
```

- 已在 x86/ARM CPU、NVIDIA CUDA、Apple Metal 上验证。
- **尚未接入高通 QNN / Hexagon NPU 后端**（在"不重新量化"约束下也已搁置，见下节）。
- GPU 路径两条：`--backend vulkan`（需设备 Vulkan 驱动可用）；实验性 OpenCL 分支（`--backend opencl`，见 `docs/opencl_reboot_minimal_verification_zh.md`）。
- 在 QCS8550 上若无可用 GPU 驱动，直接 fallback 到 CPU。


## VoxCPM 各模块的后端部署建议

### 1. BaseLM（24 层，4096-dim）→ CPU Q8_0

- 模型权重和计算量最大的部分，是解码吞吐的主要决定因素。
- 结构规则，主要为 `matmul + softmax + FFN`，Q8_0 的 `mul_mat` 在 Cortex-X3/A715/A710 上走 dotprod/i8mm 快速路径。
- 线程数设为**大核数**（本机 5：1×X3 + 2×A715 + 2×A710），全开 8 线程反而更慢（小核拖后腿）。
- 参考量级：Q4_K 下实测约 31 token/s；Q8_0 权重带宽约 1.6×，解码吞吐预计下降，**需重测基线**（`voxcpm_perf` + `--seed` 固定输出对比）。

### 2. ResidualLM（8 层，4096-dim）→ CPU Q8_0

- 与 BaseLM 结构类似，同为 Transformer，与 BaseLM 共享同一权重加载路径，无额外工作。
- 无后端切换问题（同在 CPU），无数据搬运开销。

### 3. LocEnc（8 层，1024-dim）+ LocDiT（8 层，1024-dim）→ CPU Q8_0（GPU 为可选）

- 包含局部 attention 和 1D 卷积；CFM 采样步数多（`--inference-timesteps`），计算密集。
- 主线走 CPU；若 GPU 路线（Vulkan/OpenCL）通过稳定性验证，LocDiT 的 CFM 循环是 GPU 收益的第二候选。
- 提速手段（已内置、无需量化改动）：`--inference-timesteps` 从 10 降到 6，CFM 阶段提速约 40%，音质略降。

### 4. FSQ 量化器 / proj 投影层 → CPU

- 计算量小，多为 `argmax`、`reshape`、`gather`、矩阵投影。
- 留在 CPU，避免任何后端间数据搬运；对整体耗时影响极小。

### 5. AudioVAE Encoder（prompt/reference 音频编码）→ CPU 保留精度层

- 以卷积和下采样为主；GGUF 中 AudioVAE 敏感层按量化策略保留 f16/f32（见 `docs/quantization-log-20260827.md` 的保留策略说明）。
- 仅在启用 prompt/reference audio 时触发；一次克隆会话内只需执行一次，非热路径。

### 6. AudioVAE Decoder（最终波形生成）→ CPU 为主，GPU 为有条件选项

- CPU 上的第二大瓶颈（Q4_K 时代约占总时间 30%，Q8_0 下待重测）。
- 以转置卷积 + 残差块为主，结构上对 GPU 友好——但 GPU 路线受整机稳定性约束（见下）。
- **GPU 路线现状**（详见 `docs/opencl_reboot_minimal_verification_zh.md`，2026-08-27）：OpenCL 栈曾致整机静默复位，两轮最小化验证（17+ 次触碰、含 89 kernel 编译、256MB 级分配、120 次 init/teardown、密集 soak）**未能复现**；当前定位为"探针级可用"，**全栈 `--backend opencl` 推理仍需串口在录才能试**，生产推理暂不依赖 GPU。
- 若后续串口在录下的全栈回归通过，AudioVAE decode 是 GPU 上的第一优先目标（预期 3~4 秒级 → 1 秒级）。

### 7. 预处理 / 后处理 → CPU

- 文本 tokenizer、音频重采样、VAD、WAV 读写、patch 拼接。
- 控制流复杂、数据量小，CPU 最合适。

## 推荐的端到端流水线（CPU 主线）

```text
输入文本
   │
   ▼
CPU: 文本 tokenizer / 文本编码
   │
   ▼
CPU: 构造 full_text_tokens / text_mask / feat_mask / feat
   │
   ▼
CPU: AudioVAE Encoder（保留精度层，仅 prompt 克隆时）
   │
   ▼
CPU Q8_0: BaseLM prefill + decode loop
          生成 FSQ tokens / latent steps
   │
   ▼
CPU Q8_0: proj + LocDiT CFM（条件流匹配，--inference-timesteps 可降）
   │
   ▼
CPU: FSQ 反量化 + patch 拼接
   │
   ▼
CPU: AudioVAE decode ──► 波形        ┐
   │                                  ├─ GPU 版本为有条件选项（需通过
   ▼                                  │  串口在录的全栈回归，见 opencl 文档）
CPU: 写 WAV 文件                     ┘
```


## 后端与量化策略速查表

| 模块 | 推荐后端 | 精度（现 GGUF 内实际格式） | Q8_0 GGUF 直接可用 | 理由 |
|-----|---------|---------|-----------------|------|
| BaseLM | CPU | Q8_0 | ✅ | dotprod/i8mm 快速路径；NPU 路线搁置 |
| ResidualLM | CPU | Q8_0 | ✅ | 与 BaseLM 同路径，无后端切换 |
| LocEnc | CPU | Q8_0 | ✅ | 计算量小，留在 CPU |
| LocDiT | CPU（GPU 可选） | Q8_0 | ✅ | CFM 步数多；`--inference-timesteps` 可降 |
| FSQ / proj | CPU | f32/f16 | ✅ | 计算量小，避免后端切换开销 |
| AudioVAE Encoder | CPU | f16/f32（策略保留） | ✅ | 仅 prompt 克隆时执行一次 |
| AudioVAE Decoder | CPU（GPU 有条件） | f16/f32（策略保留） | ✅ | GPU 路线待稳定性回归 |
| 预处理/后处理 | CPU | FP32 | ✅ 不涉及权重 | 控制流复杂、数据量小 |