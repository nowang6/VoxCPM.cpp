#!/usr/bin/env bash
# 打包 Android 交叉编译产物为可直接 adb push 的自包含目录（不含模型权重）。
#
# 用法:
#   scripts/package_android.sh                       # 默认 build-android -> voxcpm-android
#   scripts/package_android.sh -b build-android -o /tmp/voxcpm-android
#   scripts/package_android.sh --push                # 打包后直接 adb push 到已连接设备
#
# 产物布局:
#   <out>/run.sh          mksh 兼容启动脚本（设 LD_LIBRARY_PATH，VOXCPM_BIN 可切换工具）
#   <out>/bin/*           voxcpm_tts / voxcpm-server / voxcpm_quantize / voxcpm_imatrix / voxcpm_perf
#   <out>/lib/*           libggml.so / libggml-base.so / libggml-cpu.so（SONAME 无版本号）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-android"
STAGE_DIR="${ROOT_DIR}/voxcpm-android"
DEVICE_DIR="/data/local/tmp/voxcpm-android"
DO_PUSH=0

BINS=(voxcpm_tts voxcpm-server voxcpm_quantize voxcpm_imatrix voxcpm_perf)
LIBS=(libggml.so libggml-base.so libggml-cpu.so)

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build)  BUILD_DIR="$2"; shift 2 ;;
    -o|--output) STAGE_DIR="$2"; shift 2 ;;
    --push)      DO_PUSH=1; shift ;;
    -h|--help)   usage ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done

[[ -d "$BUILD_DIR" ]] || { echo "错误: 构建目录不存在: $BUILD_DIR（先按 README 的 Android 章节交叉编译）" >&2; exit 1; }

# 产物齐全性检查：缺哪个列哪个，避免推到设备上才发现
MISSING=0
for f in "${BINS[@]}"; do
  [[ -f "$BUILD_DIR/examples/$f" ]] || { echo "缺少可执行文件: $BUILD_DIR/examples/$f" >&2; MISSING=1; }
done
for f in "${LIBS[@]}"; do
  [[ -f "$BUILD_DIR/ggml/src/$f" ]] || { echo "缺少动态库: $BUILD_DIR/ggml/src/$f" >&2; MISSING=1; }
done
[[ $MISSING -eq 0 ]] || exit 1

# 探测 NDK llvm-strip（宿主 GNU strip 处理不了 aarch64 ELF）
STRIP=""
for cand in \
  "${ANDROID_NDK_ROOT:-}" "${NDK_ROOT:-}" "$HOME/android-ndk-r27d" "$HOME/Android/Sdk/ndk"/*; do
  [[ -n "$cand" ]] || continue
  s="$cand/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
  if [[ -x "$s" ]]; then STRIP="$s"; break; fi
done
if [[ -z "$STRIP" ]]; then
  echo "警告: 未找到 NDK llvm-strip，产物保留 debug_info（体积约大 10 倍）" >&2
fi

echo "==> 暂存目录: $STAGE_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/bin" "$STAGE_DIR/lib"

for f in "${BINS[@]}"; do cp "$BUILD_DIR/examples/$f" "$STAGE_DIR/bin/"; done
for f in "${LIBS[@]}"; do cp "$BUILD_DIR/ggml/src/$f" "$STAGE_DIR/lib/"; done

if [[ -n "$STRIP" ]]; then
  echo "==> strip: $STRIP"
  "$STRIP" "$STAGE_DIR"/bin/* "$STAGE_DIR"/lib/*
fi

# 依赖自检：so 若仍依赖设备上不存在的库（如 libomp.so），推上设备才报
# CANNOT LINK EXECUTABLE——在这里提前拦下
READELF="${STRIP%llvm-strip}llvm-readelf"
if [[ -x "$READELF" ]]; then
  BAD_DEP=$("$READELF" -d "$STAGE_DIR"/lib/*.so 2>/dev/null \
    | awk '/Shared library/ {print $NF}' | tr -d '[]' \
    | grep -vE '^(libggml|libggml-base|libggml-cpu|libc|libm|libdl|liblog|libz|libc\+\+_shared)\.so$' || true)
  if [[ -n "$BAD_DEP" ]]; then
    echo "错误: 产物依赖设备上不存在的库: $BAD_DEP" >&2
    echo "      （OpenMP 依赖请用 -DGGML_OPENMP=OFF 重编，CMakeLists 在 Android 下已默认关闭）" >&2
    exit 1
  fi
fi

cat > "$STAGE_DIR/run.sh" <<'EOF'
#!/system/bin/sh
# mksh 兼容启动脚本。默认调 voxcpm_tts，切工具: VOXCPM_BIN=voxcpm-server ./run.sh ...
DIR=$(cd "$(dirname "$0")" && pwd)
LD_LIBRARY_PATH="$DIR/lib" exec "$DIR/bin/${VOXCPM_BIN:-voxcpm_tts}" "$@"
EOF
chmod +x "$STAGE_DIR/run.sh" "$STAGE_DIR"/bin/*

echo "==> 打包完成: $(du -sh "$STAGE_DIR" | cut -f1)"
find "$STAGE_DIR/bin" "$STAGE_DIR/lib" -type f -exec ls -lh {} + | awk '{print "    " $5 "\t" $9}'

if [[ $DO_PUSH -eq 1 ]]; then
  echo "==> adb push -> $DEVICE_DIR"
  adb push "$STAGE_DIR" "$DEVICE_DIR"
  echo "==> 设备端用法:"
  echo "    adb shell && cd $DEVICE_DIR"
  echo "    LD_LIBRARY_PATH=lib bin/voxcpm_perf --sys-only   # 先查拓扑/亲和性"
  echo "    ./run.sh --model-path ./<model>.gguf --text \"你好\" --output out.wav --threads 4"
else
  echo "==> 推送: adb push $STAGE_DIR $DEVICE_DIR"
  echo "    模型权重单独推: adb push models/<model>.gguf $DEVICE_DIR/"
fi
