# Local Dream Flux2 - 构建与部署指南

将 FLUX.2-klein-4B 文本生成图片模型运行在 Android 手机上的完整前端 App。

## 项目架构

```
┌────────────────────────────────────────┐
│  Kotlin UI (Jetpack Compose)           │
│  ModelListScreen → 选择/下载模型        │
│  ModelRunScreen  → 输入 prompt，生成图片 │
│           │                            │
│           │ OkHttp (localhost:8081)     │
│           ▼                            │
│  C++ Backend (libstable_diffusion_core)│
│  HTTP Server → ExecuTorch 推理         │
│  Text Encoder → Transformer → VAE      │
└────────────────────────────────────────┘
```

App 采用双进程设计：
- **Kotlin 层**：Jetpack Compose UI，处理用户交互
- **C++ 层**：`main_flux2.cpp` 编译为 `libstable_diffusion_core.so`，作为独立进程运行 HTTP 服务器

推理流水线：Qwen3 Text Encoder → Flux2 MMDiT Transformer (4步去噪) → VAE Decoder → 图片

## 前置条件

| 依赖 | 版本要求 | 安装方式 |
|------|---------|---------|
| **CMake** | 3.18+ | `brew install cmake` |
| **Ninja** | any | `brew install ninja` |
| **Rust** | 1.84.0 | 见下方 |
| **Java (JDK)** | 17+ | `brew install openjdk@17` |
| **Android NDK** | r28c | [下载](https://developer.android.com/ndk/downloads) |
| **Android SDK** | API 36 | 见下方 |
| **ExecuTorch** | — | 需提前为 Android 构建 |

## Step 0: 安装依赖

### Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default 1.84.0
rustup target add aarch64-linux-android
```

> 必须使用 Rust 1.84.0，更新版本可能导致构建失败。

### Java 17

```bash
brew install openjdk@17
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
echo 'export JAVA_HOME=/opt/homebrew/opt/openjdk@17' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Android SDK (无需 Android Studio)

```bash
cd ~
curl -O https://dl.google.com/android/repository/commandlinetools-mac-11076708_latest.zip
mkdir -p ~/Library/Android/sdk/cmdline-tools
unzip commandlinetools-mac-11076708_latest.zip -d ~/Library/Android/sdk/cmdline-tools
mv ~/Library/Android/sdk/cmdline-tools/cmdline-tools ~/Library/Android/sdk/cmdline-tools/latest
rm commandlinetools-mac-11076708_latest.zip

echo 'export ANDROID_HOME=~/Library/Android/sdk' >> ~/.zshrc
echo 'export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"' >> ~/.zshrc
source ~/.zshrc

yes | sdkmanager --licenses
sdkmanager "platforms;android-36" "build-tools;36.0.0" "platform-tools"
```

## Step 1: 构建 ExecuTorch for Android

```bash
export ANDROID_NDK=~/android-ndk-r28c
cd <executorch-root>

rm -rf cmake-out-android

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF \
    -DEXECUTORCH_BUILD_COREML=OFF \
    -DEXECUTORCH_BUILD_MPS=OFF \
    -Bcmake-out-android .

cmake --build cmake-out-android -j16 --target install --config Release
```

构建完成后会在 `cmake-out-android/` 下生成所需的 `.a` 静态库。

## Step 2: 配置路径

编辑 `app/src/main/cpp/CMakePresets.json`，修改 `environment` 部分：

```json
"environment": {
    "ANDROID_NDK_ROOT": "<你的 NDK 路径，如 /Users/xxx/android-ndk-r28c>",
    "EXECUTORCH_ROOT": "<你的 ExecuTorch 路径>/cmake-out-android"
}
```

> 注意：`EXECUTORCH_ROOT` 指向的是 ExecuTorch 的**构建输出目录** `cmake-out-android`，不是源码根目录。

## Step 3: 构建 C++ 原生库

```bash
cd app/src/main/cpp/
git submodule update --init --recursive
bash ./build.sh
```

构建成功后会自动将 `libstable_diffusion_core.so` 复制到 `app/src/main/jniLibs/arm64-v8a/`。

## Step 4: 构建 APK

```bash
cd <项目根目录>
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
./gradlew assembleBasicDebug
```

首次运行会自动下载 Gradle 和所需的 Android SDK 组件。构建成功后 APK 位于：

```
app/build/outputs/apk/basic/debug/LocalDream_armv8a_2.3.2.apk
```

## Step 5: 安装到手机

用 USB 连接手机并开启 USB 调试，然后：

```bash
adb install app/build/outputs/apk/basic/debug/LocalDream_armv8a_2.3.2.apk
```

如果手机上已有同名但签名不同的旧版本，需要先卸载：

```bash
adb uninstall io.github.xororz.localdream
adb install app/build/outputs/apk/basic/debug/LocalDream_armv8a_2.3.2.apk
```

## 使用方法

1. 打开 App → 进入 **Model List** 页面
2. 下载 FLUX.2-klein 模型（约 4.5GB，需联网）或手动 push 模型文件
3. 点击模型 → 进入 **Model Run** 页面
4. 输入 prompt（如 "a cat sitting on a windowsill at sunset"）
5. 点击 **Generate**，等待 4 步去噪完成
6. 查看生成结果，可保存到相册

### 手动 push 模型（可选）

如果在 App 内下载不便，可以手动将导出的模型文件 push 到手机：

```bash
adb shell mkdir -p /data/local/tmp/localdream/models/flux2klein
adb push text_encoder.pte /data/local/tmp/localdream/models/flux2klein/
adb push transformer.pte  /data/local/tmp/localdream/models/flux2klein/
adb push vae_decoder.pte  /data/local/tmp/localdream/models/flux2klein/
adb push export_config.json /data/local/tmp/localdream/models/flux2klein/
adb push vae_bn_stats.json /data/local/tmp/localdream/models/flux2klein/
adb push tokenizer/ /data/local/tmp/localdream/models/flux2klein/tokenizer/
```

## 硬件要求

- **ARM64 Android 手机**（arm64-v8a）
- **16GB+ RAM**（推荐，模型较大）
- **Android 9+**（minSdk 28）

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| `CMAKE_MAKE_PROGRAM is not set` / 找不到 Ninja | `brew install ninja` |
| `cargo: no such file or directory` | 安装 Rust 并添加 Android target |
| `SDK location not found` | 创建 `local.properties` 写入 `sdk.dir=...` |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | 先 `adb uninstall io.github.xororz.localdream` 再安装 |
| ExecuTorch `.a` 文件找不到 | 确认已完成 Step 1 构建，且 `EXECUTORCH_ROOT` 指向 `cmake-out-android` |
