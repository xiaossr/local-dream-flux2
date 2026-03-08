# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local Dream is an Android app for on-device Stable Diffusion image generation. It supports NPU acceleration (Qualcomm Snapdragon via QNN SDK), CPU inference (via MNN framework), and GPU inference. The app supports txt2img, img2img, inpainting, and upscaling.

## Build Commands

### Native C++ Libraries (must be built first)
```bash
cd app/src/main/cpp/
# Set QNN_SDK_ROOT in CMakeLists.txt and ANDROID_NDK_ROOT in CMakePresets.json first
bash ./build.sh
```
This builds `libstable_diffusion_core.so` and copies QNN runtime libs to assets.

### Android APK
Build via Android Studio: Build → Generate App Bundles or APKs → Generate APKs

Or via Gradle:
```bash
./gradlew assembleBasicDebug    # basic flavor, debug
./gradlew assembleFilterDebug   # filter flavor (NSFW filter), debug
./gradlew assembleBasicRelease  # basic flavor, release (needs signing config)
```

### Build Prerequisites
- Rust 1.84.0 with `aarch64-linux-android` target
- Ninja, CMake
- QNN SDK 2.39, Android NDK

## Architecture

### Two-Process Design
The app runs inference in a **separate native process**. The C++ backend (`main.cpp`) starts an HTTP server on `127.0.0.1:8081` that the Kotlin frontend communicates with via localhost HTTP requests (using OkHttp).

### Kotlin Layer (`app/src/main/java/io/github/xororz/localdream/`)
- **`MainActivity.kt`** — Entry point; sets up Jetpack Compose NavHost with three screens
- **`navigation/Navigation.kt`** — Route definitions: `ModelList`, `ModelRun/{modelId}`, `Upscale`
- **`ui/screens/`** — Compose UI screens:
  - `ModelListScreen.kt` — Model selection, download management
  - `ModelRunScreen.kt` — Main generation UI (txt2img/img2img parameter controls, generation progress)
  - `UpscaleScreen.kt` — Image upscaling UI
  - `InpaintScreen.kt`, `CropImageScreen.kt` — Inpainting and image cropping
- **`data/`**:
  - `Model.kt` — Model definitions, `ModelRepository` (model registry with download URLs), `PatchScanner` (resolution detection from .patch files)
  - `Preferences.kt` — DataStore-based user preferences
  - `HistoryManager.kt` — Generation history persistence
- **`service/`**:
  - `BackendService.kt` — Foreground service managing the native C++ process lifecycle; launches `libstable_diffusion_core.so` as a subprocess
  - `BackgroundGenerationService.kt` — Handles generation when app is in background
  - `ModelDownloadService.kt` — Foreground service for model downloads
- **`utils/ImageUtils.kt`** — Image processing utilities

### C++ Layer (`app/src/main/cpp/`)
- **`src/main.cpp`** — HTTP server (cpp-httplib) handling inference requests; orchestrates the full SD pipeline
- **`src/QnnModel.hpp`** — QNN SDK model loading/execution for NPU
- **`src/SafeTensor2MNN.hpp`** — Converts SafeTensor weights to MNN format for CPU inference
- **`src/PromptProcessor.hpp`** — CLIP text tokenization and encoding
- **`src/Scheduler.hpp`** — Base scheduler interface
- **`src/DPMSolverMultistepScheduler.hpp`**, **`src/EulerAncestralDiscreteScheduler.hpp`** — Diffusion schedulers
- **`src/SDStructure.hpp`**, **`src/SDUtils.hpp`** — SD pipeline data structures and utilities
- **`3rdparty/`** — Vendored dependencies (QNN SampleApp, patched via `SampleApp.patch`)

### Product Flavors
- **basic** — Standard build
- **filter** — Includes NSFW safety checker

### Key Technical Details
- Target: arm64-v8a only (aarch64 Android)
- Min SDK 28, Target SDK 36
- C++17, Java 17
- NPU models use W8A16 static quantization at fixed 512×512; CPU models use W8 dynamic quantization with flexible resolutions
- Model resolution variants detected via `.patch` files in model directories
