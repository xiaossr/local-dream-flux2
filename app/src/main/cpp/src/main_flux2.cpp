// FLUX.2-klein ExecuTorch inference server
//
// Pipeline: Qwen3 text encoder → FlowMatch DiT (4 steps) → VAE decoder
// Models are loaded/unloaded per stage to minimize peak memory.
// HTTP server preserves the same SSE streaming API as the original backend.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ExecuTorch
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

// External Libraries
#include "httplib.h"
#include "json.hpp"
#include "tokenizers_cpp.h"

namespace et = executorch::extension;
namespace rt = executorch::runtime;
using et::Module;
using et::from_blob;
using rt::EValue;
using rt::Error;
using executorch::aten::ScalarType;

// ============================================================================
// Inline base64 encoder / decoder
// ============================================================================

static inline std::string base64_encode(const std::string &in) {
  static const auto lookup =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  unsigned int val = 0;
  int valb = -6;
  for (auto c : in) {
    val = (val << 8) + static_cast<uint8_t>(c);
    valb += 8;
    while (valb >= 0) {
      out.push_back(lookup[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6)
    out.push_back(lookup[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

static inline std::string base64_decode(const std::string &in) {
  static const std::string chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  static int lookup[256] = {0};
  static bool init = false;
  if (!init) {
    std::fill(lookup, lookup + 256, -1);
    for (int i = 0; i < 64; i++) lookup[(unsigned char)chars[i]] = i;
    init = true;
  }
  std::string out;
  int val = 0, bits = -8;
  for (unsigned char c : in) {
    if (lookup[c] == -1) continue;
    val = (val << 6) + lookup[c];
    bits += 6;
    if (bits >= 0) {
      out.push_back(char((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

// ============================================================================
// Global configuration
// ============================================================================

struct ExportConfig {
  int height = 512;
  int width = 512;
  int max_text_len = 512;
  int num_inference_steps = 4;
  float guidance_scale = 1.0f;
  bool is_distilled = true;
  int vae_scale_factor = 8;

  // Transformer
  int in_channels = 128;     // post-patchify channels
  int out_channels = 128;
  int joint_attention_dim = 15360;

  // VAE
  int latent_channels = 32;
  float scaling_factor = 1.0f;
  float shift_factor = 0.0f;
  float batch_norm_eps = 1e-5f;

  // Patch dims (post-patchify spatial)
  int patch_h = 0;
  int patch_w = 0;

  // img2img
  int num_img2img_images = 0;

  // VAE batch-norm stats
  std::vector<float> bn_running_mean;
  std::vector<float> bn_running_var;
};

static int g_port = 8081;
static std::string g_listen_address = "127.0.0.1";

static std::string g_encoder_path;
static std::string g_dit_path;
static std::string g_dit_img2img_path;
static std::string g_vae_decoder_path;
static std::string g_vae_encoder_path;
static std::string g_tokenizer_path;
static std::string g_config_path;
static std::string g_bn_stats_path;

static ExportConfig g_config;

// Only the tokenizer stays loaded globally (tiny memory footprint)
static std::shared_ptr<tokenizers::Tokenizer> g_tokenizer;

// Per-request state
static std::string g_prompt;
static unsigned g_seed = 0;
static int g_output_width = 512;
static int g_output_height = 512;
static std::string g_input_image_b64;
static float g_denoise_strength = 1.0f;

// ============================================================================
// Utility
// ============================================================================

static double now_sec() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static unsigned hashSeed(unsigned long long v) {
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<unsigned>(v ^ (v >> 31));
}

static std::string loadFileToString(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open file: " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

static nlohmann::json loadJsonFile(const std::string &path) {
  auto content = loadFileToString(path);
  return nlohmann::json::parse(content);
}

// ============================================================================
// Config loading
// ============================================================================

static void loadExportConfig(const std::string &config_path) {
  auto j = loadJsonFile(config_path);

  g_config.height = j.value("height", 512);
  g_config.width = j.value("width", 512);
  g_config.max_text_len = j.value("max_text_len", 512);
  g_config.num_inference_steps = j.value("num_inference_steps", 4);
  g_config.guidance_scale = j.value("guidance_scale", 1.0f);
  g_config.is_distilled = j.value("is_distilled", true);
  g_config.vae_scale_factor = j.value("vae_scale_factor", 8);

  if (j.contains("patch_dims")) {
    auto pd = j["patch_dims"];
    g_config.patch_h = pd[0].get<int>();
    g_config.patch_w = pd[1].get<int>();
  } else {
    int vae_sf = g_config.vae_scale_factor;
    g_config.patch_h = (2 * (g_config.height / (vae_sf * 2))) / 2;
    g_config.patch_w = (2 * (g_config.width / (vae_sf * 2))) / 2;
  }

  if (j.contains("transformer")) {
    auto &t = j["transformer"];
    g_config.in_channels = t.value("in_channels", 128);
    g_config.out_channels = t.value("out_channels", 128);
    g_config.joint_attention_dim = t.value("joint_attention_dim", 15360);
  }

  if (j.contains("vae")) {
    auto &v = j["vae"];
    g_config.latent_channels = v.value("latent_channels", 32);
    g_config.scaling_factor = (v.contains("scaling_factor") && !v["scaling_factor"].is_null())
        ? v["scaling_factor"].get<float>() : 0.3611f;
    g_config.shift_factor = (v.contains("shift_factor") && !v["shift_factor"].is_null())
        ? v["shift_factor"].get<float>() : 0.1159f;
    g_config.batch_norm_eps = v.value("batch_norm_eps", 1e-5f);
  }

  g_config.num_img2img_images = j.value("num_img2img_images", 0);

  std::cout << "Config loaded: " << g_config.height << "x" << g_config.width
            << ", steps=" << g_config.num_inference_steps
            << ", patch=" << g_config.patch_h << "x" << g_config.patch_w
            << std::endl;
}

static void loadBNStats(const std::string &path) {
  int nc = g_config.latent_channels;
  g_config.bn_running_mean.resize(nc, 0.0f);
  g_config.bn_running_var.resize(nc, 1.0f);

  if (path.empty() || !std::filesystem::exists(path)) {
    std::cerr << "WARN: No BN stats file, using identity transform\n";
    return;
  }

  // Try JSON format first
  try {
    auto j = loadJsonFile(path);
    if (j.contains("running_mean")) {
      auto &m = j["running_mean"];
      for (int i = 0; i < nc && i < (int)m.size(); ++i)
        g_config.bn_running_mean[i] = m[i].get<float>();
    }
    if (j.contains("running_var")) {
      auto &v = j["running_var"];
      for (int i = 0; i < nc && i < (int)v.size(); ++i)
        g_config.bn_running_var[i] = v[i].get<float>();
    }
    std::cout << "BN stats loaded from JSON: " << path << std::endl;
    return;
  } catch (...) {}

  // Fall back to raw binary: [nc floats mean][nc floats var]
  std::ifstream f(path, std::ios::binary);
  if (f) {
    f.read(reinterpret_cast<char *>(g_config.bn_running_mean.data()),
           nc * sizeof(float));
    f.read(reinterpret_cast<char *>(g_config.bn_running_var.data()),
           nc * sizeof(float));
    std::cout << "BN stats loaded from binary: " << path << std::endl;
  }
}

// ============================================================================
// Sigma schedule (FlowMatchEulerDiscreteScheduler with dynamic time-shift)
// Matches diffusers / reference pipeline exactly.
// ============================================================================

static float compute_mu(int image_seq_len, int num_steps) {
  constexpr float a1 = 8.73809524e-05f, b1 = 1.89833333f;
  constexpr float a2 = 0.00016927f, b2 = 0.45666666f;
  if (image_seq_len > 4300)
    return a2 * image_seq_len + b2;
  float m_200 = a2 * image_seq_len + b2;
  float m_10 = a1 * image_seq_len + b1;
  float a = (m_200 - m_10) / 190.0f;
  float b = m_200 - 200.0f * a;
  return a * num_steps + b;
}

static std::vector<float> build_sigmas(int num_steps, int image_seq_len) {
  std::vector<float> sigmas(num_steps);
  if (num_steps == 1) {
    sigmas[0] = 1.0f;
  } else {
    float end = 1.0f / num_steps;
    for (int i = 0; i < num_steps; i++)
      sigmas[i] = 1.0f + (float)i * (end - 1.0f) / (num_steps - 1);
  }
  float mu = compute_mu(image_seq_len, num_steps);
  float emu = expf(mu);
  for (auto &s : sigmas)
    s = emu / (emu + (1.0f / s - 1.0f));
  sigmas.push_back(0.0f);
  return sigmas;
}

// ============================================================================
// Positional ID helpers (4-D coordinates: T, H, W, L)
// ============================================================================

static std::vector<float> make_img_ids(int patch_h, int patch_w) {
  int N = patch_h * patch_w;
  std::vector<float> ids(N * 4, 0.0f);
  int idx = 0;
  for (int h = 0; h < patch_h; h++) {
    for (int w = 0; w < patch_w; w++) {
      ids[idx * 4 + 1] = static_cast<float>(h);
      ids[idx * 4 + 2] = static_cast<float>(w);
      idx++;
    }
  }
  return ids;
}

static std::vector<float> make_txt_ids(int seq_len) {
  std::vector<float> ids(seq_len * 4, 0.0f);
  for (int s = 0; s < seq_len; s++)
    ids[s * 4 + 3] = static_cast<float>(s);
  return ids;
}

// Build reference image positional IDs with T-offset (for img2img)
static std::vector<float> make_ref_img_ids(int patch_h, int patch_w,
                                            int img_index) {
  int t_offset = 10 + 10 * img_index;
  int N = patch_h * patch_w;
  std::vector<float> ids(N * 4, 0.0f);
  int idx = 0;
  for (int h = 0; h < patch_h; h++) {
    for (int w = 0; w < patch_w; w++) {
      ids[idx * 4 + 0] = static_cast<float>(t_offset);
      ids[idx * 4 + 1] = static_cast<float>(h);
      ids[idx * 4 + 2] = static_cast<float>(w);
      idx++;
    }
  }
  return ids;
}

// ============================================================================
// Latent packing / unpacking / unpatchify / BN
// ============================================================================

// (C, H, W) → (H*W, C) — pack noise into token layout
static std::vector<float> pack_chw_to_nc(
    const std::vector<float> &src, int C, int H, int W) {
  std::vector<float> dst(H * W * C);
  for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++)
        dst[(h * W + w) * C + c] = src[c * H * W + h * W + w];
  return dst;
}

// (N, C) → (C, H, W) using H/W from the positional IDs
static std::vector<float> unpack_nc_to_chw(
    const float *data, const float *ids, int N, int C, int H, int W) {
  std::vector<float> out(C * H * W, 0.0f);
  for (int i = 0; i < N; i++) {
    int h = static_cast<int>(ids[i * 4 + 1]);
    int w = static_cast<int>(ids[i * 4 + 2]);
    for (int c = 0; c < C; c++)
      out[c * H * W + h * W + w] = data[i * C + c];
  }
  return out;
}

// x = x * sqrt(var + eps) + mean (per-channel, CHW layout)
static void bn_unnormalize(
    float *data, const float *mean, const float *var,
    int C, int H, int W, float eps) {
  for (int c = 0; c < C; c++) {
    float s = sqrtf(var[c] + eps);
    float m = mean[c];
    float *ch = data + c * H * W;
    for (int i = 0; i < H * W; i++)
      ch[i] = ch[i] * s + m;
  }
}

// Forward batch-norm: x = (x - mean) / sqrt(var + eps) (for img2img ref encode)
static void bn_normalize(
    float *data, const float *mean, const float *var,
    int C, int H, int W, float eps) {
  for (int c = 0; c < C; c++) {
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float m = mean[c];
    float *ch = data + c * H * W;
    for (int i = 0; i < H * W; i++)
      ch[i] = (ch[i] - m) * inv_std;
  }
}

// (C*4, H2, W2) → (C, H2*2, W2*2)  — undo 2x2 patchification
static std::vector<float> unpatchify(
    const float *src, int C4, int H2, int W2) {
  int C = C4 / 4;
  int H = H2 * 2;
  int W = W2 * 2;
  std::vector<float> dst(C * H * W);
  for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++) {
        int src_c = c * 4 + (h % 2) * 2 + (w % 2);
        dst[c * H * W + h * W + w] =
            src[src_c * H2 * W2 + (h / 2) * W2 + (w / 2)];
      }
  return dst;
}

// (C, H2*2, W2*2) → (C*4, H2, W2)  — 2x2 patchification (for img2img)
static std::vector<float> patchify_chw(
    const float *src, int C, int H, int W) {
  int H2 = H / 2;
  int W2 = W / 2;
  int C4 = C * 4;
  std::vector<float> dst(C4 * H2 * W2);
  for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++) {
        int dst_c = c * 4 + (h % 2) * 2 + (w % 2);
        dst[dst_c * H2 * W2 + (h / 2) * W2 + (w / 2)] =
            src[c * H * W + h * W + w];
      }
  return dst;
}

// ============================================================================
// VAE post-processing
// ============================================================================

// Apply VAE scaling: latents = (latents - shift) / scaling
static void vaeScaleLatents(std::vector<float> &latents) {
  float sf = g_config.scaling_factor;
  float sh = g_config.shift_factor;
  if (sf == 0.0f) sf = 1.0f;
  for (auto &v : latents) {
    v = (v - sh) / sf;
  }
}

// Convert float pixels [-1,1] to uint8 RGB
static std::vector<uint8_t> pixelsToRGB(const float *data, int height,
                                        int width) {
  std::vector<uint8_t> rgb(height * width * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        float val = data[c * height * width + y * width + x];
        val = std::clamp((val + 1.0f) * 0.5f, 0.0f, 1.0f);
        rgb[(y * width + x) * 3 + c] = static_cast<uint8_t>(val * 255.0f);
      }
    }
  }
  return rgb;
}

// ============================================================================
// Image input helpers (img2img)
// ============================================================================

static std::vector<float> rgbToFloat(const uint8_t *data, int height,
                                     int width) {
  std::vector<float> chw(3 * height * width);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        float v = data[(y * width + x) * 3 + c] / 255.0f;
        chw[c * height * width + y * width + x] = v * 2.0f - 1.0f;
      }
    }
  }
  return chw;
}

// ============================================================================
// Generation result
// ============================================================================

struct GenerationResult {
  std::vector<uint8_t> image_data;  // raw RGB bytes
  int width = 0;
  int height = 0;
  int channels = 3;
  long long generation_time_ms = 0;
  long long first_step_time_ms = 0;
};

// ============================================================================
// Image generation pipeline
// Models are loaded and released per-stage to keep peak memory to ~1 model.
// ============================================================================

static GenerationResult generateImage(
    std::function<void(int step, int total_steps, const std::string &img)>
        progress_cb) {
  auto pipeline_start = std::chrono::high_resolution_clock::now();

  int ph = g_config.patch_h;
  int pw = g_config.patch_w;
  int latent_h = ph * 2;
  int latent_w = pw * 2;
  int latent_ch = g_config.latent_channels;
  int in_ch = g_config.in_channels;
  int num_tokens = ph * pw;
  int max_text_len = g_config.max_text_len;
  int num_steps = g_config.num_inference_steps;
  bool has_bn = !g_config.bn_running_mean.empty();

  // Determine if this is an img2img request
  bool is_img2img = !g_input_image_b64.empty() &&
                    !g_vae_encoder_path.empty() &&
                    !g_dit_img2img_path.empty() &&
                    g_config.num_img2img_images > 0;

  // ======================= 1. TEXT ENCODING ================================
  // Load text encoder, run it, then release to free memory.
  std::cout << "\n[1/4] Tokenizing + text encoding..." << std::endl;
  fflush(stdout);
  double t0 = now_sec();

  // Apply Qwen3 chat template before tokenizing
  std::string chat_prompt = "<|im_start|>user\n" + g_prompt +
                            "<|im_end|>\n<|im_start|>assistant\n";

  auto encoded = g_tokenizer->Encode(chat_prompt);
  std::vector<int64_t> input_ids(max_text_len, 0);
  std::vector<int64_t> attention_mask(max_text_len, 0);
  int token_count = std::min((int)encoded.size(), max_text_len);
  for (int i = 0; i < token_count; ++i) {
    input_ids[i] = encoded[i];
    attention_mask[i] = 1;
  }
  std::cout << "  tokens: " << token_count << "/" << max_text_len << std::endl;

  std::vector<float> prompt_embeds;
  int joint_dim = 0;
  {
    Module text_encoder(g_encoder_path);
    std::cout << "  text_encoder loaded (" << (now_sec() - t0) << " s)"
              << std::endl;
    fflush(stdout);

    auto ids_tp = from_blob(input_ids.data(), {1, max_text_len}, ScalarType::Long);
    auto mask_tp = from_blob(attention_mask.data(), {1, max_text_len}, ScalarType::Long);

    std::vector<EValue> te_inputs;
    te_inputs.push_back(*ids_tp);
    te_inputs.push_back(*mask_tp);

    std::cout << "  running text encoder..." << std::endl;
    fflush(stdout);
    t0 = now_sec();
    auto te_res = text_encoder.execute("forward", te_inputs);
    if (!te_res.ok()) {
      std::cerr << "  text_encoder.execute failed with error code: "
                << static_cast<int>(te_res.error()) << std::endl;
      fflush(stderr);
      throw std::runtime_error("Text encoder forward failed");
    }
    const auto &te_tensor = (*te_res)[0].toTensor();
    prompt_embeds.assign(
        te_tensor.const_data_ptr<float>(),
        te_tensor.const_data_ptr<float>() + te_tensor.numel());
    joint_dim = static_cast<int>(te_tensor.numel()) / max_text_len;
    std::cout << "  prompt_embeds: " << max_text_len << " x " << joint_dim
              << " (" << (now_sec() - t0) << " s)" << std::endl;
    fflush(stdout);
  }
  // text_encoder is destroyed here — memory freed
  std::cout << "  text encoder released" << std::endl;
  fflush(stdout);

  // ======================= 2. NOISE + SCHEDULING ===========================
  std::cout << "\n[2/4] Noise generation + scheduling" << std::endl;
  fflush(stdout);

  // Generate noise in CHW then pack to NC (token layout)
  std::vector<float> noise(in_ch * ph * pw);
  {
    std::mt19937 rng(g_seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &x : noise) x = dist(rng);
  }
  auto latents = pack_chw_to_nc(noise, in_ch, ph, pw);
  noise.clear();
  noise.shrink_to_fit();
  std::cout << "  latents: " << num_tokens << " x " << in_ch << std::endl;

  // Build sigma schedule with dynamic time-shift
  auto sigmas = build_sigmas(num_steps, num_tokens);
  std::cout << "  sigmas:";
  for (auto s : sigmas) printf(" %.4f", s);
  std::cout << std::endl;

  auto img_ids = make_img_ids(ph, pw);
  auto txt_ids = make_txt_ids(max_text_len);

  // ======================= 2b. IMG2IMG ENCODE (optional) ===================
  std::vector<float> ref_tokens;
  std::vector<float> ref_ids;
  int num_ref_patches = 0;

  if (is_img2img) {
    std::cout << "  Encoding reference image for img2img..." << std::endl;
    t0 = now_sec();

    std::string raw_rgb = base64_decode(g_input_image_b64);
    int img_h = g_config.height;
    int img_w = g_config.width;
    int expected_size = img_h * img_w * 3;
    if ((int)raw_rgb.size() != expected_size) {
      throw std::runtime_error(
          "Input image size mismatch: got " + std::to_string(raw_rgb.size()) +
          " bytes, expected " + std::to_string(expected_size));
    }

    auto pixel_float =
        rgbToFloat(reinterpret_cast<const uint8_t *>(raw_rgb.data()),
                   img_h, img_w);

    // Load VAE encoder, encode, release
    std::vector<float> ref_latents;
    {
      Module vae_encoder(g_vae_encoder_path);
      auto vae_enc_tp = from_blob(pixel_float.data(),
                                  {1, 3, img_h, img_w}, ScalarType::Float);
      std::vector<EValue> vae_inputs;
      vae_inputs.push_back(*vae_enc_tp);

      auto vae_res = vae_encoder.execute("forward", vae_inputs);
      if (!vae_res.ok()) {
        throw std::runtime_error("VAE encoder forward failed");
      }
      const auto &enc_t = (*vae_res)[0].toTensor();
      ref_latents.assign(enc_t.const_data_ptr<float>(),
                         enc_t.const_data_ptr<float>() + enc_t.numel());
    }
    // VAE encoder released

    // Forward BN normalize
    if (has_bn) {
      bn_normalize(ref_latents.data(), g_config.bn_running_mean.data(),
                   g_config.bn_running_var.data(),
                   latent_ch, latent_h, latent_w, g_config.batch_norm_eps);
    }

    // Patchify (CHW → C*4, H/2, W/2) then pack to NC
    auto ref_patchified = patchify_chw(ref_latents.data(),
                                        latent_ch, latent_h, latent_w);
    ref_tokens = pack_chw_to_nc(ref_patchified, in_ch, ph, pw);
    num_ref_patches = num_tokens;

    ref_ids = make_ref_img_ids(ph, pw, 0);

    std::cout << "  ref encoded (" << (now_sec() - t0) << " s)" << std::endl;
    fflush(stdout);
  }

  // ======================= 3. DENOISING LOOP ===============================
  std::cout << "\n[3/4] Denoising (" << num_steps << " steps)"
            << (is_img2img ? " [img2img]" : " [txt2img]") << std::endl;
  fflush(stdout);
  t0 = now_sec();

  long long first_step_ms = 0;
  {
    // Load the appropriate transformer
    std::string dit_path = is_img2img ? g_dit_img2img_path : g_dit_path;
    Module transformer(dit_path);
    std::cout << "  transformer loaded (" << (now_sec() - t0) << " s)"
              << std::endl;
    fflush(stdout);
    double denoise_t0 = now_sec();

    for (int step = 0; step < num_steps; step++) {
      double step_t0 = now_sec();
      float sigma = sigmas[step];
      float sigma_next = sigmas[step + 1];
      std::vector<float> ts_buf = {sigma};

      std::vector<EValue> tf_inputs;

      if (is_img2img) {
        // Concatenate [noise_tokens, ref_tokens] for hidden_states
        int total_patches = num_tokens + num_ref_patches;
        std::vector<float> combined_hs(total_patches * in_ch);
        std::copy(latents.begin(), latents.end(), combined_hs.begin());
        std::copy(ref_tokens.begin(), ref_tokens.end(),
                  combined_hs.begin() + num_tokens * in_ch);

        std::vector<float> combined_ids(total_patches * 4);
        std::copy(img_ids.begin(), img_ids.end(), combined_ids.begin());
        std::copy(ref_ids.begin(), ref_ids.end(),
                  combined_ids.begin() + num_tokens * 4);

        auto hs_tp = from_blob(combined_hs.data(),
                                {1, total_patches, in_ch}, ScalarType::Float);
        auto ehs_tp = from_blob(prompt_embeds.data(),
                                 {1, max_text_len, joint_dim}, ScalarType::Float);
        auto ts_tp = from_blob(ts_buf.data(), {1}, ScalarType::Float);
        auto iid_tp = from_blob(combined_ids.data(),
                                 {1, total_patches, 4}, ScalarType::Float);
        auto tid_tp = from_blob(txt_ids.data(),
                                 {1, max_text_len, 4}, ScalarType::Float);

        tf_inputs.push_back(*hs_tp);
        tf_inputs.push_back(*ehs_tp);
        tf_inputs.push_back(*ts_tp);
        tf_inputs.push_back(*iid_tp);
        tf_inputs.push_back(*tid_tp);

        auto tf_res = transformer.execute("forward", tf_inputs);
        if (!tf_res.ok()) {
          throw std::runtime_error("Transformer img2img failed at step " +
                                   std::to_string(step));
        }
        const auto &pred_t = (*tf_res)[0].toTensor();
        const float *pred = pred_t.const_data_ptr<float>();

        // Euler step on noise tokens only (first num_tokens * in_ch elements)
        float dt = sigma_next - sigma;
        for (int j = 0; j < (int)latents.size(); j++)
          latents[j] += dt * pred[j];
      } else {
        auto hs_tp = from_blob(latents.data(),
                                {1, num_tokens, in_ch}, ScalarType::Float);
        auto ehs_tp = from_blob(prompt_embeds.data(),
                                 {1, max_text_len, joint_dim}, ScalarType::Float);
        auto ts_tp = from_blob(ts_buf.data(), {1}, ScalarType::Float);
        auto iid_tp = from_blob(img_ids.data(),
                                 {1, num_tokens, 4}, ScalarType::Float);
        auto tid_tp = from_blob(txt_ids.data(),
                                 {1, max_text_len, 4}, ScalarType::Float);

        tf_inputs.push_back(*hs_tp);
        tf_inputs.push_back(*ehs_tp);
        tf_inputs.push_back(*ts_tp);
        tf_inputs.push_back(*iid_tp);
        tf_inputs.push_back(*tid_tp);

        auto tf_res = transformer.execute("forward", tf_inputs);
        if (!tf_res.ok()) {
          throw std::runtime_error("Transformer failed at step " +
                                   std::to_string(step));
        }
        const auto &pred_t = (*tf_res)[0].toTensor();
        const float *pred = pred_t.const_data_ptr<float>();

        float dt = sigma_next - sigma;
        for (size_t j = 0; j < latents.size(); j++)
          latents[j] += dt * pred[j];
      }

      double step_sec = now_sec() - step_t0;
      auto step_ms = static_cast<long long>(step_sec * 1000.0);
      if (step == 0) first_step_ms = step_ms;

      printf("  step %d/%d  sigma %.4f -> %.4f  (%.2f s)\n",
             step + 1, num_steps, sigma, sigma_next, step_sec);
      fflush(stdout);

      progress_cb(step + 1, num_steps, "");
    }
    std::cout << "  total denoise: " << (now_sec() - denoise_t0) << " s"
              << std::endl;
    fflush(stdout);
  }
  // transformer is destroyed here — memory freed
  std::cout << "  transformer released" << std::endl;
  fflush(stdout);

  // Free intermediate data no longer needed
  prompt_embeds.clear();
  prompt_embeds.shrink_to_fit();
  img_ids.clear();
  img_ids.shrink_to_fit();
  txt_ids.clear();
  txt_ids.shrink_to_fit();
  ref_tokens.clear();
  ref_tokens.shrink_to_fit();
  ref_ids.clear();
  ref_ids.shrink_to_fit();

  // ======================= 4. POST-PROCESSING + VAE DECODE =================
  std::cout << "\n[4/4] Post-processing + VAE decode" << std::endl;
  fflush(stdout);

  // Unpack from NC token layout back to CHW spatial
  auto spatial_ids = make_img_ids(ph, pw);
  auto spatial = unpack_nc_to_chw(
      latents.data(), spatial_ids.data(), num_tokens, in_ch, ph, pw);
  latents.clear();
  latents.shrink_to_fit();
  std::cout << "  unpacked: " << in_ch << " x " << ph << " x " << pw
            << std::endl;

  // Inverse batch-norm (in patchified space, in_channels=128)
  // Note: BN stats must match in_channels if applied here.
  // If BN stats are for latent_channels (32), apply after unpatchify instead.
  bool bn_matches_in_ch = (int)g_config.bn_running_mean.size() == in_ch;
  if (has_bn && bn_matches_in_ch) {
    bn_unnormalize(spatial.data(), g_config.bn_running_mean.data(),
                   g_config.bn_running_var.data(),
                   in_ch, ph, pw, g_config.batch_norm_eps);
    std::cout << "  BN un-normalised (patchified space, " << in_ch << " ch)"
              << std::endl;
  }

  // Unpatchify: (in_ch, ph, pw) → (latent_ch, latent_h, latent_w)
  auto lat_full = unpatchify(spatial.data(), in_ch, ph, pw);
  spatial.clear();
  spatial.shrink_to_fit();
  std::cout << "  unpatchified: " << latent_ch << " x " << latent_h << " x "
            << latent_w << std::endl;

  // If BN stats are for latent_channels (32), apply after unpatchify
  if (has_bn && !bn_matches_in_ch) {
    bn_unnormalize(lat_full.data(), g_config.bn_running_mean.data(),
                   g_config.bn_running_var.data(),
                   latent_ch, latent_h, latent_w, g_config.batch_norm_eps);
    std::cout << "  BN un-normalised (latent space, " << latent_ch << " ch)"
              << std::endl;
  }

  // VAE decode (load → run → release)
  std::cout << "  loading vae_decoder..." << std::endl;
  fflush(stdout);
  t0 = now_sec();
  std::vector<uint8_t> rgb;
  {
    Module vae_decoder(g_vae_decoder_path);
    std::cout << "  vae_decoder loaded (" << (now_sec() - t0) << " s)"
              << std::endl;
    fflush(stdout);

    auto vae_tp = from_blob(lat_full.data(),
                             {1, latent_ch, latent_h, latent_w},
                             ScalarType::Float);
    std::vector<EValue> vae_inputs;
    vae_inputs.push_back(*vae_tp);

    t0 = now_sec();
    auto vae_res = vae_decoder.execute("forward", vae_inputs);
    if (!vae_res.ok()) {
      throw std::runtime_error("VAE decoder forward failed");
    }
    const auto &pix_t = (*vae_res)[0].toTensor();
    std::cout << "  decoded: numel=" << pix_t.numel()
              << " (" << (now_sec() - t0) << " s)" << std::endl;
    fflush(stdout);

    rgb = pixelsToRGB(pix_t.const_data_ptr<float>(),
                      g_config.height, g_config.width);
  }
  // vae_decoder released
  std::cout << "  vae decoder released" << std::endl;

  auto pipeline_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      pipeline_end - pipeline_start)
                      .count();
  std::cout << "Total generation time: " << total_ms << "ms" << std::endl;

  GenerationResult result;
  result.image_data = std::move(rgb);
  result.width = g_config.width;
  result.height = g_config.height;
  result.channels = 3;
  result.generation_time_ms = total_ms;
  result.first_step_time_ms = first_step_ms;
  return result;
}

// ============================================================================
// Command-line parsing
// ============================================================================

static void showHelp(const char *prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --encoder <path>       Path to text_encoder.pte\n"
            << "  --dit <path>           Path to transformer.pte\n"
            << "  --vae_decoder <path>   Path to vae_decoder.pte\n"
            << "  --tokenizer <path>     Path to tokenizer.json\n"
            << "  --config <path>        Path to export_config.json\n"
            << "  --bn_stats <path>      Path to BN stats (JSON or binary)\n"
            << "  --vae_encoder <path>   Path to vae_encoder.pte (optional, for img2img)\n"
            << "  --dit_img2img <path>   Path to transformer_img2img.pte (optional)\n"
            << "  --port <num>           Server port (default: 8081)\n";
}

static void parseArgs(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        exit(1);
      }
      return argv[++i];
    };
    if (arg == "--encoder")
      g_encoder_path = next();
    else if (arg == "--dit")
      g_dit_path = next();
    else if (arg == "--dit_img2img")
      g_dit_img2img_path = next();
    else if (arg == "--vae_decoder")
      g_vae_decoder_path = next();
    else if (arg == "--vae_encoder")
      g_vae_encoder_path = next();
    else if (arg == "--tokenizer")
      g_tokenizer_path = next();
    else if (arg == "--config")
      g_config_path = next();
    else if (arg == "--bn_stats")
      g_bn_stats_path = next();
    else if (arg == "--port")
      g_port = std::stoi(next());
    else if (arg == "--help" || arg == "-h") {
      showHelp(argv[0]);
      exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      showHelp(argv[0]);
      exit(1);
    }
  }

  if (g_encoder_path.empty() || g_dit_path.empty() ||
      g_vae_decoder_path.empty()) {
    std::cerr << "ERROR: --encoder, --dit, and --vae_decoder are required\n";
    showHelp(argv[0]);
    exit(1);
  }
  if (g_tokenizer_path.empty()) {
    std::cerr << "ERROR: --tokenizer is required\n";
    exit(1);
  }
  if (g_config_path.empty()) {
    std::cerr << "ERROR: --config is required\n";
    exit(1);
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  parseArgs(argc, argv);

  // Configure thread pool to use only performance cores.
  // Without this, pthreadpool defaults to ALL cores, including slow efficiency
  // cores, which can cause 10-20x slowdowns on Android.
  {
    uint32_t num_threads =
        ::executorch::extension::cpuinfo::get_num_performant_cores();
    if (num_threads == 0) {
      num_threads = 4;
    }
    std::cout << "Setting threadpool to " << num_threads << " threads"
              << std::endl;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_threads);
#pragma GCC diagnostic pop
  }

  // --- Load config ---
  std::cout << "Loading export config: " << g_config_path << std::endl;
  loadExportConfig(g_config_path);

  // --- Load BN stats ---
  loadBNStats(g_bn_stats_path);

  // --- Load tokenizer (stays in memory — tiny footprint) ---
  std::cout << "Loading tokenizer: " << g_tokenizer_path << std::endl;
  try {
    auto blob = loadFileToString(g_tokenizer_path);
    g_tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
    if (!g_tokenizer) throw std::runtime_error("Tokenizer creation failed.");
  } catch (const std::exception &e) {
    std::cerr << "ERROR loading tokenizer: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // --- Verify model files exist ---
  for (const auto &[name, path] :
       std::vector<std::pair<std::string, std::string>>{
           {"text_encoder", g_encoder_path},
           {"transformer", g_dit_path},
           {"vae_decoder", g_vae_decoder_path}}) {
    if (!std::filesystem::exists(path)) {
      std::cerr << "ERROR: " << name << " not found: " << path << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "  " << name << ": " << path << std::endl;
  }

  bool img2img_available = !g_vae_encoder_path.empty() &&
                           !g_dit_img2img_path.empty() &&
                           std::filesystem::exists(g_vae_encoder_path) &&
                           std::filesystem::exists(g_dit_img2img_path) &&
                           g_config.num_img2img_images > 0;
  if (img2img_available) {
    std::cout << "  vae_encoder: " << g_vae_encoder_path << std::endl;
    std::cout << "  dit_img2img: " << g_dit_img2img_path << std::endl;
  }

  std::cout << "Ready. Models will be loaded per-stage to minimize memory."
            << (img2img_available ? " (img2img available)" : " (txt2img only)")
            << std::endl;

  // --- HTTP Server ---
  httplib::Server svr;

  svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
    res.set_header("Access-Control-Allow-Origin", "*");
    res.status = 200;
  });

  svr.Post("/generate", [&](const httplib::Request &req,
                             httplib::Response &res) {
    try {
      auto json = nlohmann::json::parse(req.body);
      if (!json.contains("prompt"))
        throw std::invalid_argument("Missing 'prompt'");

      g_prompt = json["prompt"].get<std::string>();
      g_seed = json.value(
          "seed",
          hashSeed(
              std::chrono::system_clock::now().time_since_epoch().count()));

      g_input_image_b64 = json.value("image", std::string());
      g_denoise_strength = json.value("denoise_strength", 1.0f);

      g_output_width = json.value("width", g_config.width);
      g_output_height = json.value("height", g_config.height);

      if (g_output_width != g_config.width ||
          g_output_height != g_config.height) {
        std::cerr << "WARN: Requested " << g_output_width << "x"
                  << g_output_height << " but model exported for "
                  << g_config.width << "x" << g_config.height
                  << ". Using exported dimensions." << std::endl;
        g_output_width = g_config.width;
        g_output_height = g_config.height;
      }

      std::cout << "\nGenerate: prompt=\"" << g_prompt << "\" seed=" << g_seed
                << " size=" << g_output_width << "x" << g_output_height
                << std::endl;

      // SSE streaming response
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      res.set_header("Access-Control-Allow-Origin", "*");

      res.set_chunked_content_provider(
          "text/event-stream",
          [&](intptr_t, httplib::DataSink &sink) -> bool {
            try {
              auto result = generateImage(
                  [&sink](int s, int t, const std::string &img) {
                    nlohmann::json p = {
                        {"type", "progress"}, {"step", s}, {"total_steps", t}};
                    if (!img.empty()) {
                      p["image"] = img;
                    }
                    std::string ev = "data: " + p.dump() + "\n\n";
                    sink.write(ev.c_str(), ev.size());
                  });

              // Base64 encode the result image
              std::string image_str(result.image_data.begin(),
                                    result.image_data.end());
              std::string enc_img = base64_encode(image_str);

              nlohmann::json c = {
                  {"type", "complete"},
                  {"image", enc_img},
                  {"seed", g_seed},
                  {"width", result.width},
                  {"height", result.height},
                  {"channels", result.channels},
                  {"generation_time_ms", result.generation_time_ms},
                  {"first_step_time_ms", result.first_step_time_ms}};
              std::string ev = "data: " + c.dump() + "\n\n";
              sink.write(ev.c_str(), ev.size());
              std::string done = "data: [DONE]\n\n";
              sink.write(done.c_str(), done.size());
              sink.done();
              return true;
            } catch (const std::exception &e) {
              nlohmann::json err = {{"type", "error"}, {"message", e.what()}};
              std::string ev = "data: " + err.dump() + "\n\n";
              sink.write(ev.c_str(), ev.size());
              sink.done();
              return false;
            }
          });
    } catch (const nlohmann::json::parse_error &e) {
      nlohmann::json err = {
          {"error",
           {{"message", "Invalid JSON: " + std::string(e.what())},
            {"type", "request_error"}}}};
      res.status = 400;
      res.set_content(err.dump(), "application/json");
      res.set_header("Access-Control-Allow-Origin", "*");
    } catch (const std::invalid_argument &e) {
      nlohmann::json err = {
          {"error",
           {{"message", "Invalid Arg: " + std::string(e.what())},
            {"type", "request_error"}}}};
      res.status = 400;
      res.set_content(err.dump(), "application/json");
      res.set_header("Access-Control-Allow-Origin", "*");
    } catch (const std::exception &e) {
      nlohmann::json err = {
          {"error",
           {{"message", "Server Err: " + std::string(e.what())},
            {"type", "server_error"}}}};
      res.status = 500;
      res.set_content(err.dump(), "application/json");
      res.set_header("Access-Control-Allow-Origin", "*");
    }
  });

  std::cout << "\nServer listening on " << g_listen_address << ":" << g_port
            << std::endl;
  svr.listen(g_listen_address.c_str(), g_port);

  return EXIT_SUCCESS;
}
