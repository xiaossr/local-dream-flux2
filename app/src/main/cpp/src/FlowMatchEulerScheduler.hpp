#pragma once

#include <cmath>
#include <vector>

// FlowMatchEulerDiscreteScheduler for FLUX.2-klein (distilled, 4 steps).
//
// Flow-matching formulation: x_t = (1 - t) * x_0 + t * noise
// The model predicts velocity v, and we integrate with Euler steps.
//
// Timesteps go from ~1.0 (pure noise) toward 0.0 (clean image).

class FlowMatchEulerScheduler {
 public:
  FlowMatchEulerScheduler(int num_inference_steps = 4, float shift = 1.0f)
      : num_steps_(num_inference_steps), shift_(shift) {
    // Compute sigma schedule: linearly spaced from 1 to 0
    sigmas_.resize(num_steps_ + 1);
    for (int i = 0; i <= num_steps_; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(num_steps_);
      sigmas_[i] = 1.0f - t;
    }

    // Apply time shift if needed (mu-shift for resolution-dependent scheduling)
    if (shift_ != 1.0f) {
      for (auto &s : sigmas_) {
        s = shift_ * s / (1.0f + (shift_ - 1.0f) * s);
      }
    }

    // Timesteps are the sigma values (excluding the last 0)
    timesteps_.resize(num_steps_);
    for (int i = 0; i < num_steps_; ++i) {
      timesteps_[i] = sigmas_[i];
    }
  }

  int num_steps() const { return num_steps_; }

  const std::vector<float> &timesteps() const { return timesteps_; }

  // Scale model input (identity for flow matching)
  // latents are returned unchanged; sigma is informational
  void scale_model_input(std::vector<float> & /*latents*/,
                         float /*timestep*/) const {
    // No-op for flow matching Euler
  }

  // Euler step: latents += dt * model_output
  // where dt = sigma_{i+1} - sigma_i (negative, moving toward 0)
  void step(std::vector<float> &latents, const float *model_output,
            int step_index) const {
    float dt = sigmas_[step_index + 1] - sigmas_[step_index];
    size_t n = latents.size();
    for (size_t j = 0; j < n; ++j) {
      latents[j] += dt * model_output[j];
    }
  }

 private:
  int num_steps_;
  float shift_;
  std::vector<float> sigmas_;
  std::vector<float> timesteps_;
};
