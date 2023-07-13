#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>
#include <random>
#include <omp.h>
#include <stdlib.h> 
#include <ATen/NativeFunctions.h>
#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/core/Generator.h>
#include <ATen/CPUGeneratorImpl.h>

// torch::Tensor edge_sample(torch::Tensor p_cumsum, int64_t batch_size, c10::optional<at::Generator> generator){
//     auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
//     const double* p_cumsum_ptr = p_cumsum.data_ptr<double>();
//     int64_t size = p_cumsum.numel();
//     int64_t sampled_indices[size] = {0};
//     auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(generator, at::detail::getDefaultCPUGenerator());
//     std::lock_guard<std::mutex> lock(gen->mutex_);
//     #pragma omp parallel for
//     for (int64_t i = 0; i < batch_size; i++) {
//     // for (const auto i : c10::irange(batch_size)){
//       at::uniform_real_distribution<double> standard_uniform(0, 1);
//       double uniform_sample = standard_uniform(gen);
//       int64_t left_pointer = 0;
//       int64_t right_pointer = p_cumsum.numel();
//       int64_t mid_pointer;
//       int64_t sample_idx;
//       double cum_prob;
//       while(right_pointer - left_pointer > 0) {
//         mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
//         cum_prob = p_cumsum_ptr[mid_pointer];
//         if (cum_prob < uniform_sample) {
//           left_pointer = mid_pointer + 1;}
//         else {right_pointer = mid_pointer;}
//       }
//       sample_idx = left_pointer;
//       sampled_indices[sample_idx] += 1;
//       #pragma omp critical
//       {
//         sampled_indices[sample_idx] += 1;
//       }
//     }
//     return torch::from_blob(sampled_indices, {size,}, options=options).clone();
// }

torch::Tensor edge_sample(torch::Tensor p_cumsum, int64_t batch_size, uint64_t seed_val){
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const double* p_cumsum_ptr = p_cumsum.data_ptr<double>();
    torch::Tensor sampled_indices = torch::zeros(batch_size, options);
    std::vector<std::default_random_engine> generators;
    for (int i = 0, N = omp_get_max_threads(); i < N; ++i) {
        generators.emplace_back(std::default_random_engine(seed_val + i));
    }
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
      std::uniform_real_distribution<double> standard_uniform(0, 1);
      std::default_random_engine& engine = generators[omp_get_thread_num()];
      double uniform_sample = standard_uniform(engine);
      int64_t left_pointer = 0;
      int64_t right_pointer = p_cumsum.numel();
      int64_t mid_pointer;
      int64_t sample_idx;
      double cum_prob;
      while(right_pointer - left_pointer > 0) {
        mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        cum_prob = p_cumsum_ptr[mid_pointer];
        if (cum_prob < uniform_sample) {
          left_pointer = mid_pointer + 1;}
        else {right_pointer = mid_pointer;}
      }
      sample_idx = left_pointer;
      sampled_indices[i] = sample_idx;
    }
    return sampled_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_sample", &edge_sample, "fast edge sampler");
}