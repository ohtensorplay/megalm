// SPDX-License-Identifier: Apache-2.0
// TensorPlay C ABI for the vLLM grouped Marlin W4A16 device kernel.
#include "marlin/moe_template.h"

#include <cub/block/block_scan.cuh>

namespace {
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kExperts = 256;
constexpr int kGroup = 32;
constexpr int kTileN = 128;
constexpr auto kBF16 = vllm::kBFloat16.id();
constexpr auto kU4B8 = vllm::kU4B8.id();

// A warp owns a contiguous span of route IDs. Prefix sums first reserve each
// warp's portion of every expert, then match_any ranks routes within a warp.
// Padding comes after valid routes, as required by the Marlin row loader.
__global__ void align_routes(const int64_t* ids, int routes, int experts,
                             int block_rows, int* sorted, int* expert_ids,
                             int* padded) {
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ Scan::TempStorage scan;
  __shared__ int counts[kWarps][kExperts];
  const int t = threadIdx.x;
  const int warp = t / kWarpSize;
  const int lane = t % kWarpSize;
  const int rounds = (routes + kThreads - 1) / kThreads;
  const int begin = warp * rounds * kWarpSize + lane;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) counts[w][t] = 0;
  __syncthreads();

  for (int i = 0; i < rounds; ++i) {
    const int route = begin + i * kWarpSize;
    const unsigned active = __ballot_sync(0xffffffff, route < routes);
    if (route < routes) {
      const int expert = ids[route];
      const unsigned peers = __match_any_sync(active, expert);
      if (lane == __ffs(peers) - 1) counts[warp][expert] += __popc(peers);
    }
    __syncwarp();
  }
  __syncthreads();

  int count = 0;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) count += counts[w][t];
  const int blocks = (count + block_rows - 1) / block_rows;
  int first, total;
  Scan(scan).ExclusiveSum(blocks, first, total);
  if (t == 0) *padded = total * block_rows;
  if (t < experts) {
    for (int i = 0; i < blocks; ++i) expert_ids[first + i] = t;
    int cursor = first * block_rows;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const int size = counts[w][t];
      counts[w][t] = cursor;
      cursor += size;
    }
    for (int i = cursor; i < (first + blocks) * block_rows; ++i) {
      sorted[i] = routes;
    }
  }
  __syncthreads();

  for (int i = 0; i < rounds; ++i) {
    const int route = begin + i * kWarpSize;
    const unsigned active = __ballot_sync(0xffffffff, route < routes);
    if (route < routes) {
      const int expert = ids[route];
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      int offset = lane == leader ? counts[warp][expert] : 0;
      offset = __shfl_sync(peers, offset, leader);
      sorted[offset + __popc(peers & ((1u << lane) - 1))] = route;
      if (lane == leader) counts[warp][expert] = offset + __popc(peers);
    }
    __syncwarp();
  }
}

__global__ void align_dense(int rows, int block_rows, int* sorted,
                            int* expert_ids, int* padded) {
  const int blocks = (rows + block_rows - 1) / block_rows;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) *padded = blocks * block_rows;
  if (index < blocks) expert_ids[index] = 0;
  if (index < blocks * block_rows) sorted[index] = min(index, rows);
}

__global__ void gelu_mul(const nv_bfloat16* gate_up, nv_bfloat16* activation,
                         int elements, int width) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    const int gate = (i / width) * width * 2 + i % width;
    const float x = __bfloat162float(gate_up[gate]);
    const float gelu =
        0.5f * x * (1.0f + tanhf(0.7978845608028654f *
                                 (x + 0.044715f * x * x * x)));
    activation[i] = __float2bfloat16(
        __bfloat162float(__float2bfloat16(gelu)) *
        __bfloat162float(gate_up[gate + width]));
  }
}

__global__ void reduce_routes(const nv_bfloat16* down, const float* weights,
                              nv_bfloat16* output, int tokens, int top_k,
                              int hidden) {
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= tokens * hidden) return;
  const int token = element / hidden;
  const int column = element % hidden;
  const int route = token * top_k;
  float sum = 0.0f;
  for (int slot = 0; slot < top_k; ++slot) {
    sum += __bfloat162float(down[(route + slot) * hidden + column]) *
           weights[route + slot];
  }
  output[element] = __float2bfloat16(sum);
}

// Marlin permutes 16x16 subtiles in groups of four output tiles. Each thread
// assembles one word of eight nibbles directly from the checkpoint slab.
__global__ void pack_codes(const uint8_t* codes, uint32_t* packed,
                           int elements, int n, int k) {
  constexpr int interleave[] = {0, 2, 4, 6, 1, 3, 5, 7};
  const int row_words = n * 2;
  const int expert_words = n * (k / 8);
  for (int out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += blockDim.x * gridDim.x) {
    const int expert = out / expert_words;
    const int word = out % row_words;
    const int k_tile = (out % expert_words) / row_words;
    const int stripe = word / 128;
    const int fragment = (word % 128) / 4;
    const int n_tile = (word % 4) + stripe * 4;
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int lane = interleave[i];
      const int row = (fragment % 4) * 2 + lane % 2 + ((lane % 4) / 2) * 8;
      const int col = (fragment / 4) + (lane / 4) * 8;
      const int src_n = n_tile * 16 + col;
      const int src_k = k_tile * 16 + row;
      const uint32_t byte = codes[(expert * n + src_n) * (k / 2) + src_k / 2];
      result |= ((byte >> (4 * (src_k % 2))) & 0xf) << (4 * i);
    }
    packed[out] = result;
  }
}

__global__ void pack_scales(const uint16_t* scale_zero, uint16_t* scales,
                            int elements) {
  for (int out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += blockDim.x * gridDim.x) {
    // Transpose each 8x8 block of the contiguous output-channel dimension.
    const int src = (out / 64) * 64 + (out % 8) * 8 + (out % 64) / 8;
    scales[out] = scale_zero[2 * src];
  }
}

using Kernel = decltype(&marlin_moe_wna16::Marlin<
    kBF16, kU4B8, kBF16, kBF16, 128, 1, 8, 4, true, 4, 2, false>);
struct Config {
  Kernel kernel;
  int threads;
  int shared;
};

template <int Rows, int TileK>
constexpr Config config() {
  constexpr int m = Rows <= 16 ? 1 : Rows / 16;
  constexpr int threads = TileK == 128 ? 256 : 128;
  constexpr int stages = 4;
  constexpr int kQuantizedTile = stages * TileK * kTileN / 2;
  constexpr int kReduceTile = m * 16 * (kTileN + 8) * 2;
  constexpr int kBias = kTileN * 2;
  constexpr int kSharedMain = std::max(
      std::max(kQuantizedTile, kReduceTile),
      std::min(kQuantizedTile, kReduceTile) + kBias);
  constexpr int shared = 16 * Rows + kSharedMain +
                         stages * m * 16 * TileK * 2 +
                         stages * (TileK / kGroup) * kTileN * 2;
  return {&marlin_moe_wna16::Marlin<kBF16, kU4B8, kBF16, kBF16, threads,
                                   m, 8, TileK / 16, Rows == 8, stages,
                                   kGroup / 16, false>,
          threads, shared};
}

constexpr Config kConfigs[] = {
    config<8, 64>(), config<8, 128>(), config<16, 64>(), config<16, 128>(),
    config<32, 64>(), config<48, 64>(), config<64, 64>(),
};

void project(const Config& cfg, const void* input, const void* packed,
             const void* scales, const int* sorted, const int* expert_ids,
             const int* padded, void* output, void* temporary, int* locks,
             int m, int n, int k, int top_k, int grid, cudaStream_t stream) {
  cfg.kernel<<<grid, cfg.threads, cfg.shared, stream>>>(
      static_cast<const int4*>(input), static_cast<const int4*>(packed),
      static_cast<int4*>(output), static_cast<int4*>(temporary), nullptr,
      nullptr, static_cast<const int4*>(scales), nullptr, nullptr, nullptr,
      sorted, expert_ids, padded, nullptr, top_k, false, k / kGroup, m, n, k,
      locks, false, false, true);
}
}  // namespace

extern "C" int megalm_marlin_moe_pack(
    const uint8_t* codes, const uint16_t* scale_zero, uint32_t* packed,
    uint16_t* scales, int experts, int n, int k, void* raw_stream) {
  const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const int words = experts * n * (k / 8);
  const int scale_count = experts * n * (k / kGroup);
  pack_codes<<<std::min((words + kThreads - 1) / kThreads, 4096), kThreads,
               0, stream>>>(codes, packed, words, n, k);
  pack_scales<<<std::min((scale_count + kThreads - 1) / kThreads, 4096), kThreads,
                0, stream>>>(scale_zero, scales, scale_count);
  return cudaGetLastError();
}

extern "C" int megalm_marlin_moe_prepare(int device, int* grids) {
  int sms, shared;
  auto error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess) return error;
  error = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                 device);
  if (error != cudaSuccess) return error;
  for (int i = 0; i < sizeof(kConfigs) / sizeof(kConfigs[0]); ++i) {
    const auto& cfg = kConfigs[i];
    if (cfg.shared > shared) {
      grids[i] = 0;
      continue;
    }
    error = cudaFuncSetAttribute(cfg.kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 cfg.shared);
    if (error != cudaSuccess) return error;
    int resident;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, cfg.kernel, cfg.threads, cfg.shared);
    if (error != cudaSuccess) return error;
    grids[i] = sms * std::min(resident, i < 4 ? 4 : 2);
  }
  return cudaSuccess;
}

extern "C" int megalm_marlin_project(
    const void* input, const void* packed, const void* scales,
    int* sorted, int* expert_ids, int* padded, int* locks, void* temporary,
    void* output, int rows, int n, int k, int block_rows, int configuration,
    int grid, void* raw_stream) {
  const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  auto error = cudaMemsetAsync(locks, 0, grid * sizeof(int), stream);
  if (error != cudaSuccess) return error;
  align_dense<<<(rows + kThreads - 1) / kThreads,
                kThreads, 0, stream>>>(rows, block_rows, sorted, expert_ids, padded);
  project(kConfigs[configuration], input, packed, scales, sorted, expert_ids,
          padded, output, temporary, locks, rows, n, k, 1, grid, stream);
  return cudaGetLastError();
}

extern "C" int megalm_marlin_moe(
    const void* x, const int64_t* ids, const float* weights,
    const void* q13, const void* s13, const void* q2, const void* s2,
    int* sorted, int* expert_ids, int* padded, int* locks, void* temporary,
    void* stage, void* activation, void* output, int tokens, int top_k,
    int experts, int hidden, int intermediate, int block_rows, int up_config,
    int down_config, int up_grid, int down_grid, void* raw_stream) {
  const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const int routes = tokens * top_k;
  auto error = cudaMemsetAsync(locks, 0, std::max(up_grid, down_grid) * sizeof(int),
                               stream);
  if (error != cudaSuccess) return error;
  align_routes<<<1, kThreads, 0, stream>>>(
      ids, routes, experts, block_rows, sorted, expert_ids, padded);
  project(kConfigs[up_config], x, q13, s13, sorted, expert_ids, padded, stage,
          temporary, locks, tokens, 2 * intermediate, hidden, top_k, up_grid,
          stream);
  const int elements = routes * intermediate;
  gelu_mul<<<std::min((elements + kThreads - 1) / kThreads, 4096),
             kThreads, 0, stream>>>(static_cast<const nv_bfloat16*>(stage),
                                  static_cast<nv_bfloat16*>(activation),
                                  elements, intermediate);
  project(kConfigs[down_config], activation, q2, s2, sorted, expert_ids, padded,
          stage, temporary, locks, routes, hidden, intermediate, 1, down_grid,
          stream);
  reduce_routes<<<(tokens * hidden + kThreads - 1) / kThreads,
                  kThreads, 0, stream>>>(
      static_cast<const nv_bfloat16*>(stage), weights,
      static_cast<nv_bfloat16*>(output), tokens, top_k, hidden);
  return cudaGetLastError();
}
