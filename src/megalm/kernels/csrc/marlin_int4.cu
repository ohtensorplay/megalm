// Marlin-class W4A16 int4 GEMM for MegaLM, lifted from vLLM's Apache-2.0
// Marlin kernel (the vendored headers in ./marlin/). Two C entry points:
//   marlin_decode  : small-M decode config (m_block_size_8, 128x128 tile)
//   marlin_prefill : large-M prefill config (16*TM-row m-tile, 256x64 tile),
//                    replicating vLLM's prob_m_split chunking.
// Safe, bit-equivalent consumers of the AWQ checkpoint weights (zero = 8 baked
// into the uint4 code, dequant = (code-8)*scale folded into an hfma2 chain).
#include "marlin/kernel.h"
#include "marlin/marlin_template.h"

#include <cstdio>
#include <cuda_runtime.h>

namespace {
using namespace marlin;
constexpr auto kBf16 = vllm::kBFloat16.id();
constexpr auto kU4 = vllm::kU4B8.id();

// Device attribute query and per-kernel shared-memory opt-in are illegal inside
// a CUDA graph capture region, so they run exactly once, lazily, before the
// first launch (which always happens during capture-free warm-up in practice).
int cached_sms() {
  static int sms = 0;
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  return sms;
}
}  // namespace

// Fast path: decode / small-M (m <= 8 fits the m_block_size_8 tile).
extern "C" int marlin_decode(const void* A, const void* B, void* C,
                             void* C_tmp, int* locks,
                             const void* scales, int num_groups,
                             int M, int N, int K, int lda,
                             int max_shared_mem, int blocks, void* stream) {
  constexpr int threads = 256;
  // 704 (the 26B MoE down K) is divisible by 64 but not 128, so the
  // vLLM-validated config here is {64,128,256}, not {128,128,256}.
  constexpr int tm = 1, tn = 8, tk = 4, stages = 4, gb = 2;
  using KFn = void (*)(MARLIN_KERNEL_PARAMS);
  KFn kernel = &Marlin<kBf16, kU4, kBf16, kBf16, threads, tm, tn, tk, true,
                        stages, gb, false>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem);
    configured = true;
  }
  const int grid = blocks > 0 ? blocks : cached_sms();
  kernel<<<grid, threads, max_shared_mem, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const int4*)A, (const int4*)B, (int4*)C, (int4*)C_tmp, nullptr,
      nullptr, (const int4*)scales, nullptr, nullptr, nullptr, num_groups, M,
      N, K, lda, locks, false, false, true, max_shared_mem);
  return 0;
}

// Prefill / large-M. Mirrors vLLM marlin_mm: prob_m_split chunks of
// par_count*64 rows (capped by max_par), one launch each.  The row tile follows
// vLLM's `thread_m_blocks = min(ceil(split / 16), 4)`, so M = 9..63 does not pay
// for a 64-row tile it cannot fill.
namespace {
template <int TM>
void launch_prefill_chunk(const void* A, const void* B, void* C,
                          void* C_tmp, int* locks,
                          const void* scales, int num_groups, int M, int N,
                          int K, int lda, int max_shared_mem, int grid,
                          void* stream) {
  constexpr int threads = 256;
  // vLLM rejects configs where prob_n % thread_n != 0.  The 26B expert
  // gate/up width is 1408 = 11*128 but not a multiple of 256, so a 256-wide
  // thread tile silently drops the last 128 output columns.
  constexpr int tn = 8, tk = 4, stages = 4, gb = 2;
  using KFn = void (*)(MARLIN_KERNEL_PARAMS);
  KFn kernel = &Marlin<kBf16, kU4, kBf16, kBf16, threads, TM, tn, tk, false,
                       stages, gb, false>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem);
    configured = true;
  }
  kernel<<<grid, threads, max_shared_mem, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const int4*)A, (const int4*)B, (int4*)C, (int4*)C_tmp, nullptr,
      nullptr, (const int4*)scales, nullptr, nullptr, nullptr, num_groups, M,
      N, K, lda, locks, false, false, true, max_shared_mem);
}
}  // namespace

extern "C" int marlin_prefill(const void* A, const void* B, void* C,
                              void* C_tmp, int* locks,
                              const void* scales, int num_groups,
                              int M, int N, int K, int lda,
                              int max_shared_mem, int blocks, void* stream) {
  const int grid = blocks > 0 ? blocks : cached_sms();
  const int max_par = (N <= 4096) ? 16 * 8 : 16;
  int rest = M;
  const char* arow = (const char*)A;
  char* crow = (char*)C;
  while (rest > 0) {
    int par_count = rest / (4 * 16);
    if (par_count > max_par) par_count = max_par;
    int split = par_count > 0 ? par_count * (4 * 16) : rest;
    int tm = (split + 15) / 16;
    if (tm > 4) tm = 4;
    switch (tm) {
      case 1:
        launch_prefill_chunk<1>(arow, B, crow, C_tmp, locks, scales, num_groups, split,
                                N, K, lda, max_shared_mem, grid, stream);
        break;
      case 2:
        launch_prefill_chunk<2>(arow, B, crow, C_tmp, locks, scales, num_groups, split,
                                N, K, lda, max_shared_mem, grid, stream);
        break;
      case 3:
        launch_prefill_chunk<3>(arow, B, crow, C_tmp, locks, scales, num_groups, split,
                                N, K, lda, max_shared_mem, grid, stream);
        break;
      default:
        launch_prefill_chunk<4>(arow, B, crow, C_tmp, locks, scales, num_groups, split,
                                N, K, lda, max_shared_mem, grid, stream);
        break;
    }
    arow += (long)split * lda * 2;
    crow += (long)split * N * 2;
    rest -= split;
  }
  return 0;
}
