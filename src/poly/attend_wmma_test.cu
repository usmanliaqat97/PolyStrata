// src/poly/attend_wmma_test.cu - attend_batch() for Qwen3.6's heads (256 values, 16 heads in 2 groups of 8) against
// a CPU reference in double: whichever path this build and card take (RDNA3: attn_wmma_kernel; STRATA_POLY_WMMA_ATTN=0
// or another card: the cuBLAS products), so running it both ways compares the two.  The CPU takes the queries
// rounded to halves, as both GPU paths do, and every 7th token of each case.  Each case runs on an fp16 cache, on a
// q8_0 cache whose workspaces hold the whole context (RDNA3: the rows dequantized at once, then attn_wmma_kernel),
// and on a q8_0 cache with 2048-row workspaces (the products a block where the case attends more rows); the q8 cases'
// reference reads the dequantized rows.
//
// usage: poly_attend_wmma_test   (exit 0: every case within 5e-3 of the reference)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 256, kHeads = 16, kKvHeads = 2, kCtx = 8192;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

float h2f(uint16_t h) { return __half2float(*reinterpret_cast<const __half*>(&h)); }
uint16_t f2h(float f) {
    const __half h = __float2half(f);
    return *reinterpret_cast<const uint16_t*>(&h);
}

}  // namespace

int main() {
    const char* e = std::getenv("STRATA_POLY_WMMA_ATTN");
    std::printf("path: %s\n", e && e[0] == '0' ? "cuBLAS products (STRATA_POLY_WMMA_ATTN=0)" : "default for this card");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int max_rows = 1024, per = kHeads / kKvHeads, width = kKvHeads * kDim;
    std::vector<uint16_t> k(size_t(kCtx) * width), v(k.size());
    for (size_t i = 0; i < k.size(); ++i) k[i] = f2h(nd(rng)), v[i] = f2h(nd(rng));
    std::vector<float> q(size_t(max_rows) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);   // scores of a few units, so the softmax is neither flat nor one-hot
    // the same rows as Q8_0 blocks of 32 (llama.cpp's quantize_row_q8_0: d = max |x| / 127), and their dequantized
    // halves for the reference
    const int blocks = width / 32;
    std::vector<int8_t> kc(k.size()), vc(v.size());
    std::vector<uint16_t> ks(size_t(kCtx) * blocks), vs(ks.size()), kd(k.size()), vd(v.size());
    auto quantize = [&](const std::vector<uint16_t>& x, std::vector<int8_t>& c, std::vector<uint16_t>& sc,
                        std::vector<uint16_t>& dq) {
        for (size_t b = 0; b < sc.size(); ++b) {
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(h2f(x[b * 32 + i])));
            const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
            sc[b] = f2h(d);
            for (int i = 0; i < 32; ++i) {
                c[b * 32 + i] = (int8_t) std::lround(h2f(x[b * 32 + i]) * id);
                dq[b * 32 + i] = f2h((float) c[b * 32 + i] * h2f(sc[b]));
            }
        }
    };
    quantize(k, kc, ks, kd);
    quantize(v, vc, vs, vd);
    uint16_t *dk = upload(k), *dv = upload(v), *dks = upload(ks), *dvs = upload(vs);
    int8_t *dkc = upload(kc), *dvc = upload(vc);
    const size_t whole = size_t(kCtx) * width, part = size_t(2048) * width;
    uint16_t* dqw = nullptr;   // the workspaces: K's, then V's
    cudaMalloc(&dqw, 2 * whole * sizeof(uint16_t));
    float* dq = upload(q);
    float *dout = nullptr, *dscore = nullptr, *dwork = nullptr;
    const size_t score_n = size_t(64) << 20, work_n = size_t(max_rows) * kHeads * (kDim + kDim / 2 + 3) + 1;
    cudaMalloc(&dout, q.size() * sizeof(float));
    cudaMalloc(&dscore, score_n * sizeof(float));
    cudaMalloc(&dwork, work_n * sizeof(float));
    cudaStream_t st;
    cudaStreamCreate(&st);
    void* blas = blas_create(st);
    const float scale = 1.f / 16.f;
    struct Case { long long pos; int rows; };
    const Case cases[] = {{0, 1024}, {3000, 1024}, {0, 37}, {511, 200}, {7000, 1024}};
    int fails = 0;
    const char* modes[] = {"fp16", "q8 whole", "q8 2048"};
    for (int mode = 0; mode < 3; ++mode)
    for (const Case& c : cases) {
        const bool q8 = mode > 0;
        const size_t halves = mode == 1 ? whole : part;
        const std::vector<uint16_t>& kr = q8 ? kd : k;
        const std::vector<uint16_t>& vr = q8 ? vd : v;
        auto call = [&] {
            if (q8)
                attend_batch(blas, dq, nullptr, nullptr, c.pos, c.rows, dscore, score_n, dwork, dout, kDim, kHeads,
                             kKvHeads, scale, st, 0, 0, false, Kvq8{dkc, dks}, Kvq8{dvc, dvs}, dqw, dqw + halves,
                             halves);
            else
                attend_batch(blas, dq, dk, dv, c.pos, c.rows, dscore, score_n, dwork, dout, kDim, kHeads, kKvHeads,
                             scale, st);
        };
        call();
        cudaEvent_t a, b;
        cudaEventCreate(&a), cudaEventCreate(&b);
        cudaEventRecord(a, st);
        for (int r = 0; r < 5; ++r) call();
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        std::vector<float> out(size_t(c.rows) * kHeads * kDim);
        cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double worst = 0.0, sum_err = 0.0;
        size_t n_err = 0;
        std::vector<double> s(size_t(c.pos) + c.rows);
        for (int t = 0; t < c.rows; t += 7) {
            const long long own = c.pos + t;
            for (int h = 0; h < kHeads; ++h) {
                const int g = h / per;
                const float* qr = &q[(size_t(t) * kHeads + h) * kDim];
                double mx = -1e300;
                for (long long j = 0; j <= own; ++j) {
                    double d = 0.0;
                    for (int i = 0; i < kDim; ++i)
                        d += double(h2f(f2h(qr[i] * scale * 1.44269504f))) * h2f(kr[(size_t(j) * kKvHeads + g) * kDim + i]);
                    s[size_t(j)] = d;
                    mx = std::max(mx, d);
                }
                double total = 0.0;
                for (long long j = 0; j <= own; ++j) total += s[size_t(j)] = std::exp2(s[size_t(j)] - mx);
                for (int i = 0; i < kDim; ++i) {
                    double y = 0.0;
                    for (long long j = 0; j <= own; ++j) y += s[size_t(j)] * h2f(vr[(size_t(j) * kKvHeads + g) * kDim + i]);
                    y /= total;
                    const double err = std::fabs(y - out[(size_t(t) * kHeads + h) * kDim + i]);
                    worst = std::max(worst, err);
                    sum_err += err, ++n_err;
                }
            }
        }
        const bool ok = worst <= 5e-3;
        fails += !ok;
        std::printf("%s %-8s pos %5lld rows %4d: max error %.2e, mean %.2e; %.2f ms a call\n", ok ? "ok  " : "FAIL",
                    modes[mode], c.pos, c.rows, worst, sum_err / double(n_err), ms / 5.f);
    }
    blas_destroy(blas);
    return fails ? 1 : 0;
}
