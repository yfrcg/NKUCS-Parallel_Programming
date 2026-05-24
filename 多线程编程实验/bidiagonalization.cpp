// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证 m >= n）通过 Householder 变换化为上双对角形。
//
// 算法说明：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步：
//   1. 从左侧作用 Householder，消去第 k 列对角线以下元素；
//   2. 从右侧作用 Householder，消去第 k 行中 k+2 及之后元素。
//
// 本组件输出：
//   A = U * B * V^T
//
// 其中：
//   U 为 m×m 正交矩阵；
//   V 为 n×n 正交矩阵；
//   B 为 m×n 上双对角矩阵。
//
// 本版 SIMD 优化点：
//   1. vector_norm 使用 SIMD 点积；
//   2. Householder 中的 v^T v 使用 SIMD 点积；
//   3. 连续内存上的向量更新使用 SIMD AXPY；
//   4. 将原本按列访问的 w[j] 累加改为按行连续访问，改善 cache 行为；
//   5. 数学流程保持不变，仍然维护 A = U * B * V^T。

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define SVD_USE_NEON 1
#else
#define SVD_USE_NEON 0
#endif

// SIMD 点积：sum a[i] * b[i]
//
// ARM NEON 128-bit 寄存器一次可处理两个 double。
// 在 Kunpeng 920 这类 aarch64 平台上，该函数可用于加速：
//   1. 向量范数计算；
//   2. Householder 中 v^T v；
//   3. 行向量与 Householder 向量的内积。
static double dot_product_simd(const double *a, const double *b, int len)
{
    double sum = 0.0;

#if SVD_USE_NEON
    int i = 0;
    float64x2_t acc = vdupq_n_f64(0.0);

    for (; i + 1 < len; i += 2)
    {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        acc = vaddq_f64(acc, vmulq_f64(va, vb));
    }

    sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

    for (; i < len; ++i)
    {
        sum += a[i] * b[i];
    }
#else
    for (int i = 0; i < len; ++i)
    {
        sum += a[i] * b[i];
    }
#endif

    return sum;
}

// SIMD AXPY：dst[i] += scale * src[i]
//
// AXPY 是数值计算中非常常见的向量操作。
// Householder 更新中大量出现如下形式：
//   row = row - beta * coeff * v
//   w   = w + vi * row
//
// 这些都是连续内存上的向量加乘，非常适合 SIMD。
static void add_scaled_simd(double *dst, const double *src, int len, double scale)
{
#if SVD_USE_NEON
    int i = 0;
    float64x2_t vscale = vdupq_n_f64(scale);

    for (; i + 1 < len; i += 2)
    {
        float64x2_t vd = vld1q_f64(dst + i);
        float64x2_t vs = vld1q_f64(src + i);
        vd = vaddq_f64(vd, vmulq_f64(vscale, vs));
        vst1q_f64(dst + i, vd);
    }

    for (; i < len; ++i)
    {
        dst[i] += scale * src[i];
    }
#else
    for (int i = 0; i < len; ++i)
    {
        dst[i] += scale * src[i];
    }
#endif
}

// 计算向量二范数。
// 原始版本手写 for 循环求平方和；
// 本版复用 dot_product_simd(v, v)。
static double vector_norm(const std::vector<double> &v)
{
    if (v.empty())
    {
        return 0.0;
    }

    return std::sqrt(dot_product_simd(v.data(), v.data(), static_cast<int>(v.size())));
}

// 将 m×n 矩阵 A（m >= n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）。
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();

    Matrix B = A;

    // 初始化 U = I_m
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
    {
        U.at(i, i) = 1.0;
    }

    // 初始化 V = I_n
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        V.at(i, i) = 1.0;
    }

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1：从左侧作用 Householder 变换
        // 目标：消去第 k 列中对角线以下元素。
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量 x。
        std::vector<double> x(m - k);

        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // 构造 Householder 向量 v = x + sigma * e1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            std::vector<double> v(x);
            v[0] += sigma;

            double vTv = dot_product_simd(v.data(), v.data(), static_cast<int>(v.size()));

            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 左侧 Householder：
                //   B_new = B_old - beta * v * (v^T * B_old)
                //
                // 原始写法通常是：
                //   for j:
                //       for i:
                //           w[j] += v[i] * B[k+i][k+j]
                //
                // 这种写法会按列访问 B，不利于 cache。
                //
                // 本版改成：
                //   for i:
                //       w += v[i] * B_row
                //
                // B_row 是连续内存，可以用 SIMD AXPY。
                std::vector<double> w(n - k, 0.0);

                for (int i = 0; i < m - k; ++i)
                {
                    const double vi = v[i];
                    const double *brow = &B.at(k + i, k);
                    add_scaled_simd(w.data(), brow, n - k, vi);
                }

                // 更新 B(k:m, k:n)：
                //   B_row -= beta * v[i] * w
                for (int i = 0; i < m - k; ++i)
                {
                    double *brow = &B.at(k + i, k);
                    const double scale = -beta * v[i];
                    add_scaled_simd(brow, w.data(), n - k, scale);
                }

                // 累积 U：
                //   U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                //
                // 每一行 U(i, k:m) 是连续内存，
                // 因此内积和向量更新都可以用 SIMD。
                std::vector<double> wU(m, 0.0);

                for (int i = 0; i < m; ++i)
                {
                    const double *urow = &U.at(i, k);
                    wU[i] = dot_product_simd(urow, v.data(), m - k);
                }

                for (int i = 0; i < m; ++i)
                {
                    double *urow = &U.at(i, k);
                    const double scale = -beta * wU[i];
                    add_scaled_simd(urow, v.data(), m - k, scale);
                }
            }
        }

        // 理论上第 k 列对角线以下元素已为 0。
        // 为避免浮点误差影响后续结构判断，这里显式置零。
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2：从右侧作用 Householder 变换
        // 目标：消去第 k 行中 k+2 及之后的元素。
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从 k+1 到末尾的子向量 y。
            std::vector<double> y(n - k - 1);

            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = dot_product_simd(v.data(), v.data(), static_cast<int>(v.size()));

                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 右侧 Householder：
                    //   B_new = B_old - beta * (B_old * v) * v^T
                    //
                    // 对每一行 B(k+i, k+1:n)，它在内存中是连续的，
                    // 因此 dot 和 AXPY 都适合 SIMD。
                    std::vector<double> w(m - k, 0.0);

                    for (int i = 0; i < m - k; ++i)
                    {
                        const double *brow = &B.at(k + i, k + 1);
                        w[i] = dot_product_simd(brow, v.data(), n - k - 1);
                    }

                    for (int i = 0; i < m - k; ++i)
                    {
                        double *brow = &B.at(k + i, k + 1);
                        const double scale = -beta * w[i];
                        add_scaled_simd(brow, v.data(), n - k - 1, scale);
                    }

                    // 累积 V：
                    //   V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);

                    for (int i = 0; i < n; ++i)
                    {
                        const double *vrow = &V.at(i, k + 1);
                        wV[i] = dot_product_simd(vrow, v.data(), n - k - 1);
                    }

                    for (int i = 0; i < n; ++i)
                    {
                        double *vrow = &V.at(i, k + 1);
                        const double scale = -beta * wV[i];
                        add_scaled_simd(vrow, v.data(), n - k - 1, scale);
                    }
                }
            }

            // 理论上第 k 行 k+2 之后元素已为 0。
            // 为避免浮点误差影响后续结构判断，这里显式置零。
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}