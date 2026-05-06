#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define SVD_USE_NEON 1
#else
#define SVD_USE_NEON 0
#endif

namespace
{
    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，可以认为通过这个结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    //
    // 原始版本做法：
    //   对整行所有列执行旋转更新。
    //
    // 本版优化思路：
    //   该函数只用于更新 GKH 迭代中的 B 矩阵。
    //   B 在整个 GKH 过程中始终接近“上二对角 + 局部 bulge”的带状结构，
    //   因此没有必要扫描整行所有列，只需要更新 r0/r1 附近可能受影响的局部窗口。
    //
    // 优化意义：
    //   减少大量无效访存，降低 apply_left_rows 的热点占比。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        const int n = M.cols();

        // 对于上二对角结构，左右乘 Givens 后的非零扰动只会出现在局部范围内。
        // 这里保守地取 [r0-1, r1+2]，既保证正确性，又避免整行扫描。
        const int j_begin = std::max(0, r0 - 1);
        const int j_end = std::min(n - 1, r1 + 2);

        double *row0 = &M.at(r0, 0);
        double *row1 = &M.at(r1, 0);

        for (int j = j_begin; j <= j_end; ++j)
        {
            double a = row0[j];
            double b = row1[j];

            row0[j] = c * a + s * b;
            row1[j] = -s * a + c * b;
        }
    }

    // 对稠密矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    //
    // 原始版本做法：
    //   for i in rows:
    //       a = M(i,c0)
    //       b = M(i,c1)
    //       M(i,c0) = a*c - b*s
    //       M(i,c1) = a*s + b*c
    //
    // 该函数主要用于更新 U/V 这样的稠密正交矩阵。
    // 注意：U/V 是完整正交矩阵，不能像 B 一样只更新局部窗口，必须整列更新。
    //
    // 本版优化点：
    //   1. 使用行指针减少 Matrix::at(i,j) 重复下标计算；
    //   2. 在 ARM aarch64/NEON 平台上，对相邻列旋转使用 NEON 双精度向量；
    //   3. NEON 一次处理两行的同一对列元素，体现 SIMD 并行思想；
    //   4. 保留非 NEON / 非相邻列情况下的标量路径，保证可移植性。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        const int rows = M.rows();
        const int cols = M.cols();
        const int offset = c1 - c0;

#if SVD_USE_NEON
        // GKH 中绝大多数列旋转都是相邻列，即 c1 == c0 + 1。
        // 由于 Matrix 按行连续存储，所以同一行的 p[0] 和 p[1] 连续。
        // 这里一次把两行的 c0/c1 元素分别装入 NEON 寄存器：
        //   a = [row_i_c0, row_{i+1}_c0]
        //   b = [row_i_c1, row_{i+1}_c1]
        // 然后并行计算：
        //   new0 = a*c - b*s
        //   new1 = a*s + b*c
        if (offset == 1)
        {
            const float64x2_t vc = vdupq_n_f64(c);
            const float64x2_t vs = vdupq_n_f64(s);

            int i = 0;
            for (; i + 1 < rows; i += 2)
            {
                double *p0 = &M.at(i, c0);
                double *p1 = &M.at(i + 1, c0);

                float64x2_t a = vdupq_n_f64(0.0);
                float64x2_t b = vdupq_n_f64(0.0);

                a = vsetq_lane_f64(p0[0], a, 0);
                a = vsetq_lane_f64(p1[0], a, 1);

                b = vsetq_lane_f64(p0[1], b, 0);
                b = vsetq_lane_f64(p1[1], b, 1);

                float64x2_t new0 = vsubq_f64(vmulq_f64(a, vc), vmulq_f64(b, vs));
                float64x2_t new1 = vaddq_f64(vmulq_f64(a, vs), vmulq_f64(b, vc));

                p0[0] = vgetq_lane_f64(new0, 0);
                p1[0] = vgetq_lane_f64(new0, 1);

                p0[1] = vgetq_lane_f64(new1, 0);
                p1[1] = vgetq_lane_f64(new1, 1);
            }

            // 处理奇数行尾部。
            for (; i < rows; ++i)
            {
                double *p = &M.at(i, c0);

                double a = p[0];
                double b = p[1];

                p[0] = a * c - b * s;
                p[1] = a * s + b * c;
            }

            return;
        }
#endif

        // 通用标量路径：用于非 NEON 平台或非相邻列情况。
        double *p = &M.at(0, c0);

        for (int i = 0; i < rows; ++i)
        {
            double a = p[0];
            double b = p[offset];

            p[0] = a * c - b * s;
            p[offset] = a * s + b * c;

            p += cols;
        }
    }

    // 专门用于 B 矩阵的右乘列更新。
    //
    // 重要区别：
    //   apply_right_cols      用于 U/V，必须更新完整列；
    //   apply_right_cols_banded 用于 B，只更新带状局部区域。
    //
    // 原始版本中 B 的右乘更新也会扫描整个列方向，导致 apply_right_cols 成为最大热点。
    // 但 B 本身是上二对角矩阵，Givens 旋转只会在局部引入/追赶 bulge。
    // 因此对 B 的更新可以限制在 c0/c1 附近的少数行。
    static void apply_right_cols_banded(Matrix &M, int c0, int c1, double c, double s)
    {
        const int rows = M.rows();
        const int offset = c1 - c0;

        // 保守局部窗口：覆盖上二对角元素以及 bulge 可能出现的位置。
        const int i_begin = std::max(0, c0 - 1);
        const int i_end = std::min(rows - 1, c1 + 1);

        for (int i = i_begin; i <= i_end; ++i)
        {
            double *p = &M.at(i, c0);

            double a = p[0];
            double b = p[offset];

            p[0] = a * c - b * s;
            p[offset] = a * s + b * c;
        }
    }

    // 当 B <- L * B 时，为保持 A = U * B * V^T 不变，
    // 需要同步更新 U <- U * L^T。
    //
    // L = [ c  s ]
    //     [-s  c ]
    //
    // L^T = [ c -s ]
    //       [ s  c ]
    //
    // 所以这里复用 apply_right_cols，并传入 -s。
    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // Wilkinson 偏移用于加速 QR/GKH 迭代收敛。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;

        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;

        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 清理 B 的上二对角结构外的小量数值噪声。
    //
    // 原始版本：
    //   每轮扫描整个 m*n 矩阵。
    //
    // 本版优化：
    //   GKH 的 bulge chasing 只在上二对角附近产生局部扰动，
    //   因此只扫描对角线附近的小窗口。
    //
    // 该优化对应 perf 中 cleanup_bidiagonal 热点下降。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < m; ++i)
        {
            const int j_begin = std::max(0, i - 2);
            const int j_end = std::min(n - 1, i + 3);

            for (int j = j_begin; j <= j_end; ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次单块 GKH bulge chasing 迭代。
    //
    // 流程：
    //   1. 计算 Wilkinson shift；
    //   2. 首次右乘 Givens，引入 bulge；
    //   3. 左乘 Givens，消去 bulge；
    //   4. 交替右乘/左乘，把 bulge 向右下角追赶。
    //
    // 本版优化的关键：
    //   B 使用带状局部更新；
    //   U/V 仍作为稠密正交矩阵完整更新，并在 apply_right_cols 内部使用 NEON。
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2 - mu, d_l * e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);

        givens_rotation(x, z, c, s, rr, false);

        apply_right_cols_banded(B, l, l + 1, c, s);
        apply_right_cols(V, l, l + 1, c, s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);

        apply_left_rows(B, l, l + 1, c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);

            apply_right_cols_banded(B, k, k + 1, c, s);
            apply_right_cols(V, k, k + 1, c, s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);

            apply_left_rows(B, k, k + 1, c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    // 处理“主对角元 d_k 近零但超对角 e_k 未近零”的特殊情形。
    //
    // 该过程通过连续 Givens 旋转把异常结构向右追赶，
    // 最终恢复为可以分块处理的上二对角形式。
    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();

        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;

        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);

            apply_right_cols_banded(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);

                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置。
    // 若对应 e_k 尚未收敛，则调用 chase_zero_diagonal 进行处理。
    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol &&
                std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否足够小对问题进行分块。
    //
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置已经收敛，
    // 可以置零并把问题拆分为两个独立子块。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);

            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;

        int l = 0;
        while (l < n)
        {
            int r = l;

            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }

            blocks.push_back({l, r});
            l = r + 1;
        }

        return blocks;
    }

    // 收尾步骤：
    //   1. 将奇异值调整为非负；
    //   2. 按降序排列奇异值；
    //   3. 同步重排 U/V 对应列。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);

                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);

        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }

        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];

            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }

            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代。
//
// 输入输出始终维护：
//   A = U * B * V^T
//
// 若收敛，B 最终被整理为非负降序的对角矩阵，
// 对角线元素即奇异值。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }

    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }

    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        std::vector<Block> blocks = split_active_blocks(B, n, tol);

        bool all_singletons = true;

        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                all_singletons = false;
                break;
            }
        }

        if (all_singletons)
        {
            converged = true;
            break;
        }

        // 从右向左处理活动块，减少末端块对前面块的影响。
        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
        {
            if (blocks[i].r > blocks[i].l)
            {
                one_block_step(U, B, V, blocks[i].l, blocks[i].r);
            }
        }
    }

    cleanup_bidiagonal(B, tol);

    // 理论上收敛后超对角线已经接近 0，这里显式置零，方便后续检查。
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }

    make_nonnegative_and_sort(U, B, V);

    return converged;
}
