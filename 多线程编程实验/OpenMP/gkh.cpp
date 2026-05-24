#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    struct Block
    {
        int l;
        int r;
    };


    constexpr int MIN_PARALLEL_BLOCKS = 2;
    constexpr int MIN_BLOCK_LEN = 32;
    constexpr int MIN_TOTAL_WORK = 256;
    constexpr int MAX_OMP_WORKERS = 64;

    struct Task
    {
        int l;
        int r;
        int len;
        long long work;
    };

    struct alignas(64) ThreadStats
    {
        long long task_count = 0;
        long long len_sum = 0;
        long long work_sum = 0;
        double time_ms = 0.0;
    };

    enum class ScheduleMode
    {
        Dynamic,
        Static
    };

    static int parse_positive_env(const char *name)
    {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
        {
            return 0;
        }

        char *end = nullptr;
        long v = std::strtol(value, &end, 10);
        if (end == value || v <= 0)
        {
            return 0;
        }

        return static_cast<int>(std::min<long>(v, MAX_OMP_WORKERS));
    }

    static int detect_thread_count_once()
    {
        int v = parse_positive_env("OMP_NUM_THREADS");
        if (v > 0)
        {
            return v;
        }

        const char *pbs = std::getenv("PBS_NODEFILE");
        if (pbs != nullptr && pbs[0] != '\0')
        {
            std::ifstream fin(pbs);
            std::string line;
            int count = 0;

            while (std::getline(fin, line))
            {
                if (!line.empty())
                {
                    ++count;
                }
            }

            if (count > 0)
            {
                return std::min(count, MAX_OMP_WORKERS);
            }
        }

        return 1;
    }

    static bool stats_enabled()
    {
        const char *v = std::getenv("SVD_GKH_STATS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }

    static ScheduleMode schedule_mode()
    {
        const char *v = std::getenv("SVD_GKH_SCHED");
        if (v != nullptr && std::string(v) == "static")
        {
            return ScheduleMode::Static;
        }

        return ScheduleMode::Dynamic;
    }

    // 对 B 的两行做左 Givens 旋转。B 近似带状，因此只更新局部窗口。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        const int n = M.cols();
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

    // 对稠密矩阵 U/V 的两列做右 Givens 旋转。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        const int rows = M.rows();
        const int cols = M.cols();
        const int offset = c1 - c0;

        // The two updated elements are adjacent within each row, but
        // consecutive rows are separated by the matrix stride. Manually
        // gathering rows into NEON vectors costs more than scalar pairs here.
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

    // 专门用于 B 的右乘列旋转，只更新局部带状窗口。
    static void apply_right_cols_banded(Matrix &M, int c0, int c1, double c, double s)
    {
        const int rows = M.rows();
        const int offset = c1 - c0;

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

    // 当 B <- L * B 时，为保持 A = U * B * V^T，需要 U <- U * L^T。
    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        apply_right_cols(U, r0, r1, c, -s);
    }

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

    // 对活动块 [l, r] 执行一次单块 GKH bulge chasing。
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

        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);

        givens_rotation(x, z, c, s, rr, false);

        apply_right_cols_banded(B, l, l + 1, c, s);
        apply_right_cols(V, l, l + 1, c, s);

        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);

        apply_left_rows(B, l, l + 1, c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);

            apply_right_cols_banded(B, k, k + 1, c, s);
            apply_right_cols(V, k, k + 1, c, s);

            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);

            apply_left_rows(B, k, k + 1, c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    static Task make_task(const Block &blk)
    {
        Task t;
        t.l = blk.l;
        t.r = blk.r;
        t.len = blk.r - blk.l + 1;
        t.work = std::max(0, t.len - 1);
        return t;
    }

    // Two adjacent blocks have disjoint U/V columns, but B updates include a
    // one-step halo around the block boundary and would write common entries.
    static bool tasks_conflict_on_B(const Task &a, const Task &b)
    {
        return a.l <= b.r + 1 && b.l <= a.r + 1;
    }

    static std::vector<std::vector<Task>> make_conflict_free_batches(const std::vector<Task> &tasks)
    {
        std::vector<std::vector<Task>> batches;

        for (const Task &task : tasks)
        {
            bool placed = false;

            for (std::vector<Task> &batch : batches)
            {
                bool conflicts = false;

                for (const Task &scheduled : batch)
                {
                    if (tasks_conflict_on_B(task, scheduled))
                    {
                        conflicts = true;
                        break;
                    }
                }

                if (!conflicts)
                {
                    batch.push_back(task);
                    placed = true;
                    break;
                }
            }

            if (!placed)
            {
                batches.push_back({task});
            }
        }

        return batches;
    }

    static bool should_parallelize_tasks(const std::vector<Task> &tasks, int worker_count)
    {
        if (worker_count <= 1)
        {
            return false;
        }

        if (static_cast<int>(tasks.size()) < MIN_PARALLEL_BLOCKS)
        {
            return false;
        }

        int total_len = 0;
        int max_len = 0;

        for (const Task &task : tasks)
        {
            total_len += task.len;
            max_len = std::max(max_len, task.len);
        }

        return max_len >= MIN_BLOCK_LEN && total_len >= MIN_TOTAL_WORK;
    }

    static void run_tasks_serial(Matrix &U, Matrix &B, Matrix &V, const std::vector<Task> &tasks)
    {
        for (const Task &task : tasks)
        {
            one_block_step(U, B, V, task.l, task.r);
        }
    }

    static void run_one_task(Matrix &U, Matrix &B, Matrix &V,
                             const Task &task, bool collect_stats,
                             ThreadStats &stats)
    {
        if (collect_stats)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            one_block_step(U, B, V, task.l, task.r);
            const auto t1 = std::chrono::high_resolution_clock::now();

            stats.task_count += 1;
            stats.len_sum += task.len;
            stats.work_sum += task.work;
            stats.time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            return;
        }

        one_block_step(U, B, V, task.l, task.r);
    }

    static void print_stats(const char *label, const std::vector<ThreadStats> &stats)
    {
        std::cerr << "[SVD_GKH_STATS] " << label << " worker statistics:\n";

        for (int i = 0; i < static_cast<int>(stats.size()); ++i)
        {
            std::cerr << "  worker " << i
                      << " tasks=" << stats[i].task_count
                      << " len_sum=" << stats[i].len_sum
                      << " work_sum=" << stats[i].work_sum
                      << " time_ms=" << stats[i].time_ms
                      << "\n";
        }
    }

    static void run_tasks_openmp(Matrix &U, Matrix &B, Matrix &V,
                                 const std::vector<Task> &tasks,
                                 std::vector<ThreadStats> &stats,
                                 bool &parallel_triggered,
                                 int worker_count, ScheduleMode mode,
                                 bool collect_stats)
    {
        if (!should_parallelize_tasks(tasks, worker_count))
        {
            run_tasks_serial(U, B, V, tasks);
            return;
        }

        std::vector<Task> sorted_tasks = tasks;

        std::sort(sorted_tasks.begin(), sorted_tasks.end(), [](const Task &a, const Task &b)
                  {
                      if (a.len != b.len)
                      {
                          return a.len > b.len;
                      }
                      return a.l > b.l;
                  });

        const std::vector<std::vector<Task>> batches = make_conflict_free_batches(sorted_tasks);

        for (const std::vector<Task> &batch : batches)
        {
            if (batch.size() <= 1)
            {
                run_tasks_serial(U, B, V, batch);
                continue;
            }

            parallel_triggered = true;

            if (mode == ScheduleMode::Static)
            {
#pragma omp parallel for schedule(static, 1) num_threads(worker_count)
                for (int id = 0; id < static_cast<int>(batch.size()); ++id)
                {
                    const int tid = omp_get_thread_num();
                    run_one_task(U, B, V, batch[id], collect_stats, stats[tid]);
                }
            }
            else
            {
#pragma omp parallel for schedule(dynamic, 1) num_threads(worker_count)
                for (int id = 0; id < static_cast<int>(batch.size()); ++id)
                {
                    const int tid = omp_get_thread_num();
                    run_one_task(U, B, V, batch[id], collect_stats, stats[tid]);
                }
            }
        }
    }

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

bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v1: requires m >= n");
    }

    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v1: U must be m x m");
    }

    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v1: V must be n x n");
    }

    bool converged = false;

    const int worker_count = detect_thread_count_once();
    const bool collect_stats = stats_enabled();
    const ScheduleMode mode = schedule_mode();
    std::vector<ThreadStats> stats(worker_count);
    bool parallel_triggered = false;

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

        std::vector<Task> tasks;
        tasks.reserve(blocks.size());

        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
        {
            if (blocks[i].r > blocks[i].l)
            {
                tasks.push_back(make_task(blocks[i]));
            }
        }

        run_tasks_openmp(U, B, V, tasks, stats, parallel_triggered,
                         worker_count, mode, collect_stats);
    }

    if (collect_stats && parallel_triggered)
    {
        print_stats(mode == ScheduleMode::Static ? "openmp-static" : "openmp-dynamic", stats);
    }
    else if (collect_stats)
    {
        std::cerr << "[SVD_GKH_STATS] "
                  << (mode == ScheduleMode::Static ? "openmp-static" : "openmp-dynamic")
                  << ": no parallel task batch triggered\n";
    }

    cleanup_bidiagonal(B, tol);

    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }

    make_nonnegative_and_sort(U, B, V);

    return converged;
}
