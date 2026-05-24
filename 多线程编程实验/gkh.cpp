#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <pthread.h>
#include <stdexcept>
#include <string>
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
    struct Block
    {
        int l;
        int r;
    };


    constexpr int MIN_PARALLEL_BLOCKS = 2;
    constexpr int MIN_BLOCK_LEN = 32;
    constexpr int MIN_TOTAL_WORK = 256;
    constexpr int MAX_PTHREAD_WORKERS = 64;

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
        DynamicPool,
        StaticSpawn
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

        return static_cast<int>(std::min<long>(v, MAX_PTHREAD_WORKERS));
    }

    static int detect_thread_count_once()
    {
        int v = parse_positive_env("PTHREAD_NUM_THREADS");
        if (v > 0)
        {
            return v;
        }

        v = parse_positive_env("OMP_NUM_THREADS");
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
                return std::min(count, MAX_PTHREAD_WORKERS);
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
            return ScheduleMode::StaticSpawn;
        }

        return ScheduleMode::DynamicPool;
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

#if SVD_USE_NEON
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

    struct StaticWorkerArg
    {
        Matrix *U = nullptr;
        Matrix *B = nullptr;
        Matrix *V = nullptr;
        const std::vector<Task> *tasks = nullptr;
        ThreadStats *stats = nullptr;
        int tid = 0;
        int worker_count = 1;
        bool collect_stats = false;
    };

    static void *static_worker_entry(void *arg)
    {
        StaticWorkerArg *worker = static_cast<StaticWorkerArg *>(arg);

        for (int id = worker->tid; id < static_cast<int>(worker->tasks->size()); id += worker->worker_count)
        {
            run_one_task(*worker->U, *worker->B, *worker->V,
                         (*worker->tasks)[id], worker->collect_stats,
                         *worker->stats);
        }

        return nullptr;
    }

    static void run_tasks_static_spawn(Matrix &U, Matrix &B, Matrix &V,
                                       const std::vector<Task> &tasks,
                                       std::vector<ThreadStats> &stats,
                                       int worker_count, bool collect_stats)
    {
        std::vector<pthread_t> threads(worker_count);
        std::vector<StaticWorkerArg> args(worker_count);
        int created = 0;

        for (int tid = 0; tid < worker_count; ++tid)
        {
            args[tid].U = &U;
            args[tid].B = &B;
            args[tid].V = &V;
            args[tid].tasks = &tasks;
            args[tid].stats = &stats[tid];
            args[tid].tid = tid;
            args[tid].worker_count = worker_count;
            args[tid].collect_stats = collect_stats;

            const int rc = pthread_create(&threads[tid], nullptr, &static_worker_entry, &args[tid]);
            if (rc != 0)
            {
                for (int i = 0; i < created; ++i)
                {
                    pthread_join(threads[i], nullptr);
                }
                throw std::runtime_error("static GKH scheduler: pthread_create failed");
            }
            ++created;
        }

        for (int tid = 0; tid < worker_count; ++tid)
        {
            pthread_join(threads[tid], nullptr);
        }
    }

    class GkhBlockThreadPoolV1
    {
    public:
        explicit GkhBlockThreadPoolV1(int worker_count, bool collect_stats)
            : worker_count_(std::max(1, worker_count)),
              collect_stats_(collect_stats),
              threads_(worker_count_),
              args_(worker_count_),
              stats_(worker_count_)
        {
            pthread_mutex_init(&mutex_, nullptr);
            pthread_cond_init(&work_available_, nullptr);
            pthread_cond_init(&batch_done_, nullptr);

            for (int i = 0; i < worker_count_; ++i)
            {
                args_[i].pool = this;
                args_[i].tid = i;

                int rc = pthread_create(&threads_[i], nullptr, &GkhBlockThreadPoolV1::thread_entry, &args_[i]);
                if (rc != 0)
                {
                    stop_created_threads(i);
                    throw std::runtime_error("GkhBlockThreadPoolV1: pthread_create failed");
                }
            }
        }

        ~GkhBlockThreadPoolV1()
        {
            stop();
            pthread_cond_destroy(&batch_done_);
            pthread_cond_destroy(&work_available_);
            pthread_mutex_destroy(&mutex_);
        }

        GkhBlockThreadPoolV1(const GkhBlockThreadPoolV1 &) = delete;
        GkhBlockThreadPoolV1 &operator=(const GkhBlockThreadPoolV1 &) = delete;

        void run_tasks(Matrix &U, Matrix &B, Matrix &V, const std::vector<Task> &tasks)
        {
            if (tasks.empty())
            {
                return;
            }

            pthread_mutex_lock(&mutex_);

            U_ = &U;
            B_ = &B;
            V_ = &V;
            tasks_ = &tasks;
            task_count_ = static_cast<int>(tasks.size());

            next_task_.store(0, std::memory_order_relaxed);
            remaining_tasks_.store(task_count_, std::memory_order_release);

            batch_finished_ = false;
            ++generation_;

            pthread_cond_broadcast(&work_available_);
            pthread_mutex_unlock(&mutex_);

            pthread_mutex_lock(&mutex_);
            while (!batch_finished_)
            {
                pthread_cond_wait(&batch_done_, &mutex_);
            }
            pthread_mutex_unlock(&mutex_);
        }

        void print_stats_if_requested() const
        {
            if (!collect_stats_)
            {
                return;
            }

            print_stats("dynamic-pool", stats_);
        }

    private:
        struct WorkerArg
        {
            GkhBlockThreadPoolV1 *pool = nullptr;
            int tid = 0;
        };

        static void *thread_entry(void *arg)
        {
            WorkerArg *worker_arg = static_cast<WorkerArg *>(arg);
            worker_arg->pool->worker_loop(worker_arg->tid);
            return nullptr;
        }

        void worker_loop(int tid)
        {
            int seen_generation = 0;

            while (true)
            {
                pthread_mutex_lock(&mutex_);

                while (!stop_ && generation_ == seen_generation)
                {
                    pthread_cond_wait(&work_available_, &mutex_);
                }

                if (stop_)
                {
                    pthread_mutex_unlock(&mutex_);
                    break;
                }

                seen_generation = generation_;
                pthread_mutex_unlock(&mutex_);

                while (true)
                {
                    int id = next_task_.fetch_add(1, std::memory_order_relaxed);

                    if (id >= task_count_)
                    {
                        break;
                    }

                    const Task &task = (*tasks_)[id];
                    run_one_task(*U_, *B_, *V_, task, collect_stats_, stats_[tid]);

                    if (remaining_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    {
                        pthread_mutex_lock(&mutex_);
                        batch_finished_ = true;
                        pthread_cond_signal(&batch_done_);
                        pthread_mutex_unlock(&mutex_);
                    }
                }
            }
        }

        void stop()
        {
            pthread_mutex_lock(&mutex_);

            if (!stop_)
            {
                stop_ = true;
                ++generation_;
                pthread_cond_broadcast(&work_available_);
            }

            pthread_mutex_unlock(&mutex_);

            for (pthread_t &th : threads_)
            {
                pthread_join(th, nullptr);
            }
        }

        void stop_created_threads(int created_count)
        {
            pthread_mutex_lock(&mutex_);
            stop_ = true;
            ++generation_;
            pthread_cond_broadcast(&work_available_);
            pthread_mutex_unlock(&mutex_);

            for (int i = 0; i < created_count; ++i)
            {
                pthread_join(threads_[i], nullptr);
            }
        }

        int worker_count_;
        bool collect_stats_;

        std::vector<pthread_t> threads_;
        std::vector<WorkerArg> args_;
        std::vector<ThreadStats> stats_;

        pthread_mutex_t mutex_;
        pthread_cond_t work_available_;
        pthread_cond_t batch_done_;

        bool stop_ = false;
        bool batch_finished_ = false;
        int generation_ = 0;

        Matrix *U_ = nullptr;
        Matrix *B_ = nullptr;
        Matrix *V_ = nullptr;

        const std::vector<Task> *tasks_ = nullptr;
        int task_count_ = 0;

        std::atomic<int> next_task_{0};
        std::atomic<int> remaining_tasks_{0};
    };

    static void run_tasks_v1(Matrix &U, Matrix &B, Matrix &V,
                             const std::vector<Task> &tasks,
                             std::unique_ptr<GkhBlockThreadPoolV1> &pool,
                             std::vector<ThreadStats> &static_stats,
                             bool &static_parallel_triggered,
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
            if (mode == ScheduleMode::StaticSpawn)
            {
                static_parallel_triggered = true;
                run_tasks_static_spawn(U, B, V, batch, static_stats,
                                       worker_count, collect_stats);
                continue;
            }

            if (pool == nullptr)
            {
                pool.reset(new GkhBlockThreadPoolV1(worker_count, collect_stats));
            }

            pool->run_tasks(U, B, V, batch);
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
    std::unique_ptr<GkhBlockThreadPoolV1> pool;
    std::vector<ThreadStats> static_stats(worker_count);
    bool static_parallel_triggered = false;

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

        run_tasks_v1(U, B, V, tasks, pool, static_stats, static_parallel_triggered,
                     worker_count, mode, collect_stats);
    }

    if (collect_stats && mode == ScheduleMode::StaticSpawn)
    {
        if (static_parallel_triggered)
        {
            print_stats("static-spawn", static_stats);
        }
        else
        {
            std::cerr << "[SVD_GKH_STATS] static-spawn: no parallel task batch triggered\n";
        }
    }
    else if (pool != nullptr)
    {
        pool->print_stats_if_requested();
    }
    else if (collect_stats)
    {
        std::cerr << "[SVD_GKH_STATS] dynamic-pool: no parallel task batch triggered\n";
    }

    cleanup_bidiagonal(B, tol);

    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }

    make_nonnegative_and_sort(U, B, V);

    return converged;
}
