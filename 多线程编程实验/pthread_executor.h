#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <pthread.h>
#include <utility>
#include <vector>

namespace svd_parallel
{
    class Executor
    {
    public:
        Executor()
            : thread_count_(detect_thread_count()), stopping_(false), generation_(0),
              completed_workers_(0), range_begin_(0), range_end_(0),
              dynamic_tasks_(false), task_count_(0), next_task_(0)
        {
            pthread_mutex_init(&mutex_, nullptr);
            pthread_cond_init(&work_ready_, nullptr);
            pthread_cond_init(&work_done_, nullptr);

            if (thread_count_ <= 1)
            {
                return;
            }

            threads_.resize(thread_count_ - 1);
            args_.resize(thread_count_ - 1);
            for (int i = 1; i < thread_count_; ++i)
            {
                args_[i - 1] = {this, i};
                pthread_create(&threads_[i - 1], nullptr, &Executor::worker_entry, &args_[i - 1]);
            }
        }

        ~Executor()
        {
            pthread_mutex_lock(&mutex_);
            stopping_ = true;
            pthread_cond_broadcast(&work_ready_);
            pthread_mutex_unlock(&mutex_);

            for (pthread_t thread : threads_)
            {
                pthread_join(thread, nullptr);
            }

            pthread_cond_destroy(&work_done_);
            pthread_cond_destroy(&work_ready_);
            pthread_mutex_destroy(&mutex_);
        }

        Executor(const Executor &) = delete;
        Executor &operator=(const Executor &) = delete;

        int thread_count() const
        {
            return thread_count_;
        }

        template <typename Function>
        void parallel_for(int begin, int end, long long work_items, Function &&function)
        {
            if (end <= begin)
            {
                return;
            }

            if (thread_count_ <= 1 || work_items < 20000)
            {
                function(begin, end);
                return;
            }

            std::function<void(int, int)> local_job(std::forward<Function>(function));

            pthread_mutex_lock(&mutex_);
            job_ = local_job;
            task_job_ = nullptr;
            dynamic_tasks_ = false;
            range_begin_ = begin;
            range_end_ = end;
            completed_workers_ = 0;
            ++generation_;
            pthread_cond_broadcast(&work_ready_);
            pthread_mutex_unlock(&mutex_);

            int local_begin = 0;
            int local_end = 0;
            partition(0, begin, end, local_begin, local_end);
            if (local_begin < local_end)
            {
                local_job(local_begin, local_end);
            }

            pthread_mutex_lock(&mutex_);
            while (completed_workers_ < thread_count_ - 1)
            {
                pthread_cond_wait(&work_done_, &mutex_);
            }
            job_ = nullptr;
            pthread_mutex_unlock(&mutex_);
        }

        template <typename Function>
        void parallel_tasks(int task_count, long long work_items, Function &&function)
        {
            if (task_count <= 0)
            {
                return;
            }

            std::function<void(int)> local_job(std::forward<Function>(function));
            if (thread_count_ <= 1 || task_count == 1 || work_items < 20000)
            {
                for (int task = 0; task < task_count; ++task)
                {
                    local_job(task);
                }
                return;
            }

            pthread_mutex_lock(&mutex_);
            job_ = nullptr;
            task_job_ = local_job;
            dynamic_tasks_ = true;
            task_count_ = task_count;
            next_task_.store(0, std::memory_order_relaxed);
            completed_workers_ = 0;
            ++generation_;
            pthread_cond_broadcast(&work_ready_);
            pthread_mutex_unlock(&mutex_);

            run_tasks(local_job, task_count);

            pthread_mutex_lock(&mutex_);
            while (completed_workers_ < thread_count_ - 1)
            {
                pthread_cond_wait(&work_done_, &mutex_);
            }
            task_job_ = nullptr;
            dynamic_tasks_ = false;
            pthread_mutex_unlock(&mutex_);
        }

    private:
        struct WorkerArg
        {
            Executor *executor;
            int index;
        };

        static int parse_positive(const char *value)
        {
            if (value == nullptr || *value == '\0')
            {
                return 0;
            }

            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || parsed <= 0)
            {
                return 0;
            }

            return static_cast<int>(std::min(parsed, 64L));
        }

        static int nodefile_slots()
        {
            const char *path = std::getenv("PBS_NODEFILE");
            if (path == nullptr)
            {
                return 0;
            }

            FILE *file = std::fopen(path, "r");
            if (file == nullptr)
            {
                return 0;
            }

            int slots = 0;
            char line[256];
            while (std::fgets(line, sizeof(line), file) != nullptr)
            {
                ++slots;
            }
            std::fclose(file);
            return std::min(slots, 64);
        }

        static int detect_thread_count()
        {
            const char *variables[] = {"SVD_NUM_THREADS", "PBS_NP", "NCPUS", "OMP_NUM_THREADS"};
            for (const char *variable : variables)
            {
                const int value = parse_positive(std::getenv(variable));
                if (value > 0)
                {
                    return value;
                }
            }

            const int slots = nodefile_slots();
            return (slots > 0) ? slots : 1;
        }

        void partition(int worker, int begin, int end, int &worker_begin, int &worker_end) const
        {
            const int length = end - begin;
            const int base = length / thread_count_;
            const int extra = length % thread_count_;
            worker_begin = begin + worker * base + std::min(worker, extra);
            worker_end = worker_begin + base + ((worker < extra) ? 1 : 0);
        }

        static void *worker_entry(void *parameter)
        {
            WorkerArg *arg = static_cast<WorkerArg *>(parameter);
            arg->executor->worker_loop(arg->index);
            return nullptr;
        }

        void run_tasks(const std::function<void(int)> &job, int task_count)
        {
            while (true)
            {
                const int task = next_task_.fetch_add(1, std::memory_order_relaxed);
                if (task >= task_count)
                {
                    return;
                }
                job(task);
            }
        }

        void worker_loop(int worker)
        {
            int observed_generation = 0;
            pthread_mutex_lock(&mutex_);
            while (true)
            {
                while (!stopping_ && observed_generation == generation_)
                {
                    pthread_cond_wait(&work_ready_, &mutex_);
                }

                if (stopping_)
                {
                    pthread_mutex_unlock(&mutex_);
                    return;
                }

                observed_generation = generation_;
                const bool dynamic_tasks = dynamic_tasks_;
                const std::function<void(int, int)> local_job = job_;
                const std::function<void(int)> local_task_job = task_job_;
                const int task_count = task_count_;
                int worker_begin = 0;
                int worker_end = 0;
                if (!dynamic_tasks)
                {
                    partition(worker, range_begin_, range_end_, worker_begin, worker_end);
                }

                pthread_mutex_unlock(&mutex_);
                if (dynamic_tasks)
                {
                    run_tasks(local_task_job, task_count);
                }
                else if (worker_begin < worker_end)
                {
                    local_job(worker_begin, worker_end);
                }
                pthread_mutex_lock(&mutex_);

                ++completed_workers_;
                if (completed_workers_ == thread_count_ - 1)
                {
                    pthread_cond_signal(&work_done_);
                }
            }
        }

        int thread_count_;
        bool stopping_;
        int generation_;
        int completed_workers_;
        int range_begin_;
        int range_end_;
        bool dynamic_tasks_;
        int task_count_;
        std::atomic<int> next_task_;
        pthread_mutex_t mutex_;
        pthread_cond_t work_ready_;
        pthread_cond_t work_done_;
        std::function<void(int, int)> job_;
        std::function<void(int)> task_job_;
        std::vector<pthread_t> threads_;
        std::vector<WorkerArg> args_;
    };

    inline Executor &executor()
    {
        static Executor instance;
        return instance;
    }
}
