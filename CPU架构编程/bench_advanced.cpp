#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <windows.h>

using namespace std;
namespace fs = std::filesystem;

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

static volatile double g_sink = 0.0;

inline void consumeScalar(double x) {
    g_sink += x;
}

inline void consumeVector(const vector<double>& v) {
    if (!v.empty()) g_sink += v.front() + v[v.size() / 2] + v.back();
}

double nowMs() {
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return static_cast<double>(t.QuadPart) * 1000.0 / static_cast<double>(freq.QuadPart);
}

unsigned pinThreadToCore() {
    DWORD_PTR processMask = 0, systemMask = 0;
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) return 0;
    unsigned chosen = 2;
    if (!(processMask & (static_cast<DWORD_PTR>(1) << chosen))) {
        chosen = 0;
        while (chosen < sizeof(DWORD_PTR) * 8 &&
               !(processMask & (static_cast<DWORD_PTR>(1) << chosen))) {
            ++chosen;
        }
    }
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << chosen);
    return chosen;
}

void prepareMatrixData(vector<double>& matrix, vector<double>& vec, int n) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = 0.25 + static_cast<double>((i * 17 + 13) % 251) / 251.0;
    }
    for (int i = 0; i < n; ++i) {
        vec[i] = 0.5 + static_cast<double>((i * 29 + 7) % 127) / 127.0;
    }
}

void prepareArrayData(vector<double>& arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = 1.0 + static_cast<double>((i * 19 + 5) % 257) / 257.0;
    }
}

template <typename Fn>
double measureLoopsMs(const Fn& fn, int loops) {
    const double start = nowMs();
    for (int i = 0; i < loops; ++i) fn();
    return nowMs() - start;
}

template <typename Fn>
int calibrateLoops(const Fn& fn, double targetMs, int maxLoops) {
    int loops = 1;
    while (true) {
        const double elapsed = measureLoopsMs(fn, loops);
        if (elapsed >= targetMs || loops >= maxLoops) return loops;
        if (elapsed < targetMs / 16.0 && loops <= maxLoops / 8) loops *= 8;
        else if (elapsed < targetMs / 4.0 && loops <= maxLoops / 4) loops *= 4;
        else if (loops <= maxLoops / 2) loops *= 2;
        else return maxLoops;
    }
}

template <typename Fn>
double medianPerCallMs(const Fn& fn, int loops, int rounds) {
    vector<double> samples;
    samples.reserve(rounds);
    for (int i = 0; i < rounds; ++i) {
        samples.push_back(measureLoopsMs(fn, loops) / static_cast<double>(loops));
    }
    sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

NOINLINE void matrixNaive(const vector<double>& matrix, const vector<double>& vec, vector<double>& result, int n) {
    fill(result.begin(), result.end(), 0.0);
    for (int col = 0; col < n; ++col) {
        double sum = 0.0;
        for (int row = 0; row < n; ++row) {
            sum += matrix[static_cast<size_t>(row) * n + col] * vec[row];
        }
        result[col] = sum;
    }
}

NOINLINE void matrixCacheOpt(const vector<double>& matrix, const vector<double>& vec, vector<double>& result, int n) {
    fill(result.begin(), result.end(), 0.0);
    for (int row = 0; row < n; ++row) {
        const double factor = vec[row];
        const size_t base = static_cast<size_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            result[col] += matrix[base + col] * factor;
        }
    }
}

NOINLINE void matrixCacheUnroll4(const vector<double>& matrix, const vector<double>& vec, vector<double>& result, int n) {
    fill(result.begin(), result.end(), 0.0);
    for (int row = 0; row < n; ++row) {
        const double factor = vec[row];
        const size_t base = static_cast<size_t>(row) * n;
        int col = 0;
        for (; col + 3 < n; col += 4) {
            result[col + 0] += matrix[base + col + 0] * factor;
            result[col + 1] += matrix[base + col + 1] * factor;
            result[col + 2] += matrix[base + col + 2] * factor;
            result[col + 3] += matrix[base + col + 3] * factor;
        }
        for (; col < n; ++col) {
            result[col] += matrix[base + col] * factor;
        }
    }
}

NOINLINE void transposeMatrix(const vector<double>& matrix, vector<double>& transposed, int n) {
    for (int row = 0; row < n; ++row) {
        const size_t base = static_cast<size_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            transposed[static_cast<size_t>(col) * n + row] = matrix[base + col];
        }
    }
}

NOINLINE void matrixTranspose4Acc(const vector<double>& transposed, const vector<double>& vec, vector<double>& result, int n) {
    for (int col = 0; col < n; ++col) {
        const size_t base = static_cast<size_t>(col) * n;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = 0;
        for (; i + 3 < n; i += 4) {
            s0 += transposed[base + i + 0] * vec[i + 0];
            s1 += transposed[base + i + 1] * vec[i + 1];
            s2 += transposed[base + i + 2] * vec[i + 2];
            s3 += transposed[base + i + 3] * vec[i + 3];
        }
        for (; i < n; ++i) {
            s0 += transposed[base + i] * vec[i];
        }
        result[col] = (s0 + s1) + (s2 + s3);
    }
}

NOINLINE double sumNaive(const vector<double>& arr) {
    double s = 0.0;
    for (double v : arr) s += v;
    return s;
}

NOINLINE double sumRecursiveRange(const double* data, size_t left, size_t right) {
    if (left == right) return data[left];
    const size_t mid = left + (right - left) / 2;
    return sumRecursiveRange(data, left, mid) + sumRecursiveRange(data, mid + 1, right);
}

NOINLINE double sumRecursive(const vector<double>& arr) {
    if (arr.empty()) return 0.0;
    return sumRecursiveRange(arr.data(), 0, arr.size() - 1);
}

NOINLINE double sum2Way(const vector<double>& arr) {
    double s0 = 0.0, s1 = 0.0;
    size_t i = 0;
    for (; i + 1 < arr.size(); i += 2) {
        s0 += arr[i + 0];
        s1 += arr[i + 1];
    }
    for (; i < arr.size(); ++i) s0 += arr[i];
    return s0 + s1;
}

NOINLINE double sumUnroll4(const vector<double>& arr) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 3 < arr.size(); i += 4) {
        s0 += arr[i + 0];
        s1 += arr[i + 1];
        s2 += arr[i + 2];
        s3 += arr[i + 3];
    }
    for (; i < arr.size(); ++i) s0 += arr[i];
    return (s0 + s1) + (s2 + s3);
}

NOINLINE double sumUnroll8(const vector<double>& arr) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
    size_t i = 0;
    for (; i + 7 < arr.size(); i += 8) {
        s0 += arr[i + 0];
        s1 += arr[i + 1];
        s2 += arr[i + 2];
        s3 += arr[i + 3];
        s4 += arr[i + 4];
        s5 += arr[i + 5];
        s6 += arr[i + 6];
        s7 += arr[i + 7];
    }
    for (; i < arr.size(); ++i) s0 += arr[i];
    return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
}

struct MatrixRow {
    int n;
    string algorithm;
    int loops;
    double msPerCall;
    double speedup;
    double gflops;
};

struct LayoutRow {
    int n;
    double transposeMs;
    double kernelMs;
    double cacheOptMs;
    double speedupVsNaive;
    double speedupVsCacheOpt;
    double breakEvenReuse;
};

struct SumRow {
    int n;
    string algorithm;
    int loops;
    double usPerCall;
    double nsPerElem;
    double speedup;
    double gadds;
};

void writeMatrixCsv(const fs::path& path, const vector<MatrixRow>& rows) {
    ofstream out(path);
    out << "n,algorithm,loops,ms_per_call,speedup,gflops\n";
    out << fixed << setprecision(9);
    for (const auto& row : rows) {
        out << row.n << ',' << row.algorithm << ',' << row.loops << ','
            << row.msPerCall << ',' << row.speedup << ',' << row.gflops << '\n';
    }
}

void writeLayoutCsv(const fs::path& path, const vector<LayoutRow>& rows) {
    ofstream out(path);
    out << "n,transpose_ms,kernel_ms,cache_opt_ms,speedup_vs_naive,speedup_vs_cache_opt,break_even_reuse\n";
    out << fixed << setprecision(9);
    for (const auto& row : rows) {
        out << row.n << ',' << row.transposeMs << ',' << row.kernelMs << ','
            << row.cacheOptMs << ',' << row.speedupVsNaive << ','
            << row.speedupVsCacheOpt << ',' << row.breakEvenReuse << '\n';
    }
}

void writeSumCsv(const fs::path& path, const vector<SumRow>& rows) {
    ofstream out(path);
    out << "n,algorithm,loops,us_per_call,ns_per_elem,speedup,gadds\n";
    out << fixed << setprecision(9);
    for (const auto& row : rows) {
        out << row.n << ',' << row.algorithm << ',' << row.loops << ','
            << row.usPerCall << ',' << row.nsPerElem << ',' << row.speedup << ',' << row.gadds << '\n';
    }
}

vector<MatrixRow> runMatrixAdvanced() {
    const vector<int> sizes = {256, 512, 1024, 2048, 3072};
    vector<MatrixRow> rows;
    cout << "[Advanced] Matrix benchmark\n";
    for (int n : sizes) {
        vector<double> matrix(static_cast<size_t>(n) * n), vec(n), out(n), transposed(static_cast<size_t>(n) * n);
        prepareMatrixData(matrix, vec, n);
        transposeMatrix(matrix, transposed, n);

        struct Candidate {
            string name;
            function<void()> fn;
        };

        vector<Candidate> candidates = {
            {"naive", [&] { matrixNaive(matrix, vec, out, n); consumeVector(out); }},
            {"cache_opt", [&] { matrixCacheOpt(matrix, vec, out, n); consumeVector(out); }},
            {"cache_unroll4", [&] { matrixCacheUnroll4(matrix, vec, out, n); consumeVector(out); }},
            {"transpose_4acc", [&] { matrixTranspose4Acc(transposed, vec, out, n); consumeVector(out); }},
        };

        vector<MatrixRow> local;
        for (const auto& c : candidates) {
            c.fn();
            int loops = calibrateLoops(c.fn, 180.0, 256);
            double msPerCall = medianPerCallMs(c.fn, loops, 5);
            double gflops = (2.0 * n * n) / (msPerCall * 1.0e6);
            local.push_back({n, c.name, loops, msPerCall, 0.0, gflops});
        }

        const double baseline = local.front().msPerCall;
        for (auto& row : local) {
            row.speedup = baseline / row.msPerCall;
            rows.push_back(row);
            cout << "N=" << setw(5) << n
                 << "  " << setw(14) << left << row.algorithm << right
                 << "  ms/call=" << fixed << setprecision(4) << setw(9) << row.msPerCall
                 << "  speedup=" << fixed << setprecision(2) << row.speedup << "x\n";
        }
    }
    return rows;
}

vector<LayoutRow> runMatrixLayoutExperiment() {
    const vector<int> sizes = {256, 512, 1024, 2048, 3072};
    vector<LayoutRow> rows;
    cout << "\n[Advanced] Matrix layout-transform benchmark\n";

    for (int n : sizes) {
        vector<double> matrix(static_cast<size_t>(n) * n), vec(n), out(n), transposed(static_cast<size_t>(n) * n);
        prepareMatrixData(matrix, vec, n);

        const double t0 = nowMs();
        transposeMatrix(matrix, transposed, n);
        const double transposeMs = nowMs() - t0;

        auto cacheFn = [&] {
            matrixCacheOpt(matrix, vec, out, n);
            consumeVector(out);
        };
        auto transFn = [&] {
            matrixTranspose4Acc(transposed, vec, out, n);
            consumeVector(out);
        };
        auto naiveFn = [&] {
            matrixNaive(matrix, vec, out, n);
            consumeVector(out);
        };

        const int cacheLoops = calibrateLoops(cacheFn, 180.0, 256);
        const int transLoops = calibrateLoops(transFn, 180.0, 256);
        const int naiveLoops = calibrateLoops(naiveFn, 180.0, 256);
        const double cacheMs = medianPerCallMs(cacheFn, cacheLoops, 5);
        const double kernelMs = medianPerCallMs(transFn, transLoops, 5);
        const double naiveMs = medianPerCallMs(naiveFn, naiveLoops, 5);

        double breakEven = -1.0;
        if (cacheMs > kernelMs) {
            breakEven = transposeMs / (cacheMs - kernelMs);
        }

        rows.push_back({
            n,
            transposeMs,
            kernelMs,
            cacheMs,
            naiveMs / kernelMs,
            cacheMs / kernelMs,
            breakEven
        });

        cout << "N=" << setw(5) << n
             << "  transpose_ms=" << fixed << setprecision(3) << setw(8) << transposeMs
             << "  kernel_ms=" << setw(8) << kernelMs
             << "  break_even_reuse=" << setw(8) << setprecision(2) << breakEven << '\n';
    }
    return rows;
}

vector<SumRow> runSumAdvanced() {
    const vector<int> sizes = {1 << 15, 1 << 18, 1 << 21, 1 << 24};
    vector<SumRow> rows;
    cout << "\n[Advanced] Summation benchmark\n";
    for (int n : sizes) {
        vector<double> arr(n);
        prepareArrayData(arr);

        struct Candidate {
            string name;
            function<void()> fn;
        };

        vector<Candidate> candidates = {
            {"naive", [&] { consumeScalar(sumNaive(arr)); }},
            {"superscalar_2way", [&] { consumeScalar(sum2Way(arr)); }},
            {"unroll4", [&] { consumeScalar(sumUnroll4(arr)); }},
            {"unroll8", [&] { consumeScalar(sumUnroll8(arr)); }},
        };

        vector<SumRow> local;
        for (const auto& c : candidates) {
            c.fn();
            int loops = calibrateLoops(c.fn, 180.0, 8192);
            double msPerCall = medianPerCallMs(c.fn, loops, 5);
            double usPerCall = msPerCall * 1000.0;
            double nsPerElem = msPerCall * 1.0e6 / static_cast<double>(n);
            double gadds = static_cast<double>(n) / (msPerCall * 1.0e6);
            local.push_back({n, c.name, loops, usPerCall, nsPerElem, 0.0, gadds});
        }

        const double baseline = local.front().usPerCall;
        for (auto& row : local) {
            row.speedup = baseline / row.usPerCall;
            rows.push_back(row);
            cout << "N=" << setw(8) << n
                 << "  " << setw(18) << left << row.algorithm << right
                 << "  us/call=" << fixed << setprecision(3) << setw(10) << row.usPerCall
                 << "  speedup=" << fixed << setprecision(2) << row.speedup << "x\n";
        }
    }
    return rows;
}

bool equalsIgnoreCase(string a, string b) {
    transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return a == b;
}

int runSumProfileKernel(const string& variant, int n, int seconds) {
    vector<double> arr(n);
    prepareArrayData(arr);

    function<void()> kernel;
    if (equalsIgnoreCase(variant, "naive")) {
        kernel = [&] { consumeScalar(sumNaive(arr)); };
    } else if (equalsIgnoreCase(variant, "superscalar_2way")) {
        kernel = [&] { consumeScalar(sum2Way(arr)); };
    } else if (equalsIgnoreCase(variant, "recursive")) {
        kernel = [&] { consumeScalar(sumRecursive(arr)); };
    } else if (equalsIgnoreCase(variant, "unroll4")) {
        kernel = [&] { consumeScalar(sumUnroll4(arr)); };
    } else if (equalsIgnoreCase(variant, "unroll8")) {
        kernel = [&] { consumeScalar(sumUnroll8(arr)); };
    } else {
        cerr << "Unsupported sum profile variant: " << variant << '\n';
        return 2;
    }

    const double deadline = nowMs() + static_cast<double>(seconds) * 1000.0;
    size_t iterations = 0;
    while (nowMs() < deadline) {
        kernel();
        ++iterations;
    }

    cout << "PROFILE_MODE=sum\n";
    cout << "VARIANT=" << variant << '\n';
    cout << "N=" << n << '\n';
    cout << "SECONDS=" << seconds << '\n';
    cout << "ITERATIONS=" << iterations << '\n';
    cout << "SINK=" << setprecision(12) << g_sink << '\n';
    return 0;
}

int main(int argc, char** argv) {
    pinThreadToCore();
    if (argc >= 2 && string(argv[1]) == "profile-sum") {
        const string variant = (argc >= 3) ? argv[2] : "naive";
        const int n = (argc >= 4) ? stoi(argv[3]) : (1 << 21);
        const int seconds = (argc >= 5) ? stoi(argv[4]) : 8;
        return runSumProfileKernel(variant, n, seconds);
    }

    const fs::path outDir = (argc >= 2) ? fs::path(argv[1]) : fs::current_path() / "advanced_results";
    fs::create_directories(outDir);

    const auto matrixRows = runMatrixAdvanced();
    const auto layoutRows = runMatrixLayoutExperiment();
    const auto sumRows = runSumAdvanced();
    writeMatrixCsv(outDir / "advanced_matrix.csv", matrixRows);
    writeLayoutCsv(outDir / "advanced_layout.csv", layoutRows);
    writeSumCsv(outDir / "advanced_sum.csv", sumRows);

    cout << "\nAdvanced sink = " << setprecision(12) << g_sink << '\n';
    cout << "Output dir: " << outDir.string() << '\n';
    return 0;
}
