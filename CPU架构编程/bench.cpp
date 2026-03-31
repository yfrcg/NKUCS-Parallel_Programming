#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
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

struct CacheInfo {
    size_t l1d = 0;
    size_t l2 = 0;
    size_t l3 = 0;
};

struct MatrixTimingRow {
    int n = 0;
    size_t workingSetBytes = 0;
    string workingSetClass;
    string algorithm;
    int loops = 0;
    double msPerCall = 0.0;
    double gflops = 0.0;
    double speedup = 0.0;
    double maxDiff = 0.0;
};

struct SumTimingRow {
    int n = 0;
    size_t workingSetBytes = 0;
    string workingSetClass;
    string algorithm;
    int loops = 0;
    double usPerCall = 0.0;
    double nsPerElem = 0.0;
    double gadds = 0.0;
    double speedup = 0.0;
    double absErr = 0.0;
};

inline void consumeScalar(double x) {
    g_sink += x;
}

inline void consumeVector(const vector<double>& v) {
    if (!v.empty()) {
        g_sink += v.front() + v[v.size() / 2] + v.back();
    }
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

string readCpuName() {
    HKEY key = nullptr;
    if (RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0,
            KEY_QUERY_VALUE,
            &key) != ERROR_SUCCESS) {
        return "Unknown CPU";
    }

    char buffer[256] = {};
    DWORD size = sizeof(buffer);
    const LSTATUS status = RegQueryValueExA(
        key,
        "ProcessorNameString",
        nullptr,
        nullptr,
        reinterpret_cast<LPBYTE>(buffer),
        &size);
    RegCloseKey(key);
    if (status != ERROR_SUCCESS) {
        return "Unknown CPU";
    }
    return string(buffer);
}

CacheInfo queryCacheInfo() {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes);
    vector<unsigned char> buffer(bytes);

    CacheInfo info;
    if (!GetLogicalProcessorInformationEx(
            RelationCache,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &bytes)) {
        return info;
    }

    size_t offset = 0;
    while (offset < bytes) {
        auto* entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        const CACHE_RELATIONSHIP& cache = entry->Cache;
        if (cache.Type == CacheData || cache.Type == CacheUnified) {
            if (cache.Level == 1) info.l1d = max(info.l1d, static_cast<size_t>(cache.CacheSize));
            if (cache.Level == 2) info.l2 = max(info.l2, static_cast<size_t>(cache.CacheSize));
            if (cache.Level == 3) info.l3 = max(info.l3, static_cast<size_t>(cache.CacheSize));
        }
        offset += entry->Size;
    }
    return info;
}

unsigned pinThreadToCore() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    DWORD_PTR processMask = 0;
    DWORD_PTR systemMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        return 0;
    }

    unsigned chosen = 0;
    for (unsigned bit = 2; bit < sizeof(DWORD_PTR) * 8; ++bit) {
        if (processMask & (static_cast<DWORD_PTR>(1) << bit)) {
            chosen = bit;
            break;
        }
    }
    if (chosen == 0) {
        for (unsigned bit = 0; bit < sizeof(DWORD_PTR) * 8; ++bit) {
            if (processMask & (static_cast<DWORD_PTR>(1) << bit)) {
                chosen = bit;
                break;
            }
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

string formatBytes(size_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    ostringstream oss;
    oss << fixed << setprecision(value >= 100.0 ? 1 : 2) << value << ' ' << units[unit];
    return oss.str();
}

string classifyWorkingSet(size_t bytes, const CacheInfo& cache) {
    if (cache.l1d != 0 && bytes <= cache.l1d) return "<=L1";
    if (cache.l2 != 0 && bytes <= cache.l2) return "L1<LWS<=L2";
    if (cache.l3 != 0 && bytes <= cache.l3) return "L2<LWS<=L3";
    if (cache.l3 != 0) return ">L3";
    return "unknown";
}

void printRule(char ch = '=') {
    cout << string(96, ch) << '\n';
}

double maxAbsDiff(const vector<double>& a, const vector<double>& b) {
    double diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = max(diff, fabs(a[i] - b[i]));
    }
    return diff;
}

template <typename Fn>
double measureLoopsMs(const Fn& fn, int loops) {
    const double begin = nowMs();
    for (int i = 0; i < loops; ++i) {
        fn();
    }
    return nowMs() - begin;
}

template <typename Fn>
int calibrateLoops(const Fn& fn, double targetMs, int maxLoops) {
    int loops = 1;
    while (true) {
        const double elapsed = measureLoopsMs(fn, loops);
        if (elapsed >= targetMs || loops >= maxLoops) {
            return loops;
        }
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

NOINLINE void matrixColDotNaive(
    const vector<double>& matrix,
    const vector<double>& vec,
    vector<double>& result,
    int n) {
    fill(result.begin(), result.end(), 0.0);
    for (int col = 0; col < n; ++col) {
        double sum = 0.0;
        for (int row = 0; row < n; ++row) {
            sum += matrix[static_cast<size_t>(row) * n + col] * vec[row];
        }
        result[col] = sum;
    }
}

NOINLINE void matrixColDotCacheOptimized(
    const vector<double>& matrix,
    const vector<double>& vec,
    vector<double>& result,
    int n) {
    fill(result.begin(), result.end(), 0.0);
    for (int row = 0; row < n; ++row) {
        const double factor = vec[row];
        const size_t base = static_cast<size_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            result[col] += matrix[base + col] * factor;
        }
    }
}

NOINLINE double sumNaive(const vector<double>& arr) {
    double sum = 0.0;
    for (double value : arr) {
        sum += value;
    }
    return sum;
}

NOINLINE double sumSuperscalar2Way(const vector<double>& arr) {
    double s0 = 0.0;
    double s1 = 0.0;
    size_t i = 0;
    for (; i + 1 < arr.size(); i += 2) {
        s0 += arr[i];
        s1 += arr[i + 1];
    }
    if (i < arr.size()) {
        s0 += arr[i];
    }
    return s0 + s1;
}

NOINLINE double sumRecursiveImpl(const double* data, int left, int right) {
    if (left == right) return data[left];
    const int mid = left + (right - left) / 2;
    return sumRecursiveImpl(data, left, mid) + sumRecursiveImpl(data, mid + 1, right);
}

NOINLINE double sumRecursive(const vector<double>& arr) {
    if (arr.empty()) return 0.0;
    return sumRecursiveImpl(arr.data(), 0, static_cast<int>(arr.size()) - 1);
}

void writeMatrixCsv(const fs::path& path, const vector<MatrixTimingRow>& rows) {
    ofstream out(path);
    out << "n,working_set_bytes,working_set_class,algorithm,loops,ms_per_call,gflops,speedup,max_diff\n";
    out << fixed << setprecision(9);
    for (const auto& row : rows) {
        out << row.n << ','
            << row.workingSetBytes << ','
            << row.workingSetClass << ','
            << row.algorithm << ','
            << row.loops << ','
            << row.msPerCall << ','
            << row.gflops << ','
            << row.speedup << ','
            << row.maxDiff << '\n';
    }
}

void writeSumCsv(const fs::path& path, const vector<SumTimingRow>& rows) {
    ofstream out(path);
    out << "n,working_set_bytes,working_set_class,algorithm,loops,us_per_call,ns_per_elem,gadds,speedup,abs_err\n";
    out << fixed << setprecision(9);
    for (const auto& row : rows) {
        out << row.n << ','
            << row.workingSetBytes << ','
            << row.workingSetClass << ','
            << row.algorithm << ','
            << row.loops << ','
            << row.usPerCall << ','
            << row.nsPerElem << ','
            << row.gadds << ','
            << row.speedup << ','
            << row.absErr << '\n';
    }
}

void printPlatformInfo(const CacheInfo& cache, unsigned pinnedCore) {
    printRule();
    cout << "CPU Architecture Programming Benchmark" << '\n';
    printRule();
    cout << left << setw(30) << "CPU" << ": " << readCpuName() << '\n';
    cout << left << setw(30) << "Logical processors" << ": "
         << static_cast<unsigned>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)) << '\n';
    cout << left << setw(30) << "Pinned logical CPU" << ": " << pinnedCore << '\n';
    cout << left << setw(30) << "Observed L1 data cache" << ": " << formatBytes(cache.l1d) << '\n';
    cout << left << setw(30) << "Observed L2 cache" << ": " << formatBytes(cache.l2) << '\n';
    cout << left << setw(30) << "Observed L3 cache" << ": " << formatBytes(cache.l3) << '\n';
    cout << left << setw(30) << "Compiler expectation" << ": g++ -O3 -march=native -std=c++17" << '\n';
    printRule();
}

vector<MatrixTimingRow> runMatrixTimingExperiment(const CacheInfo& cache) {
    const vector<int> sizes = {256, 512, 1024, 2048, 3072};
    vector<MatrixTimingRow> rows;

    cout << "\n[Experiment 1] Matrix-column dot product\n";
    cout << "Compare naive column-wise traversal against cache-friendly row-wise traversal.\n";

    for (int n : sizes) {
        vector<double> matrix(static_cast<size_t>(n) * n);
        vector<double> vec(n);
        vector<double> ref(n);
        vector<double> out(n);
        prepareMatrixData(matrix, vec, n);

        matrixColDotNaive(matrix, vec, ref, n);
        matrixColDotCacheOptimized(matrix, vec, out, n);
        const double optDiff = maxAbsDiff(ref, out);

        const size_t ws = matrix.size() * sizeof(double)
                        + vec.size() * sizeof(double)
                        + ref.size() * sizeof(double);
        const string wsClass = classifyWorkingSet(ws, cache);

        struct Candidate {
            string name;
            function<void()> fn;
            double diff;
        };

        vector<Candidate> cases = {
            {"naive", [&] { matrixColDotNaive(matrix, vec, out, n); consumeVector(out); }, 0.0},
            {"cache_opt", [&] { matrixColDotCacheOptimized(matrix, vec, out, n); consumeVector(out); }, optDiff},
        };

        vector<MatrixTimingRow> local;
        for (const auto& item : cases) {
            item.fn();
            const int loops = calibrateLoops(item.fn, 180.0, 256);
            const double msPerCall = medianPerCallMs(item.fn, loops, 5);
            const double gflops = (2.0 * static_cast<double>(n) * static_cast<double>(n)) / (msPerCall * 1.0e6);
            local.push_back({n, ws, wsClass, item.name, loops, msPerCall, gflops, 0.0, item.diff});
        }

        const double baseline = local.front().msPerCall;
        cout << '\n';
        printRule('-');
        cout << "N = " << n
             << ", working set = " << formatBytes(ws)
             << ", cache relation = " << wsClass << '\n';
        cout << left << setw(18) << "Algorithm"
             << right << setw(10) << "Loops"
             << setw(14) << "ms/call"
             << setw(14) << "GFLOP/s"
             << setw(12) << "Speedup"
             << setw(16) << "MaxDiff" << '\n';
        printRule('-');

        for (auto& row : local) {
            row.speedup = baseline / row.msPerCall;
            rows.push_back(row);
            cout << left << setw(18) << row.algorithm
                 << right << setw(10) << row.loops
                 << setw(14) << fixed << setprecision(3) << row.msPerCall
                 << setw(14) << fixed << setprecision(3) << row.gflops
                 << setw(12) << fixed << setprecision(2) << row.speedup << "x"
                 << setw(16) << scientific << setprecision(3) << row.maxDiff
                 << defaultfloat << '\n';
        }
    }

    return rows;
}

vector<SumTimingRow> runSumTimingExperiment(const CacheInfo& cache) {
    const vector<int> sizes = {1 << 15, 1 << 18, 1 << 21, 1 << 24};
    vector<SumTimingRow> rows;

    cout << "\n[Experiment 2] Summation of n numbers\n";
    cout << "Compare ordinary chained accumulation with two-way superscalar and recursive reduction.\n";

    for (int n : sizes) {
        vector<double> arr(n);
        prepareArrayData(arr);

        const double ref = sumNaive(arr);
        const double sum2 = sumSuperscalar2Way(arr);
        const double sumRec = sumRecursive(arr);

        const size_t ws = arr.size() * sizeof(double);
        const string wsClass = classifyWorkingSet(ws, cache);

        struct Candidate {
            string name;
            function<void()> fn;
            double error;
        };

        vector<Candidate> cases = {
            {"naive", [&] { consumeScalar(sumNaive(arr)); }, 0.0},
            {"superscalar_2way", [&] { consumeScalar(sumSuperscalar2Way(arr)); }, fabs(sum2 - ref)},
            {"recursive", [&] { consumeScalar(sumRecursive(arr)); }, fabs(sumRec - ref)},
        };

        vector<SumTimingRow> local;
        for (const auto& item : cases) {
            item.fn();
            const int loops = calibrateLoops(item.fn, 180.0, 8192);
            const double msPerCall = medianPerCallMs(item.fn, loops, 5);
            const double usPerCall = msPerCall * 1000.0;
            const double nsPerElem = msPerCall * 1.0e6 / static_cast<double>(n);
            const double gadds = static_cast<double>(n) / (msPerCall * 1.0e6);
            local.push_back({n, ws, wsClass, item.name, loops, usPerCall, nsPerElem, gadds, 0.0, item.error});
        }

        const double baseline = local.front().usPerCall;
        cout << '\n';
        printRule('-');
        cout << "N = " << n
             << ", working set = " << formatBytes(ws)
             << ", cache relation = " << wsClass << '\n';
        cout << left << setw(22) << "Algorithm"
             << right << setw(10) << "Loops"
             << setw(14) << "us/call"
             << setw(14) << "ns/elem"
             << setw(14) << "GAdd/s"
             << setw(12) << "Speedup"
             << setw(16) << "AbsErr" << '\n';
        printRule('-');

        for (auto& row : local) {
            row.speedup = baseline / row.usPerCall;
            rows.push_back(row);
            cout << left << setw(22) << row.algorithm
                 << right << setw(10) << row.loops
                 << setw(14) << fixed << setprecision(3) << row.usPerCall
                 << setw(14) << fixed << setprecision(3) << row.nsPerElem
                 << setw(14) << fixed << setprecision(3) << row.gadds
                 << setw(12) << fixed << setprecision(2) << row.speedup << "x"
                 << setw(16) << scientific << setprecision(3) << row.absErr
                 << defaultfloat << '\n';
        }
    }

    return rows;
}

bool equalsIgnoreCase(string a, string b) {
    transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return a == b;
}

int runMatrixProfileKernel(const string& variant, int n, int seconds) {
    vector<double> matrix(static_cast<size_t>(n) * n);
    vector<double> vec(n);
    vector<double> out(n);
    prepareMatrixData(matrix, vec, n);

    function<void()> kernel;
    if (equalsIgnoreCase(variant, "naive")) {
        kernel = [&] {
            matrixColDotNaive(matrix, vec, out, n);
            consumeVector(out);
        };
    } else if (equalsIgnoreCase(variant, "cache") || equalsIgnoreCase(variant, "cache_opt")) {
        kernel = [&] {
            matrixColDotCacheOptimized(matrix, vec, out, n);
            consumeVector(out);
        };
    } else {
        cerr << "Unsupported matrix profile variant: " << variant << '\n';
        return 2;
    }

    const double deadline = nowMs() + static_cast<double>(seconds) * 1000.0;
    size_t iterations = 0;
    while (nowMs() < deadline) {
        kernel();
        ++iterations;
    }

    cout << "PROFILE_MODE=matrix\n";
    cout << "VARIANT=" << variant << '\n';
    cout << "N=" << n << '\n';
    cout << "SECONDS=" << seconds << '\n';
    cout << "ITERATIONS=" << iterations << '\n';
    cout << "SINK=" << setprecision(12) << g_sink << '\n';
    return 0;
}

void printSummary(const vector<MatrixTimingRow>& matrixRows, const vector<SumTimingRow>& sumRows) {
    auto bestMatrix = max_element(
        matrixRows.begin(),
        matrixRows.end(),
        [](const MatrixTimingRow& a, const MatrixTimingRow& b) { return a.speedup < b.speedup; });
    auto bestSum = max_element(
        sumRows.begin(),
        sumRows.end(),
        [](const SumTimingRow& a, const SumTimingRow& b) { return a.speedup < b.speedup; });

    cout << "\n";
    printRule();
    cout << "Summary\n";
    printRule();
    if (bestMatrix != matrixRows.end()) {
        cout << "Best matrix speedup : " << fixed << setprecision(2)
             << bestMatrix->speedup << "x at N=" << bestMatrix->n
             << " by " << bestMatrix->algorithm << '\n';
    }
    if (bestSum != sumRows.end()) {
        cout << "Best sum speedup    : " << fixed << setprecision(2)
             << bestSum->speedup << "x at N=" << bestSum->n
             << " by " << bestSum->algorithm << '\n';
    }
    cout << "Final sink value    : " << setprecision(12) << g_sink << '\n';
    printRule();
}

void printUsage(const char* exe) {
    cout << "Usage:\n";
    cout << "  " << exe << " run-all [output_dir]\n";
    cout << "  " << exe << " profile-matrix <naive|cache> [n] [seconds]\n";
    cout << "\n";
    cout << "Examples:\n";
    cout << "  " << exe << " run-all .\n";
    cout << "  " << exe << " profile-matrix naive 1024 8\n";
    cout << "  " << exe << " profile-matrix cache 1024 8\n";
}

int main(int argc, char** argv) {
    try {
        const unsigned pinnedCore = pinThreadToCore();
        const CacheInfo cache = queryCacheInfo();

        string mode = (argc >= 2) ? argv[1] : "run-all";
        if (mode == "profile-matrix") {
            const string variant = (argc >= 3) ? argv[2] : "naive";
            const int n = (argc >= 4) ? stoi(argv[3]) : 1024;
            const int seconds = (argc >= 5) ? stoi(argv[4]) : 8;
            return runMatrixProfileKernel(variant, n, seconds);
        }

        if (mode != "run-all") {
            printUsage(argv[0]);
            return 1;
        }

        const fs::path outputDir = (argc >= 3) ? fs::path(argv[2]) : fs::current_path();
        fs::create_directories(outputDir);

        printPlatformInfo(cache, pinnedCore);
        const vector<MatrixTimingRow> matrixRows = runMatrixTimingExperiment(cache);
        const vector<SumTimingRow> sumRows = runSumTimingExperiment(cache);
        printSummary(matrixRows, sumRows);

        writeMatrixCsv(outputDir / "matrix_timing.csv", matrixRows);
        writeSumCsv(outputDir / "sum_timing.csv", sumRows);
        return 0;
    } catch (const exception& ex) {
        cerr << "Benchmark failed: " << ex.what() << '\n';
        return 1;
    } catch (...) {
        cerr << "Benchmark failed with unknown error.\n";
        return 1;
    }
}
