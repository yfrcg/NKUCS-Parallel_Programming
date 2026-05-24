#include "matrix.h"
#include "gkh.h"
#include "bidiagonalization.h"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    const long long base_seed = (argc >= 2) ? std::stoll(argv[1]) : 1024LL;

    using Clock = std::chrono::high_resolution_clock;

    Matrix A = Matrix::random(1000, 1000, -1.0, 1.0, base_seed + 4);
    Matrix U, V;

    const auto t_beg_bidiag = Clock::now();
    Matrix B = to_bidiagonal(A, U, V);
    const auto t_end_bidiag = Clock::now();

    const auto t_beg_gkh = Clock::now();
    const bool converged = gkh_svd_from_bidiagonal(U, B, V, 6000, 1e-12);
    const auto t_end_gkh = Clock::now();

    const double time_bidiag_ms =
        std::chrono::duration<double, std::milli>(t_end_bidiag - t_beg_bidiag).count();
    const double time_gkh_ms =
        std::chrono::duration<double, std::milli>(t_end_gkh - t_beg_gkh).count();

    std::cout << "=== PERF ONLY 1000x1000 ===\n";
    std::cout << "seed                      : " << base_seed << "\n";
    std::cout << "converged                 : " << (converged ? "yes" : "no") << "\n";
    std::cout << "time bidiagonalization(ms): " << time_bidiag_ms << "\n";
    std::cout << "time gkh iteration(ms)    : " << time_gkh_ms << "\n";

    return converged ? 0 : 1;
}