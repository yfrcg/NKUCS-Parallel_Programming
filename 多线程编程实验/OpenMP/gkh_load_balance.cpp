#include "gkh.h"

#include <array>
#include <iostream>

int main()
{
    constexpr std::array<int, 8> block_lengths = {512, 448, 384, 320, 256, 192, 160, 128};
    constexpr int separator_count = 2;
    constexpr int n = 2414;

    Matrix U(n, n, 0.0);
    Matrix B(n, n, 0.0);
    Matrix V(n, n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        U.at(i, i) = 1.0;
        V.at(i, i) = 1.0;
        B.at(i, i) = 2.0 + 0.001 * i;
    }

    int begin = 0;
    for (const int len : block_lengths)
    {
        for (int i = begin; i < begin + len - 1; ++i)
        {
            B.at(i, i + 1) = 0.1;
        }
        begin += len + separator_count;
    }

    std::cout << "GKH OpenMP load input: n=" << n << ", blocks=";
    for (int i = 0; i < static_cast<int>(block_lengths.size()); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << block_lengths[i];
    }
    std::cout << "\n";

    gkh_svd_from_bidiagonal(U, B, V, 1, 1e-12);
    return 0;
}
