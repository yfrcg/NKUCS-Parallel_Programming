#pragma once

#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

class Matrix
{
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(int rows, int cols, double init = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    double &at(int r, int c) { return data_[r * cols_ + c]; }
    double at(int r, int c) const { return data_[r * cols_ + c]; }

    double *row_ptr(int r) { return data_.data() + r * cols_; }
    const double *row_ptr(int r) const { return data_.data() + r * cols_; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return static_cast<int>(data_.size()); }
    double *data() { return data_.data(); }
    const double *data() const { return data_.data(); }

    Matrix operator+(const Matrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("Matrix addition: dimension mismatch");
        }

        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i)
        {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Matrix operator*(const Matrix &other) const
    {
        if (cols_ != other.rows_)
        {
            throw std::invalid_argument("Matrix multiplication: dimension mismatch");
        }

        Matrix result(rows_, other.cols_, 0.0);
        for (int i = 0; i < rows_; ++i)
        {
            for (int k = 0; k < cols_; ++k)
            {
                for (int j = 0; j < other.cols_; ++j)
                {
                    result.at(i, j) += at(i, k) * other.at(k, j);
                }
            }
        }
        return result;
    }

    static Matrix random(int rows, int cols, double lo = 0.0, double hi = 1.0,
                         long long seed = -1)
    {
        std::mt19937_64 rng;
        if (seed < 0)
        {
            rng.seed(std::random_device{}());
        }
        else
        {
            rng.seed(static_cast<std::mt19937_64::result_type>(seed));
        }

        std::uniform_real_distribution<double> dist(lo, hi);
        Matrix result(rows, cols);
        for (double &v : result.data_)
        {
            v = dist(rng);
        }
        return result;
    }

    void print(std::ostream &os = std::cout) const
    {
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                os << std::setw(12) << std::fixed << std::setprecision(6) << at(i, j);
            }
            os << '\n';
        }
    }

private:
    int rows_, cols_;
    std::vector<double> data_;
};
