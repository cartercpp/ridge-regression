//
// Created by cartercpp on 8/21/26.
//

#ifndef RIDGEREGRESSION_MATRIX_H
#define RIDGEREGRESSION_MATRIX_H

#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <utility>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include "math_vector.h"

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class matrix
{
public:

    // CONSTRUCTORS

    matrix() = default;

    explicit matrix(std::size_t rows, std::size_t columns, ValueType value = 0)
        : m_data(rows, math_vector<ValueType>(columns, value)), m_rows{rows}, m_columns{columns}
    {}

    matrix(std::initializer_list<math_vector<ValueType>> data)
        : m_data{data}, m_rows{data.size()}, m_columns{(data.size() > 0) ? (*data.begin()).size() : 0}
    {
        for (const auto& vec : data)
            if (vec.size() != m_columns)
                throw std::invalid_argument{"Each row in a matrix must have the same # of columns"};
    }

    // METHODS

    [[nodiscard]] std::size_t rows() const noexcept
    {
        return m_rows;
    }

    [[nodiscard]] std::size_t columns() const noexcept
    {
        return m_columns;
    }

    static matrix identity(std::size_t size)
    {
        matrix<ValueType> output(size, size, 0);

        for (std::size_t i = 0; i < size; ++i)
            output[i][i] = 1;

        return output;
    }

    // Gauss-Jordan elimination with partial pivoting
    matrix inverse() const
    {
        if (m_rows != m_columns)
            throw std::invalid_argument{"Cannot invert a non-square matrix"};

        const std::size_t n = m_rows;
        matrix A = *this;                     // working copy
        matrix I(n, n, 0);                    // identity

        for (std::size_t i = 0; i < n; ++i)
            I[i][i] = static_cast<ValueType>(1);

        for (std::size_t col = 0; col < n; ++col)
        {
            // Find pivot (largest absolute value in column below diagonal)
            std::size_t pivot_row = col;
            ValueType max_abs = A[col][col] < ValueType{} ? -A[col][col] : A[col][col];

            for (std::size_t row = col + 1; row < n; ++row)
            {
                ValueType abs_val = A[row][col] < ValueType{} ? -A[row][col] : A[row][col];
                if (abs_val > max_abs)
                {
                    max_abs = abs_val;
                    pivot_row = row;
                }
            }

            if (max_abs == ValueType{})
                throw std::invalid_argument{"Matrix is singular and cannot be inverted"};

            // Swap rows
            if (pivot_row != col)
            {
                std::swap(A[col], A[pivot_row]);
                std::swap(I[col], I[pivot_row]);
            }

            // Scale pivot row so that pivot becomes 1
            const ValueType pivot = A[col][col];
            A[col] /= pivot;
            I[col] /= pivot;

            // Eliminate the current column in all other rows
            for (std::size_t row = 0; row < n; ++row)
            {
                if (row == col)
                    continue;

                const ValueType factor = A[row][col];
                A[row] -= A[col] * factor;
                I[row] -= I[col] * factor;
            }
        }

        return I;
    }

    math_vector<ValueType>& operator[](std::size_t row)
    {
        if (row >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[row];
    }

    const math_vector<ValueType>& operator[](std::size_t row) const
    {
        if (row >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[row];
    }

    math_vector<ValueType> get_column(std::size_t column) const
    {
        if (column >= m_columns)
            throw std::out_of_range{"Out of bounds"};

        math_vector<ValueType> output(m_rows, 0);
        for (std::size_t row = 0; row < m_rows; ++row)
            output[row] = m_data[row][column];

        return output;
    }

    matrix transpose() const
    {
        matrix<ValueType> output(m_columns, m_rows, 0);

        for (std::size_t column = 0; column < m_columns; ++column)
            output[column] = get_column(column);

        return output;
    }

    matrix& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](auto& rowRef){rowRef *= scalar;});
        return *this;
    }

    matrix& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](auto& rowRef){rowRef /= scalar;});
        return *this;
    }

    matrix& operator+=(const matrix& other)
    {
        if ((m_rows != other.rows()) || (m_columns != other.columns()))
            throw std::invalid_argument{"You cannot add 2 matrices with different dimensions"};

        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] += other[row];

        return *this;
    }

    matrix& operator-=(const matrix& other)
    {
        if ((m_rows != other.rows()) || (m_columns != other.columns()))
            throw std::invalid_argument{"You cannot add 2 matrices with different dimensions"};

        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] -= other[row];

        return *this;
    }

private:

    std::vector<math_vector<ValueType>> m_data;
    std::size_t m_rows,
                m_columns;
};

template <typename ValueType>
bool operator==(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    if (std::pair{lArg.rows(), lArg.columns()} != std::pair{rArg.rows(), rArg.columns()})
        return false;

    for (std::size_t row = 0; row < lArg.rows(); ++row)
        if (lArg[row] != rArg[row])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& mat, const math_vector<ValueType>& vec)
{
    if (mat.columns() != vec.size())
        throw std::invalid_argument{
            "In order to do matrix-vector multiplication, the matrix must have as many columns as the vector"
            " has elements"
        };

    math_vector<ValueType> output(mat.rows(), 0);
    for (std::size_t row = 0; row < mat.rows(); ++row)
        output[row] = mat[row] * vec;

    return output;
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& vec, const matrix<ValueType>& mat)
{
    if (vec.size() != mat.rows())
        throw std::invalid_argument{
            "In order to do vector-matrix multiplication, the matrix must have as many rows as the vector"
            " has elements"
        };

    math_vector<ValueType> output(mat.columns(), 0);
    for (std::size_t column = 0; column < mat.columns(); ++column)
        output[column] = mat.get_column(column) * vec;

    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    if (lArg.columns() != rArg.rows())
        throw std::invalid_argument{
            "In order to multiply matrices A and B, A must have as many columns as B has rows"
        };

    matrix<ValueType> output(lArg.rows(), rArg.columns(), 0);
    for (std::size_t row = 0; row < output.rows(); ++row)
        for (std::size_t column = 0; column < output.columns(); ++column)
            output[row][column] = lArg[row] * rArg.get_column(column);

    return output;
}

template <typename ValueType>
auto operator+(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg -= rArg;
}

template <typename ValueType>
auto operator*(matrix<ValueType> mat, ValueType scalar)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, matrix<ValueType> mat)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator/(matrix<ValueType> mat, ValueType scalar)
{
    return mat /= scalar;
}

#endif //RIDGEREGRESSION_MATRIX_H