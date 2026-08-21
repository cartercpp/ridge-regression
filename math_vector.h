//
// Created by cartercpp on 8/21/26.
//

#ifndef RIDGEREGRESSION_MATH_VECTOR_H
#define RIDGEREGRESSION_MATH_VECTOR_H

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <concepts>
#include <cstddef>

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class math_vector
{
public:

    // CONSTRUCTORS

    math_vector() = default;


    explicit math_vector(std::size_t size, ValueType value = 0)
        : m_data(size, value)
    {}

    math_vector(const std::vector<ValueType>& data)
        : m_data{data}
    {}

    // METHODS

    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_data.size();
    }

    math_vector multiply(const math_vector& other) const
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot multiply 2 vectors which are different sizes"};

        math_vector<ValueType> output(m_data.size(), 0);
        for (std::size_t i = 0; i < m_data.size(); ++i)
            output[i] = m_data[i] * other[i];

        return output;
    }

    math_vector& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef *= scalar;});
        return *this;
    }

    math_vector& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef /= scalar;});
        return *this;
    }

    math_vector& operator+=(const math_vector& other)
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot add 2 vectors with different sizes"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] += other[i];

        return *this;
    }

    math_vector& operator-=(const math_vector& other)
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"Cannot add 2 vectors with different sizes"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] -= other[i];

        return *this;
    }

    ValueType& operator[](std::size_t index)
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    const ValueType& operator[](std::size_t index) const
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

private:

    std::vector<ValueType> m_data;
};

template <typename ValueType>
bool operator==(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        return false;

    for (std::size_t i = 0; i < lArg.size(); ++i)
        if (lArg[i] != rArg[i])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        throw std::invalid_argument{
            "Cannot compute the dot product of A and B if A and B are not the same size"
        };

    ValueType output = 0;
    for (std::size_t i = 0; i < lArg.size(); ++i)
        output += lArg[i] * rArg[i];

    return output;
}

template <typename ValueType>
auto operator+(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg -= rArg;
}

template <typename ValueType>
auto operator*(math_vector<ValueType> vec, ValueType scalar)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, math_vector<ValueType> vec)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator/(math_vector<ValueType> vec, ValueType scalar)
{
    return vec /= scalar;
}

#endif //RIDGEREGRESSION_MATH_VECTOR_H