#include <print>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <ranges>
#include <thread>
#include <chrono>
#include "matrix.h"

int main()
{
    std::vector<double> normBTCPriceHistory;
    normBTCPriceHistory.reserve(5499);

    {
        std::ifstream file{"/home/cartercpp/Documents/C++/RidgeRegression/btc.txt"};
        double price;
        while (file >> price)
            normBTCPriceHistory.push_back(price);

        const double minPrice = std::ranges::min(normBTCPriceHistory),
                     maxPrice = std::ranges::max(normBTCPriceHistory);
        for (double& priceRef : normBTCPriceHistory)
            priceRef = (priceRef - minPrice) / (maxPrice - minPrice);
    }

    constexpr std::size_t windowSize = 450; // change if necessary
    constexpr double lambda = 1; // change if necessary

    matrix<double> input(normBTCPriceHistory.size() - windowSize, windowSize);
    math_vector<double> target(normBTCPriceHistory.size() - windowSize);

    for (std::size_t start = 0; start + windowSize < normBTCPriceHistory.size(); ++start)
    {
        for (std::size_t i = 0; i < windowSize; ++i)
            input[start][i] = normBTCPriceHistory[start + i];

        target[start] = normBTCPriceHistory[start + windowSize];
    }

    const auto weights
        = (input.transpose() * input + lambda * matrix<double>::identity(windowSize)).inverse()
        * input.transpose() * target;

    constexpr std::size_t rows = 90,
                          columns = windowSize + 1;

    char grid[rows][columns];

    auto resetGrid = [&grid]()
    {
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                grid[row][column] = ' ';
    };

    std::print("\033[?25l"); // cursor off
    std::print("\033[2J"); // clear screen

    std::deque<double> normPredictionWindow;
    for (std::size_t start = 0; start + windowSize < normBTCPriceHistory.size(); ++start)
    {
        const std::vector<double>
            normPriceWindow(normBTCPriceHistory.begin() + start, normBTCPriceHistory.begin() + start + windowSize);

        const double normPredictedPrice = weights * math_vector<double>{normPriceWindow};
        normPredictionWindow.push_back(normPredictedPrice);

        if (normPredictionWindow.size() > columns)
            normPredictionWindow.pop_front();

        resetGrid();

        for (std::size_t column = 0; column < windowSize; ++column)
        {
            const double normPrice = normBTCPriceHistory[start + column];
            const std::size_t row = rows - 1 - static_cast<std::size_t>(normPrice * (rows - 1));
            grid[row][column] = '-';
        }

        for (std::size_t i = 0; i < normPredictionWindow.size(); ++i)
        {
            const double normPrediction = normPredictionWindow[normPredictionWindow.size() - i - 1];
            const std::size_t column = columns - 1 - i,
                              row = rows - 1 - static_cast<std::size_t>(normPrediction * (rows - 1));
            grid[row][column] = '/';
        }

        std::print("\033[1;1H");
        for (std::size_t row = 0; row < rows; ++row)
        {
            for (std::size_t column = 0; column < columns; ++column)
            {
                switch (grid[row][column])
                {
                    case '/':
                        std::print("\033[38;2;0;0;255m");
                        break;
                    case '-':
                        std::print("\033[38;2;0;255;255m");
                        break;
                }

                std::print("{}", grid[row][column]);
            }

            std::println();
        }
    }

    std::print("\033[?25h"); // restore cursor
}