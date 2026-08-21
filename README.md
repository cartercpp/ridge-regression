# Ridge Regression on Bitcoin (from scratch in C++)

A pure C++ implementation of ridge regression used to forecast normalized Bitcoin prices with a sliding window of past values.

This is an educational project focused on:
- Implementing the closed-form ridge regression solution from scratch
- Writing a small linear algebra library (`matrix` + `math_vector`)
- Visualizing rolling predictions in the terminal with ANSI colors

## Math

We solve the classic ridge regression problem:

\[
\mathbf{w} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}
\]

Where each row of \(X\) is a window of the previous `windowSize` normalized prices, and \(y\) is the next price.

## Features

- Custom `math_vector` and `matrix` classes (no Eigen / no external linear algebra libraries)
- Gauss-Jordan elimination with partial pivoting for matrix inversion
- Min-max normalization of the price series
- Live terminal visualization of historical prices vs model predictions
- Both in-sample and walk-forward (out-of-sample) modes

## Requirements

- C++23 compiler (`std::print`)
- A terminal that supports ANSI colors and cursor control

## Building

```bash
g++ -std=c++23 -O2 main.cpp -o ridge
