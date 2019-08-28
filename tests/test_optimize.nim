import unittest, math, sequtils, arraymancer
import numericalnim

test "steepest_descent func":
    proc df(x: float): float = 4 * x^3 - 9.0 * x^2
    let start = 6.0
    let gamma = 0.01
    let precision = 0.00001
    let max_iters = 10000
    let correct = 2.24996
    let value = steepest_descent(df, start, gamma, precision, max_iters)
    check isClose(value, correct, tol = 1e-1)

test "conjugate_gradient func":
    var A = toSeq([4.0, 1.0, 1.0, 3.0]).toTensor.reshape(2,2).astype(float64)
    var x = toSeq([2.0, 1.0]).toTensor.reshape(2,1)
    var b = toSeq([1.0,2.0]).toTensor.reshape(2,1)
    let tol = 0.001
    let correct = toSeq([0.090909, 0.636363]).toTensor.reshape(2,1).astype(float64)

    let value = conjugate_gradient(A, b, x, tol)
    check isClose(value, correct, tol = 1e-1)

test "Newtons 1 dimension func":
    proc f(x:float64): float64 = (1.0 / 3.0) * x ^ 3 - 2 * x ^ 2 + 3 * x
    proc df(x:float64): float64 = x ^ 2 - 4 * x + 3
    let x = 0.5
    let correct = 0.0
    let value = newtons(f, df, x, 0.000001, 1000)
    check isClose(value, correct, tol=1e-5)

test "Newtons 1 dimension func default args":
    proc f(x:float64): float64 = (1.0 / 3.0) * x ^ 3 - 2 * x ^ 2 + 3 * x
    proc df(x:float64): float64 = x ^ 2 - 4 * x + 3
    let x = 0.5
    let correct = 0.0
    let value = newtons(f, df, x)
    check isClose(value, correct, tol=1e-4)

test "Newtons unable to find a root":
    proc bad_f(x:float64): float64 = pow(E, x) + 1
    proc bad_df(x:float64): float64 = pow(E, x)
    expect(ArithmeticError):
        discard newtons(bad_f, bad_df, 0, 0.000001, 1000)


