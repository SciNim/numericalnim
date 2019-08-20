import unittest, math, sequtils, arraymancer
import numericalnim

proc df(x: float): float = 4 * x^3 - 9.0 * x^2
let start = 6.0
let gamma = 0.01
let precision = 0.00001
let max_iters = 10000
let correct = 2.24996

var A = toSeq([4,1,1,3]).toTensor.reshape(2,2).astype(float64)
var x = toSeq([2.0, 1.0]).toTensor.reshape(2,1)
var b = toSeq([1.0,2.0]).toTensor.reshape(2,1)
let tol = 0.001
let correct_conjugate = toSeq([0.090909, 0.636363]).toTensor.reshape(2,1).astype(float64)

test "steepest_descent func":
    let value = steepest_descent(df, start, gamma, precision, max_iters)
    check isClose(value, correct, tol = 1e-1)

test "conjugate_gradient func":
    let value = conjugate_gradient(A,b,x,tol)
    check isClose(value, correct_conjugate, tol = 1e-1)


