import unittest, math, sequtils
import arraymancer
import numericalnim

proc f(x: float, optional: seq[float]): float = optional[0] * cos(x)
proc fVector(x: float, optional: seq[Vector[float]]): Vector[float] = newVector([optional[0][0] * cos(x), optional[0][0] * cos(x), optional[0][0] * cos(x)])
proc fTensor(x: float, optional: seq[Tensor[float]]): Tensor[float] = @[optional[0][0] * cos(x), optional[0][0] * cos(x), optional[0][0] * cos(x)].toTensor()
let xStart = 0.0
let xEnd = 3.0/2.0*PI
let optional = @[2.0]
let optionalVector = @[newVector(optional)]
let optionalTensor = @[optional.toTensor()]
let correct = optional[0] * sin(xEnd)
let correctVector = newVector([optional[0] * sin(xEnd), optional[0] * sin(xEnd), optional[0] * sin(xEnd)])
let correctTensor = @[optional[0] * sin(xEnd), optional[0] * sin(xEnd), optional[0] * sin(xEnd)].toTensor()
let X = linspace(xStart, xEnd, 17)
let Y = map(X, proc(x: float): float = f(x, optional))
let cumY = map(X, proc(x: float): float = optional[0] * sin(x))

test "trapz func, N = 10":
    let value = trapz(f, xStart, xEnd, N = 10, optional = optional)
    check isClose(value, correct, tol = 1e-1)

test "trapz func, N = 100":
    let value = trapz(f, xStart, xEnd, N = 100, optional = optional)
    check isClose(value, correct, tol = 1e-3)

test "simpson func, N = 10":
    let value = simpson(f, xStart, xEnd, N = 10, optional = optional)
    check isClose(value, correct, tol = 1e-3)

test "simpson func, N = 100":
    let value = simpson(f, xStart, xEnd, N = 100, optional = optional)
    check isClose(value, correct, tol = 1e-6)

test "adaptive simpson, tol=1e-3":
    let value = adaptiveSimpson(f, xStart, xEnd, tol=1e-3, optional = optional)
    check isClose(value, correct, tol=1e-3)

test "adaptive simpson, tol=1e-8":
    let value = adaptiveSimpson(f, xStart, xEnd, tol=1e-8, optional = optional)
    check isClose(value, correct, tol=1e-8)

test "romberg func, depth = 3":
    let value = romberg(f, xStart, xEnd, depth = 3, optional = optional, tol=1e-8)
    check isClose(value, correct, tol = 1e-1)

test "romberg func, default":
    let value = romberg(f, xStart, xEnd, optional = optional)
    check isClose(value, correct, tol = 1e-8)

test "trapz discrete points":
    let value = trapz(Y, X)
    check isClose(correct, value, tol = 1e-1)

test "simpson discrete points":
    let value = simpson(Y, X)
    check isClose(correct, value, tol = 1e-3)

test "romberg discrete points":
    let value = romberg(Y, X)
    check isClose(correct, value, tol = 1e-3)

test "cumtrapz discrete points":
    let values = cumtrapz(Y, X)
    for i, value in values:
        check isClose(value, cumY[i], tol=1e-1)

test "cumtrapz func, discrete points":
    let values = cumtrapz(f, X, optional = optional)
    for i, value in values:
        check isClose(value, cumY[i], tol=1e-1)

test "cumtrapz func, dx = 0.1":
    let values = cumtrapz(f, X, dx = 0.1, optional = optional)
    for i, value in values:
        check isClose(value, cumY[i], tol=1e-1)

test "cumsimpson discrete points":
    let values = cumsimpson(Y, X)
    for i, value in values:
        check isClose(value, cumY[i], tol=1e-3)

test "cumsimpson func, discrete points":
    let values = cumsimpson(f, X, optional = optional)
    for i, value in values:
        check isClose(value, cumY[i], tol=1e-3)

test "cumsimpson func, dx = 0.1":
    let values = cumsimpson(f, X, dx = 0.1, optional = optional)
    for i, value in values:
        check isClose(value, cumY[i], tol=1e-3)


test "trapz Vector func, N = 10":
    let value = trapz(fVector, xStart, xEnd, N = 10, optional = optionalVector)
    check isClose(value, correctVector, tol = 1e-1)

test "trapz Vector func, N = 100":
    let value = trapz(fVector, xStart, xEnd, N = 100, optional = optionalVector)
    check isClose(value, correctVector, tol = 1e-3)

test "simpson Vector func, N = 10":
    let value = simpson(fVector, xStart, xEnd, N = 10, optional = optionalVector)
    check isClose(value, correctVector, tol = 1e-3)

test "simpson Vector func, N = 100":
    let value = simpson(fVector, xStart, xEnd, N = 100, optional = optionalVector)
    check isClose(value, correctVector, tol = 1e-6)

test "adaptive Vector simpson, tol=1e-3":
    let value = adaptiveSimpson(fVector, xStart, xEnd, tol=1e-3, optional = optionalVector)
    check isClose(value, correctVector, tol=1e-3)

test "adaptive Vector simpson, tol=1e-8":
    let value = adaptiveSimpson(fVector, xStart, xEnd, tol=1e-8, optional = optionalVector)
    check isClose(value, correctVector, tol=1e-8)

test "romberg Vector func, depth = 3":
    let value = romberg(fVector, xStart, xEnd, depth = 3, optional = optionalVector, tol=1e-8)
    check isClose(value, correctVector, tol = 1e-1)

test "romberg Vector func, default":
    let value = romberg(fVector, xStart, xEnd, optional = optionalVector)
    check isClose(value, correctVector, tol = 1e-8)


test "trapz Tensor func, N = 10":
    let value = trapz(fTensor, xStart, xEnd, N = 10, optional = optionalTensor)
    check isClose(value, correctTensor, tol = 1e-1)

test "trapz Tensor func, N = 100":
    let value = trapz(fTensor, xStart, xEnd, N = 100, optional = optionalTensor)
    check isClose(value, correctTensor, tol = 1e-3)

test "simpson Tensor func, N = 10":
    let value = simpson(fTensor, xStart, xEnd, N = 10, optional = optionalTensor)
    check isClose(value, correctTensor, tol = 1e-3)

test "simpson Tensor func, N = 100":
    let value = simpson(fTensor, xStart, xEnd, N = 100, optional = optionalTensor)
    check isClose(value, correctTensor, tol = 1e-6)

test "adaptive Tensor simpson, tol=1e-3":
    let value = adaptiveSimpson(fTensor, xStart, xEnd, tol=1e-3, optional = optionalTensor)
    check isClose(value, correctTensor, tol=1e-3)

test "adaptive Tensor simpson, tol=1e-8":
    let value = adaptiveSimpson(fTensor, xStart, xEnd, tol=1e-8, optional = optionalTensor)
    check isClose(value, correctTensor, tol=1e-8)

test "romberg Tensor func, depth = 3":
    let value = romberg(fTensor, xStart, xEnd, depth = 3, optional = optionalTensor, tol=1e-8)
    check isClose(value, correctTensor, tol = 1e-1)

test "romberg Tensor func, default":
    let value = romberg(fTensor, xStart, xEnd, optional = optionalTensor)
    check isClose(value, correctTensor, tol = 1e-8)



