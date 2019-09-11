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


test "gaussQuad func n=2":
    let value = gaussQuad(f, xStart, xEnd, nPoints=2, N=1, optional=optional)
    check isClose(value, correct, tol=1)

test "gaussQuad func n=2 N=10":
    let value = gaussQuad(f, xStart, xEnd, nPoints=2, N=10, optional=optional)
    check isClose(value, correct, tol=1e-2)

test "gaussQuad func n=3":
    let value = gaussQuad(f, xStart, xEnd, nPoints=3, N=1, optional=optional)
    check isClose(value, correct, tol=1e-1)

test "gaussQuad func n=4":
    let value = gaussQuad(f, xStart, xEnd, nPoints=4, N=1, optional=optional)
    check isClose(value, correct, tol=1e-3)

test "gaussQuad func n=5":
    let value = gaussQuad(f, xStart, xEnd, nPoints=5, N=1, optional=optional)
    check isClose(value, correct, tol=1e-4)

test "gaussQuad func n=5 N=10":
    let value = gaussQuad(f, xStart, xEnd, nPoints=5, N=10, optional=optional)
    check isClose(value, correct, tol=1e-6)

test "gaussQuad func n=6":
    let value = gaussQuad(f, xStart, xEnd, nPoints=6, N=1, optional=optional)
    check isClose(value, correct, tol=1e-5)

test "gaussQuad func n=7":
    let value = gaussQuad(f, xStart, xEnd, nPoints=7, N=1, optional=optional)
    check isClose(value, correct, tol=1e-6)

test "gaussQuad func n=8":
    let value = gaussQuad(f, xStart, xEnd, nPoints=8, N=1, optional=optional)
    check isClose(value, correct, tol=1e-6)

test "gaussQuad func n=9":
    let value = gaussQuad(f, xStart, xEnd, nPoints=9, N=1, optional=optional)
    check isClose(value, correct, tol=1e-7)

test "gaussQuad func n=10":
    let value = gaussQuad(f, xStart, xEnd, nPoints=10, N=1, optional=optional)
    check isClose(value, correct, tol=1e-7)

test "gaussQuad func n=10 N=10":
    let value = gaussQuad(f, xStart, xEnd, nPoints=10, N=10, optional=optional)
    check isClose(value, correct, tol=1e-9)

test "gaussQuad func n=11":
    let value = gaussQuad(f, xStart, xEnd, nPoints=11, N=1, optional=optional)
    check isClose(value, correct, tol=1e-8)

test "gaussQuad func n=12":
    let value = gaussQuad(f, xStart, xEnd, nPoints=12, N=1, optional=optional)
    check isClose(value, correct, tol=1e-8)

test "gaussQuad func n=13":
    let value = gaussQuad(f, xStart, xEnd, nPoints=13, N=1, optional=optional)
    check isClose(value, correct, tol=1e-9)

test "gaussQuad func n=14":
    let value = gaussQuad(f, xStart, xEnd, nPoints=14, N=1, optional=optional)
    check isClose(value, correct, tol=1e-9)
        
test "gaussQuad func n=15":
    let value = gaussQuad(f, xStart, xEnd, nPoints=15, N=1, optional=optional)
    check isClose(value, correct, tol=1e-10)

test "gaussQuad func n=16":
    let value = gaussQuad(f, xStart, xEnd, nPoints=16, N=1, optional=optional)
    check isClose(value, correct, tol=1e-10)

test "gaussQuad func n=17":
    let value = gaussQuad(f, xStart, xEnd, nPoints=17, N=1, optional=optional)
    check isClose(value, correct, tol=1e-11)

test "gaussQuad func n=18":
    let value = gaussQuad(f, xStart, xEnd, nPoints=18, N=1, optional=optional)
    check isClose(value, correct, tol=1e-12)

test "gaussQuad func n=19":
    let value = gaussQuad(f, xStart, xEnd, nPoints=19, N=1, optional=optional)
    check isClose(value, correct, tol=1e-13)

test "gaussQuad func n=20":
    let value = gaussQuad(f, xStart, xEnd, nPoints=20, N=1, optional=optional)
    check isClose(value, correct, tol=1e-14)

test "gaussQuad func n=20 N=10":
    let value = gaussQuad(f, xStart, xEnd, nPoints=20, N=10, optional=optional)
    check isClose(value, correct, tol=1e-14)

test "gaussAdaptive func default":
    let value = adaptiveGauss(f, xStart, xEnd, optional=optional)
    check isClose(value, correct, 1e-8)

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

test "gaussQuad Vector n=2":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=2, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1.5)

test "gaussQuad Vector n=3":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=3, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-1)

test "gaussQuad Vector n=4":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=4, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-2)

test "gaussQuad Vector n=5":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=5, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-4)

test "gaussQuad Vector n=6":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=6, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-5)

test "gaussQuad Vector n=7":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=7, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-6)

test "gaussQuad Vector n=8":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=8, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-6)

test "gaussQuad Vector n=9":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=9, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-7)

test "gaussQuad Vector n=10":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=10, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-7)

test "gaussQuad Vector n=10 N=10":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=10, N=10, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-8)

test "gaussQuad Vector n=11":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=11, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-8)

test "gaussQuad Vector n=12":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=12, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-8)

test "gaussQuad Vector n=13":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=13, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-9)

test "gaussQuad Vector n=14":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=14, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-9)
        
test "gaussQuad Vector n=15":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=15, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-10)

test "gaussQuad Vector n=16":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=16, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-10)

test "gaussQuad Vector n=17":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=17, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-11)

test "gaussQuad Vector n=18":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=18, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-12)

test "gaussQuad Vector n=19":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=19, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-13)

test "gaussQuad Vector n=20":
    let value = gaussQuad(fVector, xStart, xEnd, nPoints=20, N=1, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-14)

test "adaptiveGauss Vector default":
    let value = adaptiveGauss(fVector, xStart, xEnd, optional=optionalVector)
    check isClose(value, correctVector, tol=1e-8)

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

test "gaussQuad Tensor n=2":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=2, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1.5)

test "gaussQuad Tensor n=3":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=3, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-1)

test "gaussQuad Tensor n=4":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=4, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-2)

test "gaussQuad Tensor n=5":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=5, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-4)

test "gaussQuad Tensor n=6":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=6, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-5)

test "gaussQuad Tensor n=7":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=7, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-6)

test "gaussQuad Tensor n=8":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=8, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-6)

test "gaussQuad Tensor n=9":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=9, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-7)

test "gaussQuad Tensor n=10":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=10, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-7)

test "gaussQuad Tensor n=10 N=10":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=10, N=10, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-8)

test "gaussQuad Tensor n=11":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=11, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-8)

test "gaussQuad Tensor n=12":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=12, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-8)

test "gaussQuad Tensor n=13":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=13, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-9)

test "gaussQuad Tensor n=14":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=14, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-9)
        
test "gaussQuad Tensor n=15":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=15, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-10)

test "gaussQuad Tensor n=16":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=16, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-10)

test "gaussQuad Tensor n=17":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=17, N=1, optional=optionalTensor)    
    check isClose(value, correctTensor, tol=1e-11)

test "gaussQuad Tensor n=18":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=18, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-12)

test "gaussQuad Tensor n=19":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=19, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-13)

test "gaussQuad Tensor n=20":
    let value = gaussQuad(fTensor, xStart, xEnd, nPoints=20, N=1, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-14)

test "adaptiveGauss Tensor default":
    let value = adaptiveGauss(fTensor, xStart, xEnd, optional=optionalTensor)
    check isClose(value, correctTensor, tol=1e-8)


