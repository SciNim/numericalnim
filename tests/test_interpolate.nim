import unittest, math, sequtils
import numericalnim
import arraymancer


proc f(x: float): float = sin(x)
proc df(x: float): float = cos(x)
let t = linspace(0.0, 10.0, 100)
let y = t.map(f)
let dy = t.map(df)
let cubicSpline = newCubicSpline(t, y)
let hermiteSpline = newHermiteSpline(t, y)
let hermiteSpline2 = newHermiteSpline(t, y, dy)
let tTest = arange(0.0, 10.0, 0.2345, includeStart=true, includeEnd=false)
let yTest = tTest.map(f)
let cubicSplineProc = cubicSpline.toProc
let hermiteSplineProc = hermiteSpline.toProc
let hermiteSpline2Proc = hermiteSpline2.toProc

test "CubicSpline Eval in input points, direct":
    let res = cubicSpline.eval(t)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "CubicSpline Eval in input points, for loop":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = cubicSpline.eval(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)
    
test "CubicSpline Eval between input points":
    let res = cubicSpline.eval(tTest)
    for i, val in res:
        check isClose(val, yTest[i], 1e-4)

test "CubicSpline.toProc, single value":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = cubicSplineProc(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "CubicSpline Integrate using adaptiveSimpson, implicit":
    let computedValue = adaptiveSimpson(cubicSpline, 0.0, 7.5)
    let correct = -cos(7.5) + cos(0.0)
    check isClose(computedValue, correct, tol=1e-7)

test "CubicSpline derivEval, single value":
    let res = cubicSpline.derivEval(t[20])
    let correct = cos(t[20])
    check isClose(res, correct, 1e-6)

test "CubicSpline derivEval, seq input":
    let res = cubicSpline.derivEval(tTest)
    for i, val in res:
        check isClose(val, cos(tTest[i]), 1e-3)


test "HermiteSpline (without dY) Eval in input points, direct":
    let res = hermiteSpline.eval(t)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "HermiteSpline (without dY) Eval in input points, for loop":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = hermiteSpline.eval(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)
    
test "HermiteSpline (without dY) Eval between input points":
    let res = hermiteSpline.eval(tTest)
    for i, val in res:
        check isClose(val, yTest[i], 1e-4)

test "HermiteSpline (without dY).toProc, single value":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = hermiteSplineProc(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "HermiteSpline (without dY) Integrate using adaptiveSimpson, implicit":
    let computedValue = adaptiveSimpson(hermiteSpline, 0.0, 7.5)
    let correct = -cos(7.5) + cos(0.0)
    #check isClose(computedValue, correct, tol=1e-7)
    check abs(computedValue - correct) < 2e-6

test "HermiteSpline (without dY) derivEval, single value":
    let res = hermiteSpline.derivEval(t[20])
    let correct = cos(t[20])
    #check isClose(res, correct, 1e-6)
    check abs(res - correct) < 1e-3

test "HermiteSpline (without dY) derivEval, seq input":
    let res = hermiteSpline.derivEval(tTest)
    for i, val in res:
        #check isClose(val, cos(tTest[i]), 1e-3)
        check abs(val - cos(tTest[i])) < 2e-3


test "HermiteSpline Eval in input points, direct":
    let res = hermiteSpline2.eval(t)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "HermiteSpline Eval in input points, for loop":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = hermiteSpline2.eval(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)
    
test "HermiteSpline Eval between input points":
    let res = hermiteSpline2.eval(tTest)
    for i, val in res:
        check isClose(val, yTest[i], 1e-4)

test "HermiteSpline (without dY).toProc, single value":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = hermiteSpline2Proc(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "HermiteSpline Integrate using adaptiveSimpson, implicit":
    let computedValue = adaptiveSimpson(hermiteSpline2, 0.0, 7.5)
    let correct = -cos(7.5) + cos(0.0)
    #check isClose(computedValue, correct, tol=1e-7)
    check abs(computedValue - correct) < 1e-7

test "HermiteSpline derivEval, single value":
    let res = hermiteSpline2.derivEval(t[20])
    let correct = cos(t[20])
    #check isClose(res, correct, 1e-6)
    check abs(res - correct) < 1e-9

test "HermiteSpline derivEval, seq input":
    let res = hermiteSpline2.derivEval(tTest)
    for i, val in res:
        #check isClose(val, cos(tTest[i]), 1e-3)
        check abs(val - cos(tTest[i])) < 1e-5

# 2D Interpolation

let onesZ = ones[float](100, 100)

test "Nearest neighbour all ones":
    let nn = newNearestNeighbour2D(onesZ, (0.0, 100.0), (-100.0, 0.0))
    let x = linspace(0.0, 100.0, 789)
    let y = linspace(-100.0, 0.0, 567)
    for i in x:
        for j in y:
            check nn.eval(i, j) == 1.0

test "Bilinear all ones":
    let spline = newBilinearSpline(onesZ, (0.0, 100.0), (-100.0, 0.0))
    let x = linspace(0.0, 100.0, 789)
    let y = linspace(-100.0, 0.0, 567)
    for i in x:
        for j in y:
            check abs(spline.eval(i, j) - 1.0) < 1e-15

test "Bicubic all ones":
    let spline = newBicubicSpline(onesZ, (0.0, 100.0), (-100.0, 0.0))
    let x = linspace(0.0, 100.0, 789)
    let y = linspace(-100.0, 0.0, 567)
    for i in x:
        for j in y:
            check abs(spline.eval(i, j) - 1.0) < 1e-16

let zPoints = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2, 2]].toTensor
let xPoints = [0.0, 1.0, 2.0]
let xTestPoints = linspace(0.0, 2.0, 7)
var nnCorrect = newTensor[float](7, 7)
for i in 0 ..< 7:
    for j in 0 ..< 7:
        if i < 2:
            nnCorrect[i, j] = 0.0
        elif i > 4:
            nnCorrect[i, j] = 2.0
        else:
            nnCorrect[i, j] = 1.0

test "Nearest neighbour 3x3":
    let nn = newNearestNeighbour2D(zPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check nn.eval(x, y) == nnCorrect[i, j]

var bilinearCorrect = newTensor[float](7,7)
for i in 0 ..< 7:
    for j in 0 ..< 7:
        bilinearCorrect[i, j] = i.toFloat / 3

test "Bilinear 3x3":
    let spline = newBilinearSpline(zPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check abs(spline.eval(x, y) - bilinearCorrect[i, j]) < 3e-16

test "Bicubic 3x3":
    let spline = newBicubicSpline(zPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check abs(spline.eval(x, y) - bilinearCorrect[i, j]) < 3e-16

let zVectorPoints: Tensor[Vector[float]] = [
    [newVector[float]([0.0, 0.0, 0.0]), newVector([0.0, 0.0, 0.0]), newVector([0.0, 0.0, 0.0])], 
    [newVector[float]([1.0, 1.0, 1.0]), newVector([1.0, 1.0, 1.0]), newVector([1.0, 1.0, 1.0])], 
    [newVector[float]([2.0, 2.0, 2.0]), newVector([2.0, 2.0, 2.0]), newVector([2.0, 2.0, 2.0])]
    ].toTensor

test "Nearest neighbour T: Vector[float]":
    let nn = newNearestNeighbour2D(zVectorPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check nn.eval(x, y)[0] == nnCorrect[i, j]
            check nn.eval(x, y)[1] == nnCorrect[i, j]
            check nn.eval(x, y)[2] == nnCorrect[i, j]

test "Bilinear T: Vector[float]":
    let nn = newBilinearSpline(zVectorPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check abs(nn.eval(x, y)[0] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[1] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[2] - bilinearCorrect[i, j]) < 3e-16

test "Bicubic T: Vector[float]":
    let nn = newBicubicSpline(zVectorPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check abs(nn.eval(x, y)[0] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[1] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[2] - bilinearCorrect[i, j]) < 3e-16

let zTensorPoints: Tensor[Tensor[float]] = [
    [toTensor([0.0, 0.0, 0.0]), toTensor([0.0, 0.0, 0.0]), toTensor([0.0, 0.0, 0.0])], 
    [toTensor([1.0, 1.0, 1.0]), toTensor([1.0, 1.0, 1.0]), toTensor([1.0, 1.0, 1.0])], 
    [toTensor([2.0, 2.0, 2.0]), toTensor([2.0, 2.0, 2.0]), toTensor([2.0, 2.0, 2.0])]
    ].toTensor

test "Nearest neighbour T: Tensor[float]":
    let nn = newNearestNeighbour2D(zTensorPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check nn.eval(x, y)[0] == nnCorrect[i, j]
            check nn.eval(x, y)[1] == nnCorrect[i, j]
            check nn.eval(x, y)[2] == nnCorrect[i, j]

test "Bilinear T: Tensor[float]":
    let nn = newBilinearSpline(zTensorPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check abs(nn.eval(x, y)[0] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[1] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[2] - bilinearCorrect[i, j]) < 3e-16

test "Bicubic T: Tensor[float]":
    let nn = newBicubicSpline(zTensorPoints, (0.0, 2.0), (0.0, 2.0))
    for i, x in xTestPoints:
        for j, y in xTestPoints:
            check abs(nn.eval(x, y)[0] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[1] - bilinearCorrect[i, j]) < 3e-16
            check abs(nn.eval(x, y)[2] - bilinearCorrect[i, j]) < 3e-16
            