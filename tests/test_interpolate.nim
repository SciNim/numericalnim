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
let linear1DSpline = newLinear1D(t, y) 
let tTest = arange(0.0, 10.0, 0.2345, includeStart=true, includeEnd=false)
let yTest = tTest.map(f)
let cubicSplineProc = cubicSpline.toProc
let hermiteSplineProc = hermiteSpline.toProc
let hermiteSpline2Proc = hermiteSpline2.toProc
let linear1DSplineProc = linear1DSpline.toProc

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



test "linear1DSpline Eval in input points, direct":
    let res = linear1DSpline.eval(t)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "linear1DSpline Eval in input points, for loop":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = linear1DSpline.eval(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)
    
test "linear1DSpline Eval between input points":
    let res = linear1DSpline.eval(tTest)
    for i, val in res:
        check isClose(val, yTest[i], 5e-3)

test "linear1DSpline.toProc, single value":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = linear1DSplineProc(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "linear1DSpline Integrate using adaptiveSimpson, implicit":
    let computedValue = adaptiveSimpson(linear1DSpline, 0.0, 7.5)
    let correct = -cos(7.5) + cos(0.0)
    check isClose(computedValue, correct, tol=1e-3)

test "linear1DSpline derivEval, single value":
    let res = linear1DSpline.derivEval(t[20])
    let correct = cos(t[20])
    check isClose(res, correct, 5e-2)

test "linear1DSpline derivEval, seq input":
    let res = linear1DSpline.derivEval(tTest)
    for i, val in res:
        check isClose(val, cos(tTest[i]), 5e-2)



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

# Unstructured 2D Interpolation

let (gridX, gridY) = meshgridFlat(arraymancer.linspace(0.0, 10.0, 50), arraymancer.linspace(0.0, 10.0, 50))
let pointsXY = gridX.unsqueeze(1).concat(gridY.unsqueeze(1), axis=1)

test "Barycentric2D Ones":
    let bary = newBarycentric2D(pointsXY, ones_like(gridX))
    let x = linspace(0.0, 10.0, 789)
    let y = linspace(0.0, 10.0, 567)
    #echo bary.eval(0.8756345177664975, 0.0)
    for i in x:
        for j in y:
            check abs(bary.eval(i, j) - 1) < 1.2e-16
    

# 3D Interpolation

test "Trilinear all ones":
    let f = ones[float](100, 100, 100)
    let spline = newTrilinearSpline(f, (0.0, 100.0), (-100.0, 0.0), (-50.0, 50.0))
    let x = linspace(0.0, 100.0, 113)
    let y = linspace(-100.0, 0.0, 57)
    let z = linspace(-50, 50, 21)
    for i in x:
        for j in y:
            for k in z:
                check abs(spline.eval(i, j, k) - 1.0) < 1e-14

test "Trilinear f = x*y*z":
    var f = zeros[float](101, 101, 101)
    for i in 0..<101:
        for j in 0..<101:
            for k in 0..<101:
                f[i, j, k] = i.toFloat * j.toFloat * k.toFloat
    let spline = newTrilinearSpline(f, (0.0, 100.0), (0.0, 100.0), (0.0, 100.0))
    let x = linspace(0.0, 100, 37)
    let y = linspace(0.0, 100, 57)
    let z = linspace(0, 100, 24)
    for i in x:
       for j in y:
           for k in z:
               check abs(spline.eval(i, j, k) - i*j*k) < 3e-10

test "Trilinear f = x*y*z different sizes":
    # x: (0, 10)
    # y: (0, 2)
    # z: (-20, 0)
    var f = zeros[float](101, 101, 101)
    for i in 0..<101:
        for j in 0..<101:
            for k in 0..<101:
                f[i, j, k] = 10/100*i.toFloat * 2/100*j.toFloat * 50/100*k.toFloat
    let spline = newTrilinearSpline(f, (0.0, 10.0), (0.0, 2.0), (0.0, 50.0))
    let x = linspace(0.0, 10, 37)
    let y = linspace(0.0, 2, 57)
    let z = linspace(0.0, 50, 23)
    for i in x:
       for j in y:
           for k in z:
               check abs(spline.eval(i, j, k) - i*j*k) < 1e-12

test "Trilinear f = x*y*z T: Vector[float]":
    # x: (0, 10)
    # y: (0, 2)
    # z: (-20, 0)
    var f = newTensor[Vector[float]](101, 101, 101)
    for i in 0..<101:
        for j in 0..<101:
            for k in 0..<101:
                f[i, j, k] = newVector([10/100*i.toFloat * 2/100*j.toFloat * 50/100*k.toFloat, 10/100*i.toFloat * 2/100*j.toFloat * 50/100*k.toFloat, 1])
    let spline = newTrilinearSpline(f, (0.0, 10.0), (0.0, 2.0), (0.0, 50.0))
    let x = linspace(0.0, 10, 37)
    let y = linspace(0.0, 2, 57)
    let z = linspace(0.0, 50, 23)
    for i in x:
       for j in y:
           for k in z:
               check abs(spline.eval(i, j, k)[0] - i*j*k) < 1e-12
               check abs(spline.eval(i, j, k)[1] - i*j*k) < 1e-12
               check abs(spline.eval(i, j, k)[2] - 1) < 1e-16

test "Trilinear f = x*y*z T: Tensor[float]":
    # x: (0, 10)
    # y: (0, 2)
    # z: (-20, 0)
    var f = newTensor[Tensor[float]](101, 101, 101)
    for i in 0..<101:
        for j in 0..<101:
            for k in 0..<101:
                f[i, j, k] = toTensor([10/100*i.toFloat * 2/100*j.toFloat * 50/100*k.toFloat, 10/100*i.toFloat * 2/100*j.toFloat * 50/100*k.toFloat, 1])
    let spline = newTrilinearSpline(f, (0.0, 10.0), (0.0, 2.0), (0.0, 50.0))
    let x = linspace(0.0, 10, 37)
    let y = linspace(0.0, 2, 57)
    let z = linspace(0.0, 50, 23)
    for i in x:
       for j in y:
           for k in z:
               check abs(spline.eval(i, j, k)[0] - i*j*k) < 1e-12
               check abs(spline.eval(i, j, k)[1] - i*j*k) < 1e-12
               check abs(spline.eval(i, j, k)[2] - 1) < 1e-16

test "rbf f=x*y*z":
    let pos = meshgrid(arraymancer.linspace(0.0, 1.0, 5), arraymancer.linspace(0.0, 1.0, 5), arraymancer.linspace(0.0, 1.0, 5))
    let vals = pos[_, 0] *. pos[_, 1] *. pos[_, 2]
    let rbfObj = newRbf(pos, vals)

    # We want test points in the interior to avoid the edges
    let xTest = meshgrid(arraymancer.linspace(0.1, 0.9, 10), arraymancer.linspace(0.1, 0.9, 10), arraymancer.linspace(0.1, 0.9, 10))
    let yTest = rbfObj.eval(xTest)
    let yCorrect = xTest[_, 0] *. xTest[_, 1] *. xTest[_, 2]
    for x in abs(yCorrect - yTest):
        check x < 0.16
    check mean_squared_error(yTest, yCorrect) < 2e-4
