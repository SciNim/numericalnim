import unittest, math, sequtils
import numericalnim
import arraymancer

proc f(x: float): float = sin(x)
let t = linspace(0.0, 10.0, 100)
let y = t.map(f)
let cubicSpline = newCubicSpline(t, y)
let hermiteSpline = newHermiteSpline(t, y)
let tTest = arange(0.0, 10.0, 0.2345, includeStart=true, includeEnd=false)
let yTest = tTest.map(f)
let cubicSplineProc = cubicSpline.toProc
let hermiteSplineProc = hermiteSpline.toProc

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


test "HermiteSpline Eval in input points, direct":
    let res = hermiteSpline.eval(t)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "HermiteSpline Eval in input points, for loop":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = hermiteSpline.eval(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)
    
test "HermiteSpline Eval between input points":
    let res = hermiteSpline.eval(tTest)
    for i, val in res:
        check isClose(val, yTest[i], 1e-4)

test "HermiteSpline.toProc, single value":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = hermiteSplineProc(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "HermiteSpline Integrate using adaptiveSimpson, implicit":
    let computedValue = adaptiveSimpson(hermiteSpline, 0.0, 7.5)
    let correct = -cos(7.5) + cos(0.0)
    #check isClose(computedValue, correct, tol=1e-7)
    check abs(computedValue - correct) < 2e-6

test "HermiteSpline derivEval, single value":
    let res = hermiteSpline.derivEval(t[20])
    let correct = cos(t[20])
    #check isClose(res, correct, 1e-6)
    check abs(res - correct) < 1e-3

test "HermiteSpline derivEval, seq input":
    let res = hermiteSpline.derivEval(tTest)
    for i, val in res:
        #check isClose(val, cos(tTest[i]), 1e-3)
        check abs(val - cos(tTest[i])) < 2e-3

