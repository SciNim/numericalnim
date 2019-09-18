import unittest, math, sequtils
import numericalnim
import arraymancer

proc f(x: float): float = sin(x)
let t = linspace(0.0, 10.0, 100)
let y = t.map(f)
let spline = newCubicSpline(t, y)
let tTest = arange(0.0, 10.0, 0.2345, true, false)
let yTest = tTest.map(f)
let splineProc = spline.toProc

test "Eval in input points, direct":
    let res = spline.eval(t)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "Eval in input points, for loop":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = spline.eval(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)
    
test "Eval between input points":
    let res = spline.eval(tTest)
    for i, val in res:
        check isClose(val, yTest[i], 1e-4)

test "Spline.toProc, single value":
    var res = newSeq[float](t.len)
    for i, tTemp in t:
        res[i] = splineProc(tTemp)
    for i, val in res:
        check isClose(val, y[i], 1e-15)

test "Integrate using adaptiveSimpson, implicit":
    let computedValue = adaptiveSimpson(spline, 0.0, 7.5)
    let correct = -cos(7.5) + cos(0.0)
    check isClose(computedValue, correct, tol=1e-7)


# test deriv






