import strformat, math
import utils, integrate

# use binary search to find interval in eval(Spline)

type
  NewtonInterpolant*[T] = ref object
    X: seq[float]
    Y: seq[T]
    high: int
    len: int
  CubicSpline*[T] = ref object
    X: seq[float]
    coeffs: seq[array[5, float]]
    high: int
    len: int

proc constructCubicSpline[T](X: seq[float], Y: seq[T]): seq[array[5, float]] =
  let n = X.len - 1
  var a = newSeq[T](n+1)
  var b = newSeq[float](n)
  var d = newSeq[float](n)
  var h = newSeq[float](n)
  for i in 0 ..< n:
    a[i] = Y[i]
    h[i] = X[i+1] - X[i]
  a[n] = Y[n]
  var alpha = newSeq[T](n)
  for i in 1 ..< n:
    alpha[i] = 3.0 / h[i] * (a[i+1] - a[i]) - 3.0 / h[i-1] * (a[i] - a[i-1])
  var c = newSeq[T](n+1)
  var mu = newSeq[float](n+1)
  var l = newSeq[float](n+1)
  var z = newSeq[T](n+1)
  l[0] = 1.0
  mu[0] = 0.0
  z[0] = 0.0
  for i in 1 ..< n:
    l[i] = 2.0 * (X[i+1] - X[i-1]) - h[i-1]*mu[i-1]
    mu[i] = h[i] / l[i]
    z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
  l[n] = 1.0
  z[n] = 0.0
  c[n] = 0.0
  for j in countdown(n-1, 0):
    c[j] = z[j] - mu[j]*c[j+1]
    b[j] = (a[j+1]-a[j])/h[j] - h[j] * (c[j+1] + 2.0*c[j]) / 3.0
    d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
  result = newSeq[array[5, float]](n)
  for i in 0 ..< n:
    result[i] = [a[i], b[i], c[i], d[i], X[i]]
    

proc newCubicSpline*[T](X: seq[float], Y: seq[T]): CubicSpline[T] =
  let sortedData = sortDataset(X, Y)
  var xSorted = newSeq[float](X.len)
  var ySorted = newSeq[T](Y.len)
  for i in 0 .. sortedData.high:
    xSorted[i] = sortedData[i][0]
    ySorted[i] = sortedData[i][1]
  let coeffs = constructCubicSpline(xSorted, ySorted)
  result = CubicSpline[T](X: xSorted, coeffs: coeffs, high: xSorted.high, len: xSorted.len)

proc eval*[T](spline: CubicSpline[T], x: float): T =
  if x < spline.X[0] or spline.X[spline.high] < x:
    raise newException(ValueError, &"x = {x} isn't in the interval [{spline.X[0]}, {spline.X[spline.high]}]")
  var upper = spline.high
  var lower = 0
  var n = floorDiv(upper + lower, 2)
  # find interval using binary search
  for i in 0 .. spline.high:
    if x < spline.X[n]:
      # x is below current interval
      upper = n
      n = floorDiv(upper + lower, 2)
      continue
    if spline.X[n+1] < x:
      # x is above current interval
      lower = n + 1
      n = floorDiv(upper + lower, 2)
      continue
    # x is in the interval
    let a = spline.coeffs[n][0]
    let b = spline.coeffs[n][1]
    let c = spline.coeffs[n][2]
    let d = spline.coeffs[n][3]
    let xj = spline.coeffs[n][4]
    let xDiff = x - xj
    return a + b * xDiff + c * xDiff * xDiff + d * xDiff * xDiff * xDiff

proc eval*[T](spline: CubicSpline[T], x: openArray[float]): seq[T] =
  result = newSeq[T](x.len)
  for i, xi in x:
    result[i] = eval(spline, xi)

converter toProc*[T](spline: CubicSpline[T]): proc(x: float): T =
  result = proc(t: float): T = eval(spline, t)

converter toOptionalProc*[T](spline: CubicSpline[T]): proc(x: float, optional: seq[T] = @[]): T =
  result = proc(t: float, optional: seq[T] = @[]): T = eval(spline, t)

proc derivEval*[T](spline: CubicSpline[T], x: float): T =
  if x < spline.X[0] or spline.X[spline.high] < x:
    raise newException(ValueError, &"x = {x} isn't in the interval [{spline.X[0]}, {spline.X[spline.high]}]")
  var upper = spline.high
  var lower = 0
  var n = floorDiv(upper + lower, 2)
  # find interval using binary search
  for i in 0 .. spline.high:
    if x < spline.X[n]:
      # x is below current interval
      upper = n
      n = floorDiv(upper + lower, 2)
      continue
    if spline.X[n+1] < x:
      # x is above current interval
      lower = n + 1
      n = floorDiv(upper + lower, 2)
      continue
    # x is in the interval
    let b = spline.coeffs[n][1]
    let c = spline.coeffs[n][2]
    let d = spline.coeffs[n][3]
    let xj = spline.coeffs[n][4]
    let xDiff = x - xj
    return b + 2 * c * xDiff + 3 * d * xDiff * xDiff

proc derivEval*[T](spline: CubicSpline[T], x: openArray[float]): seq[T] =
  result = newSeq[T](x.len)
  for i, xi in x:
    result[i] = derivEval(spline, xi)

proc toDerivProc*[T](spline: CubicSpline[T]): proc(x: float): T =
  result = proc(t: float): T = derivEval(spline, t)

proc toDerivOptionalProc*[T](spline: CubicSpline[T]): proc(x: float, optional: seq[T] = @[]): T =
  result = proc(t: float, optional: seq[T] = @[]): T = derivEval(spline, t)

#[
import sequtils, math
proc f(x: float): float = exp(x)
let x = linspace(0.0, 10.0, 100)
let y: seq[float] = x.map(f)
let spline = newCubicSpline(x, y)
var splineProc2 = spline.toProc
let splineProc3 = spline.toOptionalProc
var t = 4.875586
echo spline.eval(t), "eval"
echo exp(t), "exp"
echo splineProc2(t)
echo exp(t) - spline.eval(t)
let n = 10000
timeit(spline.eval(t), n, "Spline")
timeit(splineProc2(t), n, "SplineProc2")
timeit(exp(t), n, "Exp")
echo adaptiveGauss(splineProc3, 0.0, 1.0)
echo adaptiveGauss(spline, 0.0, 1.0)
echo exp(1.0) - 1
#echo spline.eval(x), "spline"
#echo spline.binary_eval(x), "binarySeach"
#echo y, "exp"
timeit(spline.eval(x), n, "Spline")
let deriv = spline.toDerivProc
echo deriv(t)
]#