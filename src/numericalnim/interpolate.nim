import strformat, math
import utils

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

proc constructCubicSpline[T](X: openArray[float], Y: openArray[T]): seq[array[5, float]] =
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
    

proc newCubicSpline*[T: SomeFloat](X: openArray[float], Y: openArray[T]): CubicSpline[T] =
  let sortedData = sortDataset(X, Y)
  var xSorted = newSeq[float](X.len)
  var ySorted = newSeq[T](Y.len)
  for i in 0 .. sortedData.high:
    xSorted[i] = sortedData[i][0]
    ySorted[i] = sortedData[i][1]
  let coeffs = constructCubicSpline(xSorted, ySorted)
  result = CubicSpline[T](X: xSorted, coeffs: coeffs, high: xSorted.high, len: xSorted.len)

proc findInterval*(list: openArray[float], x: float): int {.inline.} =
  ## Finds the index of the element to the left of x in list using binary search. list must be ordered.
  let highIndex = list.high
  if x < list[0] or list[highIndex] < x:
    raise newException(ValueError, &"x = {x} isn't in the interval [{list[0]}, {list[highIndex]}]")
  var upper = highIndex
  var lower = 0
  var n = floorDiv(upper + lower, 2)
  # find interval using binary search
  for i in 0 .. highIndex:
    if x < list[n]:
      # x is below current interval
      upper = n
      n = floorDiv(upper + lower, 2)
      continue
    if list[n+1] < x:
      # x is above current interval
      lower = n + 1
      n = floorDiv(upper + lower, 2)
      continue
    # x is in the interval
    return n

proc eval*[T](spline: CubicSpline[T], x: float): T =
  #[if x < spline.X[0] or spline.X[spline.high] < x:
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
    # x is in the interval]#
  let n = findInterval(spline.X, x)
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
  let n = findInterval(spline.X, x)
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
