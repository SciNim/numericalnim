import std/[strformat, math, tables, algorithm]
import arraymancer, cdt/[dt, vectors, edges, types]
import
  ./utils,
  ./common/commonTypes,
  ./private/arraymancerOverloads,
  ./rbf

export rbf

## # Interpolation
## This module implements various interpolation routines.
## See also:
## - `rbf module<rbf.html>`_ for RBF interpolation of scattered data in arbitrary dimensions.
##
## ## 1D interpolation
## - Hermite spline (recommended): cubic spline that works with many types of values. Accepts derivatives if available.
## - Cubic spline: cubic spline that only works with `float`s.
## - Linear spline: Linear spline that works with many types of values.
##
## ### Extrapolation
## Extrapolation is supported for all 1D interpolators by passing the type of extrapolation as an argument of `eval`.
## The default is to use the interpolator's native method to extrapolate. This means that Linear does linear extrapolation,
## while Hermite and Cubic performs cubic extrapolation. Other options are using a constant value, using the values of the edges,
## linear extrapolation and raising an error if the x-value is outside the domain.

runnableExamples:
  import numericalnim, std/[math, sequtils]

  let x = linspace(0.0, 1.0, 10)
  let y = x.mapIt(sin(it))

  let interp = newHermiteSpline(x, y)

  let val = interp.eval(0.314)

  let valExtrapolate = interp.eval(1.1, Edge)

## ## 2D interpolation
## - Bicubic: Works on gridded data.
## - Bilinear: Works on gridded data.
## - Nearest neighbour: Works on gridded data.
## - Barycentric: Works on scattered data.

runnableExamples:
  import arraymancer, numericalnim

  var z = newTensor[float](10, 10)
  for i, x in linspace(0.0, 1.0, 10):
    for j, y in linspace(0.0, 1.0, 10):
      z[i, j] = x * y

  let xlim = (0.0, 1.0)
  let ylim = (0.0, 1.0)
  let interp = newBicubicSpline(z, xlim, ylim)

  let val = interp.eval(0.314, 0.628)

## ## 3D interpolation
## - Trilinear: works on gridded data

runnableExamples:
  import arraymancer, numericalnim

  var f = newTensor[float](10, 10, 10)
  for i, x in linspace(0.0, 1.0, 10):
    for j, y in linspace(0.0, 1.0, 10):
      for k, z in linspace(0.0, 1.0, 10):
        f[i, j, k] = x * y * z

  let xlim = (0.0, 1.0)
  let ylim = (0.0, 1.0)
  let zlim = (0.0, 1.0)
  let interp = newTrilinearSpline(f, xlim, ylim, zlim)

  let val = interp.eval(0.314, 0.628, 0.414)

type
  InterpolatorType*[T] = ref object
    X*: seq[float]
    Y*: seq[T]
    coeffs_f*: Tensor[float]
    coeffs_T*: Tensor[T]
    high*: int
    len*: int
    eval_handler*: EvalHandler[T]
    deriveval_handler*: EvalHandler[T]
  EvalHandler*[T] = proc(self: InterpolatorType[T], x: float): T {.nimcall.}
  ExtrapolateKind* = enum
    Constant, Edge, Linear, Native, Error
  Eval2DHandler*[T] = proc (self: Interpolator2DType[T], x, y: float): T {.nimcall.}
  Interpolator2DType*[T] = ref object
    z*, xGrad*, yGrad*, xyGrad*: Tensor[T] # 2D tensor
    alphaCache*: Table[(int, int), Tensor[T]]
    dx*, dy*: float
    xLim*, yLim*: tuple[lower: float, upper: float]
    eval_handler*: Eval2DHandler[T]
  EvalUnstructured2DHandler*[T, U] = proc (self: InterpolatorUnstructured2DType[T, U], x, y: float): U {.nimcall.}
  InterpolatorUnstructured2DType*[T: SomeFloat, U] = ref object
    values*: Tensor[T]
    points*: Tensor[U]
    dt*: DelaunayTriangulation
    z*, gradX*, gradY*: Table[(T, T), U]
    boundPoints*: array[4, Vector2]
    eval_handler*: EvalUnstructured2DHandler[T, U]
  Eval3DHandler*[T] = proc (self: Interpolator3DType[T], x, y, z: float): T {.nimcall.}
  Interpolator3DType*[T] = ref object
    f*: Tensor[T] # 3D tensor
    dx*, dy*, dz*: float
    xLim*, yLim*, zLim*: tuple[lower: float, upper: float]
    eval_handler*: Eval3DHandler[T]


proc findInterval*(list: openArray[float], x: float): int {.inline.} =
  result = clamp(lowerbound(list, x) - 1, 0, list.high-1)

# CubicSpline

proc constructCubicSpline[T](X: openArray[float], Y: openArray[T]): Tensor[float] =
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
  result = newTensorUninit[T](n, 5)
  for i in 0 ..< n:
    result[i, _] = @[a[i], b[i], c[i], d[i], X[i]].toTensor.reshape(1, 5)


proc eval_cubicspline*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let a = spline.coeffs_T[n, 0]
  let b = spline.coeffs_T[n, 1]
  let c = spline.coeffs_T[n, 2]
  let d = spline.coeffs_T[n, 3]
  let xj = spline.coeffs_T[n, 4]
  let xDiff = x - xj
  return a + b * xDiff + c * xDiff * xDiff + d * xDiff * xDiff * xDiff

proc derivEval_cubicspline*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let b = spline.coeffs_T[n, 1]
  let c = spline.coeffs_T[n, 2]
  let d = spline.coeffs_T[n, 3]
  let xj = spline.coeffs_T[n, 4]
  let xDiff = x - xj
  return b + 2 * c * xDiff + 3 * d * xDiff * xDiff

proc newCubicSpline*[T: SomeFloat](X: openArray[float], Y: openArray[
    T]): InterpolatorType[T] =
  ## Returns a cubic spline.
  let (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
  let coeffs = constructCubicSpline(xSorted, ySorted)
  result = InterpolatorType[T](X: xSorted, Y: ySorted, coeffs_T: coeffs, high: xSorted.high,
      len: xSorted.len, eval_handler: eval_cubicspline,
      deriveval_handler: derivEval_cubicspline)


# HermiteSpline

proc eval_hermitespline*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let xDiff = spline.X[n+1] - spline.X[n]
  let t = (x - spline.X[n]) / xDiff
  let t2 = t * t
  let t3 = t2 * t
  let h00 = 2*t3 - 3*t2 + 1
  let h10 = t3 - 2*t2 + t
  let h01 = -2*t3 + 3*t2
  let h11 = t3 - t2
  let p1 = spline.coeffs_T[n, 0]
  let p2 = spline.coeffs_T[n+1, 0]
  let m1 = spline.coeffs_T[n, 1]
  let m2 = spline.coeffs_T[n+1, 1]
  result = h00*p1 + h10*xDiff*m1 + h01*p2 + h11*xDiff*m2

proc derivEval_hermitespline*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let xDiff = spline.X[n+1] - spline.X[n]
  let t = (x - spline.X[n]) / xDiff
  let t2 = t * t
  let h00 = 6*t2 - 6*t
  let h10 = 3*t2 - 4*t + 1
  let h01 = -6*t2 + 6*t
  let h11 = 3*t2 - 2*t
  let p1 = spline.coeffs_T[n, 0]
  let p2 = spline.coeffs_T[n+1, 0]
  let m1 = spline.coeffs_T[n, 1]
  let m2 = spline.coeffs_T[n+1, 1]
  result = (h00*p1 + h10*xDiff*m1 + h01*p2 + h11*xDiff*m2) / xDiff

proc newHermiteSpline*[T](X: openArray[float], Y, dY: openArray[
    T]): InterpolatorType[T] =
  ## Constructs a cubic Hermite spline using x, y and derivatives of y.
  #let sortedData = sortDataset(X, Y)
  #let sortedData_dY = sortDataset(X, dY)
  #var xSorted = newSeq[float](X.len)
  #var ySorted = newSeq[T](Y.len)
  #var dySorted = newSeq[T](dY.len)
  #for i in 0 .. sortedData.high:
  #  xSorted[i] = sortedData[i][0]
  #  ySorted[i] = sortedData[i][1]
  #  dySorted[i] = sortedData_dY[i][1]
  if X.len != Y.len or X.len != dY.len:
    raise newException(ValueError, &"X and Y and dY must have the same length. X.len is {X.len} and Y.len is {Y.len} and dY is {dY.len}")
  let sortedDataset = sortAndTrimDataset(@X, @[@Y, @dY])
  let xSorted = sortedDataset.x
  let ySorted = sortedDataset.y[0]
  let dySorted = sortedDataset.y[1]
  var coeffs = newTensorUninit[T](ySorted.len, 2)
  for i in 0 .. ySorted.high:
    coeffs[i, _] = @[ySorted[i], dySorted[i]].toTensor.reshape(1, 2)
  result = InterpolatorType[T](X: xSorted, Y: ySorted, coeffs_T: coeffs, high: xSorted.high,
      len: xSorted.len, eval_handler: eval_hermitespline,
      deriveval_handler: derivEval_hermitespline)

proc newHermiteSpline*[T](X: openArray[float], Y: openArray[
    T]): InterpolatorType[T] =
  ## Constructs a cubic Hermite spline by approximating the derivatives.
  # if only (x, y) is given, use three-point difference to calculate dY.
  let (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
  var dySorted = newSeq[T](ySorted.len)
  let highest = dySorted.high
  dySorted[0] = (ySorted[1] - ySorted[0]) / (xSorted[1] - xSorted[0])
  dySorted[highest] = (ySorted[highest] - ySorted[highest-1]) / (xSorted[
      highest] - xSorted[highest-1])
  for i in 1 .. highest-1:
    dySorted[i] = 0.5 * ((ySorted[i+1] - ySorted[i])/(xSorted[i+1] - xSorted[
        i]) + (ySorted[i] - ySorted[i-1])/(xSorted[i] - xSorted[i-1]))
  var coeffs = newTensorUninit[T](Y.len, 2)
  for i in 0 .. ySorted.high:
    coeffs[i, _] = @[ySorted[i], dySorted[i]].toTensor.reshape(1, 2)
  result = InterpolatorType[T](X: xSorted, Y: ySorted, coeffs_T: coeffs, high: xSorted.high,
      len: xSorted.len, eval_handler: eval_hermitespline,
      deriveval_handler: derivEval_hermitespline)


# Linear1D

proc eval_linear1d*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let xDiff = spline.X[n+1] - spline.X[n]
  let yDiff = spline.coeffs_T[n+1, 0] - spline.coeffs_T[n, 0]
  let k = yDiff / xDiff
  result = spline.coeffs_T[n, 0] + k * (x - spline.X[n])

proc derivEval_linear1d*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let xDiff = spline.X[n+1] - spline.X[n]
  let yDiff = spline.coeffs_T[n+1, 0] - spline.coeffs_T[n, 0]
  let k = yDiff / xDiff
  result = k

proc newLinear1D*[T](X: openArray[float], Y: openArray[
    T]): InterpolatorType[T] =
  ## Constructs a linear interpolator.
  if X.len != Y.len:
    raise newException(ValueError, &"X and Y and dY must have the same length. X.len is {X.len} and Y.len is {Y.len}")
  let sortedDataset = sortAndTrimDataset(@X, @Y)
  let xSorted = sortedDataset.x
  let ySorted: seq[T] = sortedDataset.y
  var coeffs = newTensor[T](ySorted.len, 1)
  for i in 0 .. ySorted.high:
    coeffs[i, 0] = ySorted[i]
  result = InterpolatorType[T](X: xSorted, Y: ySorted, coeffs_T: coeffs, high: xSorted.high,
      len: xSorted.len, eval_handler: eval_linear1d,
      deriveval_handler: derivEval_linear1d)

# General Spline stuff

type Missing = object
proc missing(): Missing = discard

proc eval*[T; U](interpolator: InterpolatorType[T], x: float, extrap: ExtrapolateKind = Native, extrapValue: U = missing()): T =
  ## Evaluates an interpolator.
  ## - `x`: The point to evaluate the interpolator at.
  ## - `extrap`: The extrapolation method to use. Available options are:
  ##  - `Constant`: Set all points outside the range of the interpolator to `extrapValue`.
  ##  - `Edge`: Use the value of the left/right edge.
  ##  - `Linear`: Uses linear extrapolation using the two points closest to the edge.
  ##  - `Native` (default): Uses the native method of the interpolator to extrapolate. For Linear1D it will be a linear extrapolation, and for Cubic and Hermite splines it will be cubic extrapolation.
  ##  - `Error`: Raises an `ValueError` if `x` is outside the range.
  ## - `extrapValue`: The extrapolation value to use when `extrap = Constant`.
  ##
  ## > Beware: `Native` extrapolation for the cubic splines can very quickly diverge if the extrapolation is too far away from the interpolation points.
  when U is Missing:
    assert extrap != Constant, "When using `extrap = Constant`, a value `extrapValue` must be supplied!"
  else:
    when not T is U:
      {.error: &"Type of `extrap` ({U}) is not the same as the type of the interpolator ({T})!".}

  let xLeft = x < interpolator.X[0]
  let xRight = x > interpolator.X[^1]
  if xLeft or xRight:
    case extrap
    of Constant:
      when U is Missing:
        discard
      else:
        return extrapValue
    of Native:
      discard
    of Edge:
      return
        if xLeft: interpolator.Y[0]
        else: interpolator.Y[^1]
    of Linear:
      let (xs, ys) =
        if xLeft:
          ((interpolator.X[0], interpolator.X[1]), (interpolator.Y[0], interpolator.Y[1]))
        else:
          ((interpolator.X[^2], interpolator.X[^1]), (interpolator.Y[^2], interpolator.Y[^1]))
      let k = (x - xs[0]) / (xs[1] - xs[0])
      return ys[0] + k * (ys[1] - ys[0])
    of Error:
      raise newException(ValueError, &"x = {x} isn't in the interval [{interpolator.X[0]}, {interpolator.X[^1]}]")

  result = interpolator.eval_handler(interpolator, x)


proc derivEval*[T; U](interpolator: InterpolatorType[T], x: float, extrap: ExtrapolateKind = Native, extrapValue: U = missing()): T =
  ## Evaluates the derivative of an interpolator.
  ## - `x`: The point to evaluate the interpolator at.
  ## - `extrap`: The extrapolation method to use. Available options are:
  ##  - `Constant`: Set all points outside the range of the interpolator to `extrapValue`.
  ##  - `Edge`: Use the value of the left/right edge.
  ##  - `Linear`: Uses linear extrapolation using the two points closest to the edge.
  ##  - `Native` (default): Uses the native method of the interpolator to extrapolate. For Linear1D it will be a linear extrapolation, and for Cubic and Hermite splines it will be cubic extrapolation.
  ##  - `Error`: Raises an `ValueError` if `x` is outside the range.
  ## - `extrapValue`: The extrapolation value to use when `extrap = Constant`.
  ##
  ## > Beware: `Native` extrapolation for the cubic splines can very quickly diverge if the extrapolation is too far away from the interpolation points.
  when U is Missing:
    assert extrap != Constant, "When using `extrap = Constant`, a value `extrapValue` must be supplied!"
  else:
    when not T is U:
      {.error: &"Type of `extrap` ({U}) is not the same as the type of the interpolator ({T})!".}

  let xLeft = x < interpolator.X[0]
  let xRight = x > interpolator.X[^1]
  if xLeft or xRight:
    case extrap
    of Constant:
      when U is Missing:
        discard
      else:
        return extrapValue
    of Native:
      discard
    of Edge:
      return
        if xLeft: interpolator.deriveval_handler(interpolator, interpolator.X[0])
        else: interpolator.deriveval_handler(interpolator, interpolator.X[^1])
    of Linear:
      let (xs, ys) =
        if xLeft:
          ((interpolator.X[0], interpolator.X[1]), (interpolator.deriveval_handler(interpolator, interpolator.X[0]), interpolator.deriveval_handler(interpolator, interpolator.X[1])))
        else:
          ((interpolator.X[^2], interpolator.X[^1]), (interpolator.deriveval_handler(interpolator, interpolator.X[^2]), interpolator.deriveval_handler(interpolator, interpolator.X[^1])))
      let k = (x - xs[0]) / (xs[1] - xs[0])
      return ys[0] + k * (ys[1] - ys[0])
    of Error:
      raise newException(ValueError, &"x = {x} isn't in the interval [{interpolator.X[0]}, {interpolator.X[^1]}]")

  result = interpolator.deriveval_handler(interpolator, x)

proc eval*[T; U](spline: InterpolatorType[T], x: openArray[float], extrap: ExtrapolateKind = Native, extrapValue: U = missing()): seq[T] =
  ## Evaluates an interpolator at all points in `x`.
  result = newSeq[T](x.len)
  for i, xi in x:
    result[i] = eval(spline, xi, extrap, extrapValue)

proc toProc*[T](spline: InterpolatorType[T]): InterpolatorProc[T] =
  ## Returns a proc to evaluate the interpolator.
  result = proc(x: float): T = eval(spline, x)

proc derivEval*[T; U](spline: InterpolatorType[T], x: openArray[float], extrap: ExtrapolateKind = Native, extrapValue: U = missing()): seq[T] =
  ## Evaluates the derivative of an interpolator at all points in `x`.
  result = newSeq[T](x.len)
  for i, xi in x:
    result[i] = derivEval(spline, xi, extrap, extrapValue)

proc toDerivProc*[T](spline: InterpolatorType[T]): InterpolatorProc[T] =
  ## Returns a proc to evaluate the derivative of the interpolator.
  result = proc(x: float): T = derivEval(spline, x)

proc toDerivNumContextProc*[T](spline: InterpolatorType[T]): NumContextProc[T, float] =
  ## Convert interpolator's derivative to `NumContextProc`.
  result = proc(x: float, ctx: NumContext[T, float]): T = derivEval(spline, x)


# ############################################ #
# ########### 2D Interpolation ############### #
# ############################################ #

# Nearest Neighbour interpolation

proc checkInterpolationInterval[T](self: Interpolator2DType[T], x, y: float) =
  let raiseX = not(self.xLim.lower <= x and x <= self.xLim.upper)
  let raiseY = not(self.yLim.lower <= y and y <= self.yLim.upper)

  if raiseX and raiseY:
    var raiseMsg = &"x={x} and y={y} respectively not in grid intervals [{self.xLim.lower}, {self.xLim.upper}] and [{self.yLim.lower}, {self.yLim.upper}]."
    raise newException(ValueError, raiseMsg)

  if raiseX:
    var raiseMsg = &"x={x} not in grid interval [{self.xLim.lower}, {self.xLim.upper}]."
    raise newException(ValueError, raiseMsg)

  if raiseY:
    var raiseMsg = &"y={y} not in grid interval [{self.yLim.lower}, {self.yLim.upper}]."
    raise newException(ValueError, raiseMsg)

proc eval_nearestneigh*[T](self: Interpolator2DType[T], x, y: float): T {.nimcall.} =
  when compileOption("boundChecks"):
    checkInterpolationInterval(self, x, y)
  let i = round((x - self.xLim.lower) / self.dx).toInt
  let j = round((y - self.yLim.lower) / self.dy).toInt
  result = self.z[i, j]

proc newNearestNeighbour2D*[T](z: Tensor[T], xlim, ylim: (float, float)): Interpolator2DType[T] =
  ## Returns a nearest neighbour interpolator for regularly gridded data.
  ## - z: Tensor with the function values. x corrensponds to the rows and y to the columns. Must be sorted so ascendingly in both variables.
  ## - xlim: the lowest and highest x-value
  ## - ylim: the lowest and highest y-value
  assert z.rank == 2, "z must be a 2D tensor"
  new result
  let nx = z.shape[0]
  let ny = z.shape[1]
  let xStart = xlim[0]
  let xEnd = xlim[1]
  let yStart = ylim[0]
  let yEnd = ylim[1]
  let dx = (xEnd - xStart) / (nx - 1).toFloat
  let dy = (yEnd - yStart) / (ny - 1).toFloat
  result.z = z
  result.dx = dx
  result.dy = dy
  result.xLim = (lower: xStart, upper: xEnd)
  result.yLim = (lower: yStart, upper: yEnd)
  result.eval_handler = eval_nearestneigh[T]

# Bilinear interpolation

proc eval_bilinear*[T](self: Interpolator2DType[T], x, y: float): T {.nimcall.} =
  # find interval
  when compileOption("boundChecks"):
    checkInterpolationInterval(self, x, y)
  let i = min(floor((x - self.xLim.lower) / self.dx).toInt, self.z.shape[0] - 2)
  let j = min(floor((y - self.yLim.lower) / self.dy).toInt, self.z.shape[1] - 2)
  # transform x and y to unit square
  let xCorner = self.xLim.lower + i.toFloat * self.dx
  let x = (x - xCorner) / self.dx
  let yCorner = self.yLim.lower + j.toFloat * self.dy
  let y = (y - yCorner) / self.dy
  let f00 = self.z[i, j]
  let f10 = self.z[i+1, j]
  let f01 = self.z[i, j+1]
  let f11 = self.z[i+1, j+1]
  result = f00 * (1-x) * (1-y) + f10 * x * (1-y) + f01 * (1-x) * y + f11 * x * y

proc newBilinearSpline*[T](z: Tensor[T], xlim, ylim: (float, float)): Interpolator2DType[T] =
  ## Returns a bilinear spline for regularly gridded data.
  ## - z: Tensor with the function values. x corrensponds to the rows and y to the columns. Must be sorted so ascendingly in both variables.
  ## - xlim: the lowest and highest x-value
  ## - ylim: the lowest and highest y-value
  assert z.rank == 2, "z must be a 2D tensor"
  new result
  let nx = z.shape[0]
  let ny = z.shape[1]
  let xStart = xlim[0]
  let xEnd = xlim[1]
  let yStart = ylim[0]
  let yEnd = ylim[1]
  let dx = (xEnd - xStart) / (nx - 1).toFloat
  let dy = (yEnd - yStart) / (ny - 1).toFloat
  result.z = z
  result.dx = dx
  result.dy = dy
  result.xLim = (lower: xStart, upper: xEnd)
  result.yLim = (lower: yStart, upper: yEnd)
  result.eval_handler = eval_bilinear[T]

# Bicubic interpolator (Credits to @Vindaar for main calculations here)

proc grad[T](tIn: Tensor[T], xdiff: float = 1.0): Tensor[T] =
  let t = tIn.squeeze
  doAssert t.rank == 1
  result = newTensor[T](t.size.int)
  for i in 0 ..< t.size:
    if i == 0:
      result[i] = (t[1] - t[0]) / xdiff
    elif i == t.size - 1:
      result[i] = (t[i] - t[i - 1]) / xdiff
    else:
      result[i] = (t[i + 1] - t[i - 1]) / (2 * xdiff)

proc grad[T](t: Tensor[T], axis: int): Tensor[T] =
  ## given 2D tensor, calc gradient in axis direction
  result = newTensor[T](t.shape)
  for idx, ax in enumerateAxis(t, axis):
    ## TODO: eh, how to do this in one?
    case axis
    of 0: result[idx, _] = grad(ax, 1.0).unsqueeze(0)
    of 1: result[_, idx] = grad(ax, 1.0).unsqueeze(1)
    else: doAssert false

proc bicubicGrad[T](z: Tensor[T], dx, dy: float): tuple[xGrad: Tensor[T], yGrad: Tensor[T],
    xyGrad: Tensor[T]] =
  result.xGrad = grad(z, 1)
  result.yGrad = grad(z, 0)
  result.xyGrad = grad(result.yGrad, 1)

proc computeAlpha[T](interp: Interpolator2DType[T],
                  x, y: int): Tensor[T] =
  if (x, y) in interp.alphaCache:
    result = interp.alphaCache[(x, y)]
    return
  let f = interp.z
  let fx = interp.xGrad
  let fy = interp.yGrad
  let fxy = interp.xyGrad
  var m1 = [[1.0, 0, 0, 0],
            [0.0, 0, 1, 0],
            [-3.0, 3, -2, -1],
            [2.0, -2, 1, 1]].toTensor
  var m2 = [[1.0, 0, -3, 2],
            [0.0, 0, 3, -2],
            [0.0, 1, -2, 1],
            [0.0, 0, -1, 1]].toTensor
  var A = [[f[x, y],  f[x, y + 1],  fy[x, y],  fy[x, y+1]],
           [f[x + 1, y],  f[x + 1, y + 1],  fy[x+1, y],  fy[x+1, y+1]],
           [fx[x, y], fx[x, y+1], fxy[x, y], fxy[x, y+1]],
           [fx[x+1, y], fx[x+1, y+1], fxy[x+1, y], fxy[x+1, y+1]]].toTensor()
  when T is SomeNumber and T isnot float:
    result = m1.asType(T) * A * m2.asType(T)
  else:
    result = m1 * A * m2
  interp.alphaCache[(x, y)] = result

proc eval_bicubic*[T](self: Interpolator2DType[T], x, y: float): T {.nimcall.} =
  # find interval
  when compileOption("boundChecks"):
    checkInterpolationInterval(self, x, y)
  let i = min(floor((x - self.xLim.lower) / self.dx).toInt, self.z.shape[0] - 2)
  let j = min(floor((y - self.yLim.lower) / self.dy).toInt, self.z.shape[1] - 2)
  # transform x and y to unit square
  let xCorner = self.xLim.lower + i.toFloat * self.dx
  let x = (x - xCorner) / self.dx
  let yCorner = self.yLim.lower + j.toFloat * self.dy
  let y = (y - yCorner) / self.dy
  let alpha = computeAlpha(self, i, j)
  when T is SomeNumber and T isnot float:
    result = dot([1.0, x, x*x, x*x*x].toTensor.asType(T), alpha * [1.0, y, y*y, y*y*y].toTensor.asType(T))
  elif T is float:
    result = dot([1.0, x, x*x, x*x*x].toTensor, alpha * [1.0, y, y*y, y*y*y].toTensor)
  else:
    result = dot([1.0, x, x*x, x*x*x].toTensor, alpha * [1.0, y, y*y, y*y*y].toTensor.reshape(4, 1))

proc newBicubicSpline*[T](z: Tensor[T], xlim, ylim: (float, float)): Interpolator2DType[T] =
  ## Returns a bicubic spline for regularly gridded data.
  ## - z: Tensor with the function values. x corrensponds to the rows and y to the columns. Must be sorted so ascendingly in both variables.
  ## - xlim: the lowest and highest x-value
  ## - ylim: the lowest and highest y-value
  assert z.rank == 2, "z must be a 2D tensor"
  new result
  let nx = z.shape[0]
  let ny = z.shape[1]
  let xStart = xlim[0]
  let xEnd = xlim[1]
  let yStart = ylim[0]
  let yEnd = ylim[1]
  let dx = (xEnd - xStart) / (nx - 1).toFloat
  let dy = (yEnd - yStart) / (ny - 1).toFloat
  let grads = bicubicGrad(z, dx, dy)
  result.xGrad = grads.xGrad
  result.yGrad = grads.yGrad
  result.xyGrad = grads.xyGrad
  result.z = z
  result.dx = dx
  result.dy = dy
  result.xLim = (lower: xStart, upper: xEnd)
  result.yLim = (lower: yStart, upper: yEnd)
  result.eval_handler = eval_bicubic[T]


# Barycentric Interpolator 2D

proc eval_barycentric2d*[T, U](self: InterpolatorUnstructured2DType[T, U]; x, y: float): U =
  let p = Vector2(x: x, y: y)
  let (edge, loc) = self.dt.locatePoint(p)
  case loc
  of lpFace:
    let point1 = edge.org.point
    let point2 = edge.dest.point
    let point3 = edge.oNext.dest.point
    assert not (point1 in self.boundPoints or point2 in self.boundPoints or point3 in self.boundPoints), "Point outside domain"
    let denom = (point2.y - point3.y) * (point1.x - point3.x) + (point3.x - point2.x) * (point1.y - point3.y)
    let w1 = ((point2.y - point3.y)*(p.x - point3.x) + (point3.x - point2.x)*(p.y - point3.y)) / denom
    let w2 = ((point3.y - point1.y)*(p.x - point3.x) + (point1.x - point3.x)*(p.y - point3.y)) / denom
    let w3 = 1 - w1 - w2
    let z1 = self.z[(point1.x, point1.y)]
    let z2 = self.z[(point2.x, point2.y)]
    let z3 = self.z[(point3.x, point3.y)]
    result = w1*z1 + w2*z2 + w3*z3
  of lpEdge:
    let point1 = edge.org.point
    let point2 = edge.dest.point
    assert not (point1 in self.boundPoints or point2 in self.boundPoints), "Point outside domain"
    var t: T
    if point1.x == point2.x:
      t = (y - point1.y) / (point2.y - point1.y)
    else:
      t = (x - point1.x) / (point2.x - point1.x)
    let z1 = self.z[(point1.x, point1.y)]
    let z2 = self.z[(point2.x, point2.y)]
    result = z1 + (z2 - z1)*t
  of lpOrg:
    let point1 = edge.org.point
    assert point1 notin self.boundPoints, "Point outside domain"
    result = self.z[(x: point1.x, y: point1.y)]
  of lpDest:
    let point1 = edge.dest.point
    assert point1 notin self.boundPoints, "Point outside domain"
    result = self.z[(x: point1.x, y: point1.y)]

proc newBarycentric2D*[T: SomeFloat, U](points: Tensor[T], values: Tensor[U]): InterpolatorUnstructured2DType[T, U] =
  ## Barycentric interpolation of scattered points in 2D.
  ##
  ## Inputs:
  ##  - points: Tensor of shape (nPoints, 2) with the coordinates of all points.
  ##  - values: Tensor of shape (nPoints) with the function values.
  ##
  ## Returns:
  ##  - Interpolator object that can be evaluated using `interp.eval(x, y`.
  assert points.rank == 2 and points.shape[1] == 2
  assert values.rank == 1
  assert values.shape[0] == points.shape[0]
  let x = points[_, 0].squeeze
  let y = points[_, 1].squeeze
  new result
  result.dt = initDelaunayTriangulation(Vector2(x: min(x)-0.1, y: min(y)-0.1), Vector2(x: max(x)+0.1, y: max(y)+0.1))
  result.boundPoints = [
    Vector2(x: min(x)-0.1, y: max(y)+0.1),
    Vector2(x: min(x)-0.1, y: min(y)-0.1),
    Vector2(x: max(x)+0.1, y: min(y)-0.1),
    Vector2(x: max(x)+0.1, y: max(y)+0.1)
  ]
  for i in 0 .. x.shape[0]-1:
    let coord = (x[i], y[i])
    assert coord notin result.z, &"Point {coord} has appeared twice!"
    result.z[coord] = values[i]
    discard result.dt.insert(Vector2(x: coord[0], y: coord[1]))
  result.eval_handler = eval_barycentric2d[T, U]
  return result

# General Interpolator2D stuff

template eval*[T](interpolator: Interpolator2DType[T], x, y: float): untyped =
  ## Evaluate interpolator.
  interpolator.eval_handler(interpolator, x, y)

template eval*[T, U](interpolator: InterpolatorUnstructured2DType[T, U], x, y: T): untyped =
  ## Evaluate interpolator.
  interpolator.eval_handler(interpolator, x, y)

# #############################################
# ########### 3D Interpolation ################
# #############################################

proc checkInterpolationInterval[T](self: Interpolator3DType[T], x, y, z: float) =
  let raiseX = not(self.xLim.lower <= x and x <= self.xLim.upper)
  let raiseY = not(self.yLim.lower <= y and y <= self.yLim.upper)
  let raiseZ = not(self.zLim.lower <= z and z <= self.zLim.upper)

  if raiseX or raiseY or raiseZ:
    var raiseMsg = "The following inputs was not inside the spline's domain:"
    if raiseX:
      raiseMsg.add &"\nx={x} not in grid interval [{self.xLim.lower}, {self.xLim.upper}]."
    if raiseY:
      raiseMsg.add &"\ny={y} not in grid interval [{self.yLim.lower}, {self.yLim.upper}]."
    if raiseZ:
      raiseMsg.add &"\nz={z} not in grid interval [{self.zLim.lower}, {self.zLim.upper}]."
    raise newException(ValueError, raiseMsg)

proc eval_trilinear*[T](self: Interpolator3DType[T], x, y, z: float): T {.nimcall.} =
  when compileOption("boundChecks"):
    checkInterpolationInterval(self, x, y, z)
  # find interval
  let i = min(floor((x - self.xLim.lower) / self.dx).toInt, self.f.shape[0] - 2)
  let j = min(floor((y - self.yLim.lower) / self.dy).toInt, self.f.shape[1] - 2)
  let k = min(floor((z - self.zLim.lower) / self.dz).toInt, self.f.shape[2] - 2)
  # transform x and y to unit square
  let xCorner = self.xLim.lower + i.toFloat * self.dx
  let x = (x - xCorner) / self.dx
  let yCorner = self.yLim.lower + j.toFloat * self.dy
  let y = (y - yCorner) / self.dy
  let zCorner = self.zLim.lower + k.toFloat * self.dz
  let z = (z - zCorner) / self.dz
  let oneMinusX = 1 - x
  let c00 = self.f[i, j, k] * oneMinusX + self.f[i+1, j, k] * x
  let c01 = self.f[i, j, k+1] * oneMinusX + self.f[i+1, j, k+1] * x
  let c10 = self.f[i, j+1, k] * oneMinusX + self.f[i+1, j+1, k] * x
  let c11 = self.f[i, j+1, k+1] * oneMinusX + self.f[i+1, j+1, k+1] * x
  let oneMinusY = 1 - y
  let c0 = c00 * oneMinusY + c10 * y
  let c1 = c01 * oneMinusY + c11 * y
  result = c0 * (1 - z) + c1 * z

proc newTrilinearSpline*[T](f: Tensor[T], xlim, ylim, zlim: (float, float)): Interpolator3DType[T] =
  ## Returns a trilinear spline for regularly gridded data.
  ## - z: Tensor with the function values. x corrensponds to the first dimension, y to the second and z to the third. Must be sorted so ascendingly in both variables.
  ## - xlim: the lowest and highest x-value
  ## - ylim: the lowest and highest y-value
  ## - zlim: the lowest and highest z-value
  assert f.rank == 3, "f must be a 3D tensor"
  new result
  let nx = f.shape[0]
  let ny = f.shape[1]
  let nz = f.shape[2]
  let xStart = xlim[0]
  let xEnd = xlim[1]
  let yStart = ylim[0]
  let yEnd = ylim[1]
  let zStart = zlim[0]
  let zEnd = zlim[1]
  let dx = (xEnd - xStart) / (nx - 1).toFloat
  let dy = (yEnd - yStart) / (ny - 1).toFloat
  let dz = (zEnd - zStart) / (nz - 1).toFloat
  result.f = f
  result.dx = dx
  result.dy = dy
  result.dz = dz
  result.xLim = (lower: xStart, upper: xEnd)
  result.yLim = (lower: yStart, upper: yEnd)
  result.zLim = (lower: zStart, upper: zEnd)
  result.eval_handler = eval_trilinear[T]

# General Interpolator3D stuff
template eval*[T](interpolator: Interpolator3DType[T], x, y, z: float): untyped =
  ## Evaluate interpolator.
  interpolator.eval_handler(interpolator, x, y, z)

#[
proc plot(interp: Interpolator2DType, name: string) =
  var x, y, z: seq[float]
  for i in linspace(interp.xLim.lower, interp.xLim.upper, 1000):
    for j in linspace(interp.yLim.lower, interp.yLim.upper, 1000):
      z.add interp.eval(i, j)
      x.add i
      y.add j
  let df = seqsToDf(x, y, z)
  ggplot(df, aes("x", "y", fill = "z")) +
    geom_raster() +
    #scale_x_continuous() + scale_y_continuous() +
    scale_fill_continuous(scale = (low: -70.0, high: 70.0)) +
    xlim(interp.xLim.lower, interp.xLim.upper) + ylim(interp.yLim.lower, interp.yLim.upper) +
    theme_opaque() +
    ggsave(name & ".png", width = 900, height = 800)

when isMainModule:
  var z = ones[float](3, 3)
  z[1, _] = z[1, _] +. 1.0
  z[2, _] = z[2, _] +. 2.0
  echo z
  let v = newVector[float]([1.0, 1.0, 1.0])
  var zVector: Tensor[Vector[float]] = [[v, v], [v+1.0, v+1.0], [v+2.0, v+2.0]].toTensor
  echo zVector
  let blVector = newBilinearSpline(zVector, (0.0, 9.0), (0.0, 9.0))
  let bcVector = newBicubicSpline(zVector, (0.0, 9.0), (0.0, 9.0))
  let blspline = newBilinearSpline(z, (0.0, 2.0), (0.0, 2.0))
  let bcspline = newBicubicSpline(z, (0.0, 2.0), (0.0, 2.0))
  let x = 1.8
  let y = 2.0
  echo "Linear: ", blspline.eval(x, y)
  echo "Cubic:  ", bcspline.eval(x, y)
  echo "LinearV:", blVector.eval(x, y)
  echo "CubicV: ", bcVector.eval(x, y)
  let randTensor = randomTensor([10, 10], 75).asType(float)
  let nearestInterp = newNearestNeighbour2D(randTensor, (0.0, 9.0), (0.0, 9.0))
  let linearSpline = newBilinearSpline(randTensor, (0.0, 9.0), (0.0, 9.0))
  let cubicSpline = newBicubicSpline(randTensor, (0.0, 9.0), (0.0, 9.0))
  plot(nearestInterp, "nearest")
  plot(linearSpline, "bilinear")
  plot(cubicSpline, "bicubic")
  import benchy
  let xTest = linspace(0.0, 9.0, 999)
  let yTest = linspace(0.0, 9.0, 999)
  timeIt "Nearest":
    for x in xTest:
      for y in yTest:
        keep(nearestInterp.eval(x, y))
  timeIt "Linear":
    for x in xTest:
      for y in yTest:
        keep(linearSpline.eval(x, y))
  timeIt "Cubic":
    for x in xTest:
      for y in yTest:
        keep(cubicSpline.eval(x, y))
  timeIt "LinearV":
    for x in xTest:
      for y in yTest:
        keep(blVector.eval(x, y))
  timeIt "CubicV":
    for x in xTest:
      for y in yTest:
        keep(bcVector.eval(x, y))
  # Result:
  # Nearest ............................ 14.972 ms    ±0.478  x10
  # Linear ............................. 37.913 ms    ±1.122  x10
  # Cubic ............................. 693.276 ms    ±4.326  x10
]#
