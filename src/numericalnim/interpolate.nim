import strformat, math, tables
import arraymancer, ggplotnim
import
  ./utils,
  ./common/commonTypes

type
  InterpolatorType*[T] = ref object
    X*: seq[float]
    coeffs_f*: seq[seq[float]]
    coeffs_T*: seq[seq[T]]
    high*: int
    len*: int
    eval_handler*: EvalHandler[T]
    deriveval_handler*: EvalHandler[T]
  EvalHandler*[T] = proc(self: InterpolatorType[T], x: float): T {.nimcall.}
  Interpolator2DType*[T] = ref object
    z*, xGrad*, yGrad*, xyGrad*: Tensor[T] # 2D tensor
    alphaCache*: Table[(int, int), Tensor[T]]
    dx*, dy*: float
    xLim*, yLim*: tuple[lower: float, upper: float]
    eval_handler*: proc (self: Interpolator2DType[T], x, y: float): T {.nimcall.}


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

### CubicSpline

proc constructCubicSpline[T](X: openArray[float], Y: openArray[T]): seq[seq[float]] =
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
  result = newSeq[seq[float]](n)
  for i in 0 ..< n:
    result[i] = @[a[i], b[i], c[i], d[i], X[i]]


proc eval_cubicspline*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let a = spline.coeffs_T[n][0]
  let b = spline.coeffs_T[n][1]
  let c = spline.coeffs_T[n][2]
  let d = spline.coeffs_T[n][3]
  let xj = spline.coeffs_T[n][4]
  let xDiff = x - xj
  return a + b * xDiff + c * xDiff * xDiff + d * xDiff * xDiff * xDiff

proc derivEval_cubicspline*[T](spline: InterpolatorType[T], x: float): T =
  let n = findInterval(spline.X, x)
  let b = spline.coeffs_T[n][1]
  let c = spline.coeffs_T[n][2]
  let d = spline.coeffs_T[n][3]
  let xj = spline.coeffs_T[n][4]
  let xDiff = x - xj
  return b + 2 * c * xDiff + 3 * d * xDiff * xDiff

proc newCubicSpline*[T: SomeFloat](X: openArray[float], Y: openArray[
    T]): InterpolatorType[T] =
  let sortedData = sortDataset(X, Y)
  var xSorted = newSeq[float](X.len)
  var ySorted = newSeq[T](Y.len)
  for i in 0 .. sortedData.high:
    xSorted[i] = sortedData[i][0]
    ySorted[i] = sortedData[i][1]
  let coeffs = constructCubicSpline(xSorted, ySorted)
  result = InterpolatorType[T](X: xSorted, coeffs_T: coeffs, high: xSorted.high,
      len: xSorted.len, eval_handler: eval_cubicspline,
      deriveval_handler: derivEval_cubicspline)


## HermiteSpline

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
  let p1 = spline.coeffs_T[n][0]
  let p2 = spline.coeffs_T[n+1][0]
  let m1 = spline.coeffs_T[n][1]
  let m2 = spline.coeffs_T[n+1][1]
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
  let p1 = spline.coeffs_T[n][0]
  let p2 = spline.coeffs_T[n+1][0]
  let m1 = spline.coeffs_T[n][1]
  let m2 = spline.coeffs_T[n+1][1]
  result = (h00*p1 + h10*xDiff*m1 + h01*p2 + h11*xDiff*m2) / xDiff

proc newHermiteSpline*[T](X: openArray[float], Y, dY: openArray[
    T]): InterpolatorType[T] =
  ## X, Y and dY must be sorted by X in ascending order
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
  var coeffs = newSeq[seq[T]](Y.len)
  for i in 0 .. Y.high:
    coeffs[i] = @[Y[i], dY[i]]
  result = InterpolatorType[T](X: @X, coeffs_T: coeffs, high: X.high,
      len: X.len, eval_handler: eval_hermitespline,
      deriveval_handler: derivEval_hermitespline)

proc newHermiteSpline*[T](X: openArray[float], Y: openArray[
    T]): InterpolatorType[T] =
  # if only (x, y) is given, use three-point difference to calculate dY.
  let sortedData = sortDataset(X, Y)
  var xSorted = newSeq[float](X.len)
  var ySorted = newSeq[T](Y.len)
  var dySorted = newSeq[T](Y.len)
  for i in 0 .. sortedData.high:
    xSorted[i] = sortedData[i][0]
    ySorted[i] = sortedData[i][1]
  let highest = dySorted.high
  dySorted[0] = (ySorted[1] - ySorted[0]) / (xSorted[1] - xSorted[0])
  dySorted[highest] = (ySorted[highest] - ySorted[highest-1]) / (xSorted[
      highest] - xSorted[highest-1])
  for i in 1 .. highest-1:
    dySorted[i] = 0.5 * ((ySorted[i+1] - ySorted[i])/(xSorted[i+1] - xSorted[
        i]) + (ySorted[i] - ySorted[i-1])/(xSorted[i] - xSorted[i-1]))
  var coeffs = newSeq[seq[T]](Y.len)
  for i in 0 .. Y.high:
    coeffs[i] = @[ySorted[i], dySorted[i]]
  result = InterpolatorType[T](X: xSorted, coeffs_T: coeffs, high: xSorted.high,
      len: xSorted.len, eval_handler: eval_hermitespline,
      deriveval_handler: derivEval_hermitespline)


# General Spline stuff

template eval*[T](interpolator: InterpolatorType[T], x: float): untyped =
  interpolator.eval_handler(interpolator, x)

template derivEval*[T](interpolator: InterpolatorType[T], x: float): untyped =
  interpolator.deriveval_handler(interpolator, x)

proc eval*[T](spline: InterpolatorType[T], x: openArray[float]): seq[T] =
  result = newSeq[T](x.len)
  for i, xi in x:
    result[i] = eval(spline, xi)

proc toProc*[T](spline: InterpolatorType[T]): InterpolatorProc[T] =
  result = proc(x: float): T = eval(spline, x)

converter toNumContextProc*[T](spline: InterpolatorType[T]): NumContextProc[T] =
  result = proc(x: float, ctx: NumContext[T]): T = eval(spline, x)

proc derivEval*[T](spline: InterpolatorType[T], x: openArray[float]): seq[T] =
  result = newSeq[T](x.len)
  for i, xi in x:
    result[i] = derivEval(spline, xi)

proc toDerivProc*[T](spline: InterpolatorType[T]): InterpolatorProc[T] =
  result = proc(x: float): T = derivEval(spline, x)

proc toDerivNumContextProc*[T](spline: InterpolatorType[T]): NumContextProc[T] =
  result = proc(x: float, ctx: NumContext[T]): T = derivEval(spline, x)


##############################################
############ 2D Interpolation ################
##############################################

# Nearest Neighbour interpolation

proc eval_nearestneigh*[T](self: Interpolator2DType[T], x, y: float): T {.nimcall.} =
  assert self.xLim.lower <= x and x <= self.xLim.upper and self.yLim.lower <= y and y <= self.yLim.upper, "x and y must be inside the given points"
  let i = round((x - self.xLim.lower) / self.dx).toInt
  let j = round((y - self.yLim.lower) / self.dy).toInt
  result = self.z[i, j]

proc newNearestNeighboursInterpolator*[T](z: Tensor[T], xlim, ylim: (float, float)): Interpolator2DType[T] =
  ## Returns a nearest neighbour interpolator for regularly gridded data.
  ## z - Tensor with the function values. x corrensponds to the rows and y to the columns. Must be sorted so ascendingly in both variables.
  ## xlim - the lowest and highest x-value
  ## ylim - the lowest and highest y-value
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
  assert self.xLim.lower <= x and x <= self.xLim.upper and self.yLim.lower <= y and y <= self.yLim.upper, "x and y must be inside the given points"
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
  ## z - Tensor with the function values. x corrensponds to the rows and y to the columns. Must be sorted so ascendingly in both variables.
  ## xlim - the lowest and highest x-value
  ## ylim - the lowest and highest y-value
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

proc grad(tIn: Tensor[float], xdiff: float = 1.0): Tensor[float] =
  let t = tIn.squeeze
  doAssert t.rank == 1, " no was " & $t
  result = newTensor[float](t.size.int)
  var
    yMinus, yPlus: float
    mxdiff = xdiff
  for i in 0 ..< t.size:
    if i == 0:
      result[i] = (t[1] - t[0]) / xdiff
    elif i == t.size - 1:
      result[i] = (t[i] - t[i - 1]) / xdiff
    else:
      result[i] = (t[i + 1] - t[i - 1]) / (2 * xdiff)

proc grad(t: Tensor[float], axis: int): Tensor[float] =
  ## given 2D tensor, calc gradient in axis direction
  result = newTensor[float](t.shape)
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

proc computeAlpha(interp: Interpolator2DType,
                  x, y: int): Tensor[float] =
  if (x, y) in interp.alphaCache:
    result = interp.alphaCache[(x, y)]
    return
  let f = interp.z
  let fx = interp.xGrad
  let fy = interp.yGrad
  let fxy = interp.xyGrad
  var m1 = [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [-3, 3, -2, -1],
            [2, -2, 1, 1]].toTensor.asType(float)
  var m2 = [[1, 0, -3, 2],
            [0, 0, 3, -2],
            [0, 1, -2, 1],
            [0, 0, -1, 1]].toTensor.asType(float)
  var A = [[f[x, y],  f[x, y + 1],  fy[x, y],  fy[x, y+1]],
           [f[x + 1, y],  f[x + 1, y + 1],  fy[x+1, y],  fy[x+1, y+1]],
           [fx[x, y], fx[x, y+1], fxy[x, y], fxy[x, y+1]],
           [fx[x+1, y], fx[x+1, y+1], fxy[x+1, y], fxy[x+1, y+1]]].toTensor().asType(float)
  result = m1 * A * m2
  interp.alphaCache[(x, y)] = result

proc eval_bicubic*[T](self: Interpolator2DType[T], x, y: float): T {.nimcall.} =
  # find interval
  assert self.xLim.lower <= x and x <= self.xLim.upper and self.yLim.lower <= y and y <= self.yLim.upper, "x and y must be inside the given points"
  let i = min(floor((x - self.xLim.lower) / self.dx).toInt, self.z.shape[0] - 2)
  let j = min(floor((y - self.yLim.lower) / self.dy).toInt, self.z.shape[1] - 2)
  # transform x and y to unit square
  let xCorner = self.xLim.lower + i.toFloat * self.dx
  let x = (x - xCorner) / self.dx
  let yCorner = self.yLim.lower + j.toFloat * self.dy
  let y = (y - yCorner) / self.dy
  let alpha = computeAlpha(self, i, j)
  result = dot([1.0, x, x*x, x*x*x].toTensor, alpha * [1.0, y, y*y, y*y*y].toTensor)

proc newBicubicSpline*[T](z: Tensor[T], xlim, ylim: (float, float)): Interpolator2DType[T] =
  ## Returns a bicubic spline for regularly gridded data.
  ## z - Tensor with the function values. x corrensponds to the rows and y to the columns. Must be sorted so ascendingly in both variables.
  ## xlim - the lowest and highest x-value
  ## ylim - the lowest and highest y-value
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

# General Interpolator2D stuff

template eval*[T](interpolator: Interpolator2DType[T], x, y: float): untyped =
  interpolator.eval_handler(interpolator, x, y)


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
  let blspline = newBilinearSpline(z, (0.0, 2.0), (0.0, 2.0))
  let bcspline = newBicubicSpline(z, (0.0, 2.0), (0.0, 2.0))
  let x = 1.8
  let y = 2.0
  echo "Linear: ", blspline.eval(x, y)
  echo "Cubic:  ", bcspline.eval(x, y)
  let randTensor = randomTensor([10, 10], 75).asType(float)
  let nearestInterp = newNearestNeighboursInterpolator(randTensor, (0.0, 9.0), (0.0, 9.0))
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
  # Result:
  # Nearest ............................ 14.972 ms    ±0.478  x10
  # Linear ............................. 37.913 ms    ±1.122  x10
  # Cubic ............................. 693.276 ms    ±4.326  x10
