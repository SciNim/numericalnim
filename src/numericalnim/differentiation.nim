import arraymancer

proc diff1dForward*[U, T](f: proc(x: U): T, x0: U, h: U): T =
  ## Numerically calculate the derivative of f(x) at x0 using a step size h.
  ## Uses forward difference which has accuracy O(h)
  result = (f(x0 + h) - f(x0)) / h

proc diff1dBackward*[U, T](f: proc(x: U): T, x0: U, h: U): T =
  ## Numerically calculate the derivative of f(x) at x0 using a step size h.
  ## Uses backward difference which has accuracy O(h)
  result = (f(x0) - f(x0 - h)) / h

proc diff1dCentral*[U, T](f: proc(x: U): T, x0: U, h: U): T =
  ## Numerically calculate the derivative of f(x) at x0 using a step size h.
  ## Uses central difference which has accuracy O(h^2)
  result = (f(x0 + h) - f(x0 - h)) / (2*h)

proc secondDiff1dForward*[U, T](f: proc(x: U): T, x0: U, h: U): T =
  ## Numerically calculate the second derivative of f(x) at x0 using a step size h.
  result = (f(x0 + 2*h) - 2*f(x0 + h) + f(x0)) / (h*h)

proc secondDiff1dBackward*[U, T](f: proc(x: U): T, x0: U, h: U): T =
  ## Numerically calculate the second derivative of f(x) at x0 using a step size h.
  result = (f(x0) - 2*f(x0 - h) + f(x0 - 2*h)) / (h*h)

proc secondDiff1dCentral*[U, T](f: proc(x: U): T, x0: U, h: U): T =
  ## Numerically calculate the second derivative of f(x) at x0 using a step size h.
  ## Uses central difference which has accuracy O(h^2)
  result = (f(x0 + h) - 2*f(x0) + f(x0 - h)) / (h*h)

proc tensorGradient*[U; T: not Tensor](
    f: proc(x: Tensor[U]): T,
    x0: Tensor[U],
    h: U,
    fastMode: bool = false
    ): Tensor[T] =
  ## Calculates the gradient of f(x) w.r.t vector x at x0 using step size h.
  assert x0.rank == 1 # must be a 1d vector
  let f0 = f(x0) # make use of this with a `fastMode` switch so we use forward difference instead of central difference?
  let xLen = x0.shape[0]
  result = newTensor[T](xLen)
  var x = x0.clone()
  for i in 0 ..< xLen:
    x[i] += h
    let fPlusH = f(x)
    if fastMode:
      x[i] -= h # restore to original
      result[i] = (fPlusH - f0) / h
    else:
      x[i] -= 2*h
      let fMinusH = f(x)
      x[i] += h # restore to original (± float error)
      result[i] = (fPlusH - fMinusH) / (2 * h)

proc tensorGradient*[U, T](
    f: proc(x: Tensor[U]): Tensor[T],
    x0: Tensor[U],
    h: U,
    fastMode: bool = false
    ): Tensor[T] =
  ## Calculates the gradient of f(x) w.r.t vector x at x0 using step size h.
  ## Every column is the gradient of one component of f.
  assert x0.rank == 1 # must be a 1d vector
  let f0 = f(x0) # make use of this with a `fastMode` switch so we use forward difference instead of central difference?
  assert f0.rank == 1
  let rows = x0.shape[0]
  let cols = f0.shape[0]
  result = newTensor[T](rows, cols)
  var x = x0.clone()
  for i in 0 ..< rows:
    x[i] += h
    let fPlusH = f(x)
    if fastMode:
      x[i] -= h # restore to original
      result[i, _] = ((fPlusH - f0) / h).reshape(1, cols)
    else:
      x[i] -= 2*h
      let fMinusH = f(x)
      x[i] += h # restore to original (± float error)
      result[i, _] = ((fPlusH - fMinusH) / (2 * h)).reshape(1, cols)

proc tensorJacobian*[U, T](
    f: proc(x: Tensor[U]): Tensor[T],
    x0: Tensor[U],
    h: U,
    fastMode: bool = false
    ): Tensor[T] =
  ## Calculates the jacobian of f(x) w.r.t vector x at x0 using step size h.
  ## Every row is the gradient of one component of f.
  transpose(tensorGradient(f, x0, h, fastMode))


when isMainModule:
  import std/math
  import benchy
  proc f1(x: Tensor[float]): Tensor[float] =
    x.sum(0)
  let x0 = ones[float](10)
  echo tensorGradient(f1, x0, 1e-6)
  echo tensorGradient(f1, x0, 1e-6, true)
  echo tensorJacobian(f1, x0, 1e-6)

  proc f2(x: Tensor[float]): float =
    sum(x)
  echo tensorGradient(f2, x0, 1e-6)
  echo tensorGradient(f2, x0, 1e-6, true)

  let N = 1000
  timeIt "slow mode":
    for i in 0 .. N:
      keep tensorGradient(f1, x0, 1e-6, false)
  timeIt "fast mode":
    for i in 0 .. N:
      keep tensorGradient(f1, x0, 1e-6, true)
  timeIt "slow mode float":
    for i in 0 .. N:
      keep tensorGradient(f2, x0, 1e-6, false)
  timeIt "fast mode float":
    for i in 0 .. N:
      keep tensorGradient(f2, x0, 1e-6, true)
  timeIt "jacobian slow":
    for i in 0 .. N:
      keep tensorJacobian(f1, x0, 1e-6, false)
    



