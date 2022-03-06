import std/strformat
import arraymancer

proc diff1dForward*[U, T](f: proc(x: U): T, x0: U, h: U = U(1e-6)): T =
  ## Numerically calculate the derivative of f(x) at x0 using a step size h.
  ## Uses forward difference which has accuracy O(h)
  result = (f(x0 + h) - f(x0)) / h

proc diff1dBackward*[U, T](f: proc(x: U): T, x0: U, h: U = U(1e-6)): T =
  ## Numerically calculate the derivative of f(x) at x0 using a step size h.
  ## Uses backward difference which has accuracy O(h)
  result = (f(x0) - f(x0 - h)) / h

proc diff1dCentral*[U, T](f: proc(x: U): T, x0: U, h: U = U(1e-6)): T =
  ## Numerically calculate the derivative of f(x) at x0 using a step size h.
  ## Uses central difference which has accuracy O(h^2)
  result = (f(x0 + h) - f(x0 - h)) / (2*h)

proc secondDiff1dForward*[U, T](f: proc(x: U): T, x0: U, h: U = U(1e-6)): T =
  ## Numerically calculate the second derivative of f(x) at x0 using a step size h.
  result = (f(x0 + 2*h) - 2*f(x0 + h) + f(x0)) / (h*h)

proc secondDiff1dBackward*[U, T](f: proc(x: U): T, x0: U, h: U = U(1e-6)): T =
  ## Numerically calculate the second derivative of f(x) at x0 using a step size h.
  result = (f(x0) - 2*f(x0 - h) + f(x0 - 2*h)) / (h*h)

proc secondDiff1dCentral*[U, T](f: proc(x: U): T, x0: U, h: U = U(1e-6)): T =
  ## Numerically calculate the second derivative of f(x) at x0 using a step size h.
  ## Uses central difference which has accuracy O(h^2)
  result = (f(x0 + h) - 2*f(x0) + f(x0 - h)) / (h*h)

proc tensorGradient*[U; T: not Tensor](
    f: proc(x: Tensor[U]): T,
    x0: Tensor[U],
    h: U = U(1e-6),
    fastMode: bool = false
  ): Tensor[T] =
  ## Calculates the gradient of f(x) w.r.t vector x at x0 using step size h.
  ## By default it uses central difference for approximating the derivatives. This requires two function evaluations per derivative.
  ## When fastMode is true it will instead use the forward difference which only uses 1 function evaluation per derivative but is less accurate.
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
    h: U = U(1e-6),
    fastMode: bool = false
  ): Tensor[T] =
  ## Calculates the gradient of f(x) w.r.t vector x at x0 using step size h.
  ## Every column is the gradient of one component of f.
  ## By default it uses central difference for approximating the derivatives. This requires two function evaluations per derivative.
  ## When fastMode is true it will instead use the forward difference which only uses 1 function evaluation per derivative but is less accurate.
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
    h: U = U(1e-6),
    fastMode: bool = false
  ): Tensor[T] =
    ## Calculates the jacobian of f(x) w.r.t vector x at x0 using step size h.
    ## Every row is the gradient of one component of f.
    ## By default it uses central difference for approximating the derivatives. This requires two function evaluations per derivative.
    ## When fastMode is true it will instead use the forward difference which only uses 1 function evaluation per derivative but is less accurate.
    transpose(tensorGradient(f, x0, h, fastMode))

proc mixedDerivative*[U, T](f: proc(x: Tensor[U]): T, x0: var Tensor[U], indices: (int, int), h: U = U(1e-6)): T =
  result = 0
  let i = indices[0]
  let j = indices[1]
  # f(x+h, y+h)
  x0[i] += h
  x0[j] += h
  result += f(x0)

  # f(x+h, y-h)
  x0[j] -= 2*h
  result -= f(x0)

  # f(x-h, y-h)
  x0[i] -= 2*h
  result += f(x0)

  # f(x-h, y+h)
  x0[j] += 2*h
  result -= f(x0)

  # restore x0
  x0[i] += h
  x0[j] -= h

  result *= 1 / (4 * h*h)
  

proc tensorHessian*[U; T: not Tensor](
    f: proc(x: Tensor[U]): T,
    x0: Tensor[U],
    h: U = U(1e-6)
  ): Tensor[T] =
    assert x0.rank == 1 # must be a 1d vector
    let f0 = f(x0)
    let xLen = x0.shape[0]
    var x = x0.clone()
    result = zeros[T](xLen, xLen)
    for i in 0 ..< xLen:
      for j in i ..< xLen:
        let mixed = mixedDerivative(f, x, (i, j), h)
        result[i, j] = mixed
        result[j, i] = mixed

proc checkGradient*[U; T: not Tensor](f: proc(x: Tensor[U]): T, fGrad: proc(x: Tensor[U]): Tensor[T], x0: Tensor[U], tol: T): bool =
  ## Checks if the provided gradient function `fGrad` gives the same values as numeric gradient.
  let numGrad = tensorGradient(f, x0)
  let grad = fGrad(x0)
  result = true
  for i, x in abs(numGrad - grad):
    if x > tol:
      echo fmt"Gradient at index {i[0]} has error: {x} (tol = {tol})"
      result = false

proc checkGradient*[U; T](f: proc(x: Tensor[U]): Tensor[T], fGrad: proc(x: Tensor[U]): Tensor[T], x0: Tensor[U], tol: T): bool =
  ## Checks if the provided gradient function `fGrad` gives the same values as numeric gradient.
  let numGrad = tensorGradient(f, x0)
  let grad = fGrad(x0)
  result = true
  for i, x in abs(numGrad - grad):
    if x > tol:
      echo fmt"Gradient at index {i[0]} has error: {x} (tol = {tol})"
      result = false



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
    



