import math

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


