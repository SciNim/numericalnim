import std/[unittest, math]
import numericalnim
import arraymancer

proc f_1d(x: float): float = 2 * sin(x) * cos(x) # same as sin(2*x)
proc exact_deriv_1d(x: float): float = 2*cos(2*x)
proc exact_secondderiv_1d(x: float): float = -4*sin(2*x)

suite "1D numeric differentiation":
  test "Forward difference":
    for x in numericalnim.linspace(0, 10, 100):
      let numDiff = diff1dForward(f_1d, x)
      let exact = exact_deriv_1d(x)
      check abs(numDiff - exact) < 3e-6
  
  test "Backward difference":
    for x in numericalnim.linspace(0, 10, 100):
      let numDiff = diff1dBackward(f_1d, x)
      let exact = exact_deriv_1d(x)
      check abs(numDiff - exact) < 3e-6

  test "Central difference":
    for x in numericalnim.linspace(0, 10, 100):
      let numDiff = diff1dCentral(f_1d, x)
      let exact = exact_deriv_1d(x)
      check abs(numDiff - exact) < 2e-9

  test "Forward second difference":
    for x in numericalnim.linspace(0, 10, 100):
      let numDiff = secondDiff1dForward(f_1d, x)
      let exact = exact_secondderiv_1d(x)
      check abs(numDiff - exact) < 4e-3

  test "Backward second difference":
    for x in numericalnim.linspace(0, 10, 100):
      let numDiff = secondDiff1dBackward(f_1d, x)
      let exact = exact_secondderiv_1d(x)
      check abs(numDiff - exact) < 4e-3

  test "Central second difference":
    for x in numericalnim.linspace(0, 10, 100):
      let numDiff = secondDiff1dCentral(f_1d, x)
      let exact = exact_secondderiv_1d(x)
      check abs(numDiff - exact) < 4e-4

proc fScalar(x: Tensor[float]): float =
  # This will be a function of three variables
  # f(x0, x1, x2) = x0^2 + 2 * x0 * x1 + sin(x2)
  result = x[0]*x[0] + 2 * x[0] * x[1] + sin(x[2])

proc scalarGradient(x: Tensor[float]): Tensor[float] =
  # Gradient is (2*x0 + 2*x1, 2*x0, cos(x2))
  result = zeros_like(x)
  result[0] = 2*x[0] + 2*x[1]
  result[1] = 2*x[0]
  result[2] = cos(x[2])

proc fMultidim(x: Tensor[float]): Tensor[float] =
  # Function will have 3 inputs and 2 outputs (important that they aren't the same for testing!)
  # f(x0, x1, x2) = (x0*x1*x2^2, x1*sin(2*x2))
  result = zeros[float](2)
  result[0] = x[0]*x[1]*x[2]*x[2]
  result[1] = x[1] * sin(2*x[2])

proc multidimGradient(x: Tensor[float]): Tensor[float] =
  # The gradient (Jacobian transposed) is:
  # x1*x2^2     0
  # x0*x2^2     sin(2*x2)
  # 2*x0*x1*x2  2*x1*cos(2*x2)
  result = zeros[float](3, 2)
  result[0, 0] = x[1]*x[2]*x[2]
  result[0, 1] = 0
  result[1, 0] = x[0]*x[2]*x[2]
  result[1, 1] = sin(2*x[2])
  result[2, 0] = 2*x[0]*x[1]*x[2]
  result[2, 1] = 2*x[1]*cos(2*x[2])
  

suite "Multi dimensional numeric gradients":
  test "Scalar valued function of 3 variables":
    for x in numericalnim.linspace(0, 1, 10):
      for y in numericalnim.linspace(0, 1, 10):
        for z in numericalnim.linspace(0, 1, 10):
          let x0 = [x, y, z].toTensor
          let numGrad = tensorGradient(fScalar, x0)
          let exact = scalarGradient(x0)
          for err in abs(numGrad - exact):
            check err < 5e-10

  test "Scalar valued function of 3 variables (fast mode)":
    for x in numericalnim.linspace(0, 1, 10):
      for y in numericalnim.linspace(0, 1, 10):
        for z in numericalnim.linspace(0, 1, 10):
          let x0 = [x, y, z].toTensor
          let numGrad = tensorGradient(fScalar, x0, fastMode=true)
          let exact = scalarGradient(x0)
          for err in abs(numGrad - exact):
            check err < 5e-6

  test "Multi-dimensional function of 3 variables":
    for x in numericalnim.linspace(0, 1, 10):
      for y in numericalnim.linspace(0, 1, 10):
        for z in numericalnim.linspace(0, 1, 10):
          let x0 = [x, y, z].toTensor
          let numGrad = tensorGradient(fMultidim, x0)
          let exact = multidimGradient(x0)
          for err in abs(numGrad - exact):
            check err < 1e-10

  test "Multi-dimensional function of 3 variables (fast mode)":
    for x in numericalnim.linspace(0, 1, 10):
      for y in numericalnim.linspace(0, 1, 10):
        for z in numericalnim.linspace(0, 1, 10):
          let x0 = [x, y, z].toTensor
          let numGrad = tensorGradient(fMultidim, x0, fastMode=true)
          let exact = multidimGradient(x0)
          for err in abs(numGrad - exact):
            check err < 2e-6

  
  test "Jacobian":
    for x in numericalnim.linspace(0, 1, 10):
      for y in numericalnim.linspace(0, 1, 10):
        for z in numericalnim.linspace(0, 1, 10):
          let x0 = [x, y, z].toTensor
          let numJacobian = tensorJacobian(fMultidim, x0)
          let exact = multidimGradient(x0).transpose
          for err in abs(numJacobian - exact):
            check err < 1e-10
