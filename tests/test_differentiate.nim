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

suite "Multi dimensional numeric gradients":
  discard