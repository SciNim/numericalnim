# v0.8.7
The 1D interpolation methods now support extrapolation using these methods:
- `Constant`: Set all points outside the range of the interpolator to `extrapValue`.
- `Edge`: Use the value of the left/right edge.
- `Linear`: Uses linear extrapolation using the two points closest to the edge.
- `Native` (default): Uses the native method of the interpolator to extrapolate. For Linear1D it will be a linear extrapolation, and for Cubic and Hermite splines it will be cubic extrapolation.
- `Error`: Raises an `ValueError` if `x` is outside the range. 

These are passed in as an argument to `eval` and `derivEval`:
```nim
let valEdge = interp.eval(x, Edge)
let valConstant = interp.eval(x, Constant, NaN)
```

# v0.8.6
- `levmarq` now accepts `yError`.
- `paramUncertainties` allows you to calculate the uncertainties of fitted parameters.
- `chi2` test added

# v0.8.5
Fix rbf bug.

# v0.8.4
With radial basis function interpolation, `numericalnim` finally gets an interpolation method which works on scattered data in arbitrary dimensions!

Basic usage:
```
let interp = newRbf(points, values)
let result = interp.eval(evalPoints)
```

# v0.8.1-v0.8.3
CI-related bug fixes.

# v0.8.0 - 09.05.2022
## Optimization has joined the chat
Multi-variate optimization and differentiation has been introduced.

- `numericalnim/differentiate` offers `tensorGradient(f, x)` which calculates the gradient of `f` w.r.t `x` using finite differences, `tensorJacobian` (returns the transpose of the gradient), `tensorHessian`, `mixedDerivative`. It also provides `checkGradient(f, analyticGrad, x, tol)` to verify that the analytic gradient is correct by comparing it to the finite difference approximation.
- `numericalnim/optimize` now has several multi-variate optimization methods:
  - `steepestDescent`
  - `newton`
  - `bfgs`
  - `lbfgs`
  - They all have the function signatures like:
    ```nim
    proc bfgs*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], options: OptimOptions[U, StandardOptions] = bfgsOptions[U](), analyticGradient: proc(x: Tensor[U]): Tensor[T] = nil): Tensor[U]
    ```
    where `f` is the function to be minimized, `x0` is the starting guess, `options` contain options like tolerance (each method has it own options type which can be created by for example `lbfgsOptions` or `newtonOptions`), `analyticGradient` can be supplied to avoid having to do finite difference approximations of the derivatives.
  - There are 4 different line search methods supported and those are set in the `options`: `Armijo, Wolfe, WolfeStrong, NoLineSearch`.
  - `levmarq`: non-linear least-square optimizer
    ```nim
    proc levmarq*[U; T: not Tensor](f: proc(params: Tensor[U], x: U): T, params0: Tensor[U], xData: Tensor[U], yData: Tensor[T], options: OptimOptions[U, LevmarqOptions[U]] = levmarqOptions[U]()): Tensor[U]
    ```
    - `f` is the function you want to fit to the parameters in `param` and `x` is the value to evaluate the function at. 
    - `params0` is the initial guess for the parameters
    - `xData` is a 1D Tensor with the x points and `yData` is a 1D Tensor with the y points.
    - `options` can be created using `levmarqOptions`.
    - Returns the final parameters
  

Note: There are basic tests to ensure these methods converge for simple problems, but they are not tested on more complex problems and should be considered experimental until more tests have been done. Please try them out, but don't rely on them for anything important for now. Also, the API isn't set in stone yet so expect that it may change in future versions.


# v0.7.1 -25.01.2022

Add a `nimCI` task for the Nim CI to run now that the tests have external dependencies.

# v0.7.0 - 25.01.2022

This is a *breaking* release, due to the changes in PR #25.

`NumContext` (and types taking `NumContext` as an argument) are now
two-fold generic. The floating point like type used during
computation may now be overwritten.

This is a breaking change, as the `newNumContext` procedure must now
be given two generic arguments. For most procedures the signature
was only extended to use `float` as the secondary type, leaving them
as taking single generic arguments.
`adapdiveGauss` is an exception and thus now requires the user to
hand *both* types.

- transition for `adaptiveGauss`:
  Calling as: `adaptiveGauss[T, float](...)` will produce the old
  behavior. In the future a nicer interface may be designed.
- transition for `newNumContext`:
  Calling as: `newNumContext[T, float]` will produce the old behavior.

This change was a step towards a more (likely concept based) interface
for SciNim libraries for better interop. It allows for example to
integrate over a `Measurement`.

# v0.6.3

- fixes an issue that might arise if 2D interpolation is used together
  with multithreading where the Nim compiler gets confused about GC
  unsafety (#23)
- Added `linear1d`
- Added `barycentric2d` which works on non-gridded data.
