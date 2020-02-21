# NumericalNim

NumericalNim is a collection of numerical methods written in Nim. Currently it has support for integration, optimization, interpolation and ODE. It can operate on floats and custom structures, such as vectors and tensors (if they support a set of operators).

# Installation
Install NumericalNim using Nimble:

`nimble install numericalnim`

## Suggested compilation flags

With [FMA](https://en.wikipedia.org/wiki/FMA_instruction_set) and [AVX2](https://en.wikipedia.org/wiki/AVX2) there exist some [SIMD](https://en.wikipedia.org/wiki/SIMD) instruction sets which can increase the performance on [x86](https://en.wikipedia.org/wiki/X86) machines.
To enable these, nim has to pass some flags to the C compiler.
Below is a small table, listing the flags to add when compiling with `nim c` for some widely used compilers:

Compiler     | Flags
------------ | -----
clang        | `-t:-mavx2 -t:-mfma -t:-ffp-contract=fast`
gcc          | `-t:-mavx2 -t:-mfma`
icc          | `-t:-march=core-avx2`
msvc         | `-t:arch:AVX2 -t:fp:fast`

# ODE
## Initial value problems (IVP)
## The integrators
These are the implemented ODE integrators:

### First order ODE: y' = f(t, y) 
- `rk21` - Heun's Adaptive 2nd order method.
- `BS32` - Bogacki–Shampine 3rd order adaptive method.
- `DOPRI54` - Dormand & Prince's adaptive 5th order method.
- `Heun2` - Heun's 2nd order fixed timestep method.
- `Ralston2` - Ralston's 2nd order fixed timestep method.
- `Kutta3` - Kutta's 3rd order fixed timestep method.
- `Heun3` - Heuns's 3rd order fixed timestep method.
- `Ralston3` - Ralston's 3rd order fixed timestep method.
- `SSPRK3` - Strong Stability Preserving Runge-Kutta 3rd order fixed timestep method.
- `Ralston4` - Ralston's 4th order fixed timestep method.
- `Kutta4` - Kutta's 4th order fixed timestep method.
- `RK4` - The standard 4th order, fixed timestep method we all know and love.
- `Tsit54` - Tsitouras adaptive 5th order method.
- `Vern65` - Verner's "most efficient" 6th order adaptive timestep method.
- `Vern76` - Verner's "most efficient" 7th order adaptive timestep method.




### Dense Output
All integrators support dense output using a 3rd order Hermite interpolant. Method specific interpolants may be added in the future.

## Usage
Using default parameters the methods need 3 things: the function f(t, y) = y'(t, y), the initial values and the timepoints that you want the solution y(t) at.

## Quick Tutorial
f(t, y) = 0.1*y

y(0) = 1

If we translate this to code we get:
```nim
import math
import numericalnim

proc f(t, y: float): float = 0.1*y

let y0 = 1.0
```

Now we must decide at which timepoints we want the solution at. NumericalNim provides two easy functions for creating seqs of floats:

`linspace(x1, x2: float, N: int): seq[float]` - Creates a seq of N evenly spaced points between x1 and x2.

`arange(x1, x2, dx: float): seq[float]` - Creates a seq of floats between x1 and x2 with the spacing `dx`. It has the optional parameters with default values: `includeStart=true`, `includeEnd=false`.

```nim
let tspan = linspace(-2.0, 2.0, 5)
```
This should do the trick. Now it's time to fire up the integrators!

The integrators are called using the proc `solveODE` and it returns a tuple with the timepoints and the function values at those points. We choose which integrator we wish to use by passing the name of the integrator. List of integrators can be found above.

```nim
let (t1, y1) = solveODE(f, y0, tspan, integrator="rk4")
let (t2, y2) = solveODE(f, y0, tspan, integrator="dopri54")
```

If we `echo y1, y2` we see that both gives roughly the same answer. 

```nim
@[0.8187307530779675, 0.9048374180359479, 1.0, 1.105170918075657, 1.221402758160196]
@[0.8187307530779815, 0.9048374180359576, 1.0, 1.105170918075645, 1.221402758160169]
```

The analytical solution to the ODE is y(t) = exp(0.1*t) so we can compare both methods and see if the error is different:
```nim
let answer = exp(0.1 * 2.0)
echo "RK4: ", answer - y1[4]
echo "DOPRI54: ", answer - y2[4]
```

```nim
RK4: -2.642330798607873e-014
DOPRI54: 1.110223024625157e-015
```

As we can see both methods gives a good numerical approximation of this simple ODE.

Now it's time to play with the parameters and NumericalNim makes it easy to handle them using a `ODEoptions` type that is passed to `solveODE`. Here are the defaults:
```nim
let options = newODEoptions(dt = 1e-4, relTol = 1e-4, dtMax = 1e-2, dtMin = 1e-8, tStart = 0.0)
let (t, y) = solveODE(f, y0, tspan, options = options, integrator = "dopri54")
```
- `dt` - the timestep used by the fixed timestep methods.
- `relTol` - the relative tolerance used by the adaptive methods.
- `dtMax` - the maximum allowed dt the adaptive method is allowed to use.
- `dtMin` - the smallest allowed dt the adaptive method is allowed to use.
- `tStart` - the time that the initial values are provided at.

If we lower `dt` to `1e-1` and `relTol` to `1e-1` we should get a higher error than last time. Let's see!

```nim
let options = newODEoptions(dt = 1e-1, relTol = 1e-1, dtMax = 1e-2, dtMin = 1e-8, tStart = 0.0)
let (t3, y3) = solveODE(f, y0, tspan, options = options, integrator = "rk4")
let (t4, y4) = solveODE(f, y0, tspan, options, integrator="dopri54")

echo "RK4: ", answer - y3[4]
echo "DOPRI54: ", answer - y4[4]
```
```nim
RK4: 2.018807343517892e-011
DOPRI54: 1.110223024625157e-015
```
The error of `RK4` got a bit higher but not `DOPRI54`. If we increase `dtMax` to `1.0` we get the following:
```nim
RK4: 2.018807343517892e-011
DOPRI54: -3.126410241804933e-010
```
Now the error is higher for `DOPRI54` as well.

# Integration
## 1D Integration
## The methods
- `trapz` - Uses the trapezoidal rule to integrate both functions and discrete points. 2nd order method.
- `simpson` - Uses Simpson's rule to integrate both functions and discrete points. 4th order method.
- `adaptiveSimpson` - Uses a adaptive Simpson's rule to subdivide the integration interval in finer pieces where the integrand is changing a lot and wider pieces in intervals where it doesn't change much. This allows it to perform the integral efficiently and still have accuracy. The error is approximated using Richardson extrapolation. So technically the values it outputs are actually not Simpson's, but Booles' method.
- `romberg` - Uses Romberg integration to integrate both functions and discrete points. __Note:__ If discrete points are provided they must be equally spaced and the number of points must be of the form 2^k + 1 ie 3, 5, 9, 17, 33, 65, 129 etc.
- `cumtrapz` - Uses the trapezoidal rule to integrate both functions and discrete points but it outputs a seq of the integral values at provided x-values. 
- `cumsimpson` - Uses Simpson's rule to integrate both functions and discrete points but it outputs a seq of the integral values at provided x-values.
- `gaussQuad` - Uses Gauss-Legendre Quadrature to integrate functions. Choose between 20 different accuracies by setting how many function evaluations should be made on each subinterval with the `nPoints` parameter (1 - 20 is valid options).
- `adaptiveGauss` - Uses Gauss-Kronrod Quadrature to adaptivly integrate function.

## Usage
Using the default parameters you need to provide one of these sets:
- ``f(x)``, ``xStart``, ``xEnd``
- ``Y``, ``X``, where ``Y`` contains the values of the integrand at the points in X.

For the cumulative versions you need to provide either of:
- `f(x)`, `X`, where `X` is the points you want to evaluate the integral at.
- `Y`, ``X``, where ``Y`` contains the values of the integrand at the points in ``X``.
- ``f(x)``, ``X``, ``dx``, where ``X`` is the points you want to evaluate the integral at. ``dx`` is the timestep you want to use to step using. If ``dx`` is not provided it will use ``X``. Use this if the resolution of ``X`` isn't enough to give you a low enough error.

The proc `f` must be of the form: 
```nim
proc f[T](x: float, optional: seq[T]): T
```
If you don't understand what the "T" stands for, you can replace it with "float" in your head and read up on "Generics" in Nim.
## Quick Tutorial
We want to evaluate the integral of f(x) = sin(x) from 0 to Pi. For this we can choose either `trapz`, `simpson`, `adaptiveSimpson`, `gaussQuad` or `romberg`. Let's do all of them and compare them!
```nim
import math
import numericalnim
proc f(x: float, optional: seq[float]): float = sin(x)
let xStart = 0.0
let xEnd = PI

let integral_trapz = trapz(f, xStart, xEnd)
let integral_simpson = simpson(f, xStart, xEnd)
let integral_adaptiveSimpson = adaptiveSimpson(f, xStart, xEnd)
let integral_gaussQuad = gaussQuad(f, xStart, xEnd)
let integral_adaptiveGauss = adaptiveGauss(f, xStart, xEnd)
let integral_romberg = romberg(f, xStart, xEnd)

echo "Trapz: ", integral_trapz
echo "Simpson: ", integral_simpson
echo "Adaptive Simpson: ", integral_adaptiveSimpson
echo "Gauss: ", integral_gaussQuad
echo "Adaptive Gauss: ", integral_adaptiveGauss
echo "Romberg: ", integral_romberg
```
```nim
Trapz: 1.999993420259403
Simpson: 2.000000000017319
Adaptive Simpson: 1.999999999997953
Gauss: 1.999999999999998
Adaptive Gauss: 2.0
Romberg: 1.999999999999077
```
The correct value is 2 so all of them seems to work, great! Let's compare the errors:
```nim
echo "Trapz error: ", 2.0 - integral_trapz
echo "Simpson error: ", 2.0 - integral_simpson
echo "Adaptive Simpson error: ", 2.0 - integral_adaptiveSimpson
echo "Gauss error: ", 2.0 - integral_gaussQuad
echo "Adaptive Gauss error: ", 2.0 - integral_adaptiveGauss
echo "Romberg error: ", 2.0 - integral_romberg
```
```nim
Trapz error: 6.5797405970347e-006
Simpson error: -1.731903509494259e-011
Adaptive Simpson error: 2.046807168198939e-012
Gauss error: 1.554312234475219e-015
Adaptive Gauss error: 0.0
Romberg error: 9.234835118832052e-013
```
We see that the trapezoidal rule is less accurate than the others as we could expect. 

## Example: Cumulative
If we want to calculate the cumulative integral we get the integral evaluated with different upper limits. In this example we are given the acceleration of an object at different timepoints and we want to know how far this object has traveled at each point in time. We assume the initial velocity and position are 0. Position is velocity integrated over time, and velocity is acceleration integrated over time. If we had just used one of the ordinary methods we would only have gotten the _final_ velocity, not the intermediate ones. "Why do we need them?" you may ask. It's because the more points we have, the better our approximation of the integral will be. If we only had the initial and final velocity we could at best get the approximation of a line, which may not necessarily be correct. If we have more points we can approximate it much better, hence why we want the intermediate values.

The values we are given are:
```nim
import numericalnim

let a = @[1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
let t = @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```
In this case `a` represents `Y` and `t` represents `X`. Now we calculate a new sequence containing the velocities:
```nim
let v_trapz = cumtrapz(a, t)
let v_simpson = cumsimpson(a, t)

echo "Velocity Trapz:   ", v_trapz
echo "Velocity Simpson: ", v_simpson
```
```nim
Velocity Trapz:   @[0.0, 1.25, 3.0, 5.25, 8.0, 11.25]
Velocity Simpson: @[0.0, 1.25, 3.0, 5.25, 8.0, 11.25]
```
As we can see both methods gave the same result, which makes sense because the data is linear so both methods could integrate it exactly. Let's now get the positions:
```nim
let p_trapz = cumtrapz(v_trapz, t)
let p_simpson = cumsimpson(v_simpson, t)

echo "Position Trapz:   ", p_trapz
echo "Position Simpson: ", p_simpson
```
```nim
Position Trapz:   @[0.0, 0.625, 2.75, 6.875, 13.5, 23.125]
Position Simpson: @[0.0, 0.583, 2.67, 6.75, 13.3, 22.917](Rounded values)
```
Now they are starting to differ from each outer. The rounded analytic solutions are:
```nim
@[0.0, 0.583, 6.75, 13.3, 22.917]
```
We see that simpson gets it spot on while trapz is a bit off. The velocity is a parabola so you can't exactly approximate it with a straight line as trapz does.

## Optional parameters / API
You can pass some additional parameters to the functions if you don't want to play with the defaults. For now here are the procs:
```nim
proc trapz*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, N = 500, optional: openArray[T] = @[]): T

proc trapz*[T](Y: openArray[T], X: openArray[float]): T

proc cumtrapz*[T](Y: openArray[T], X: openArray[float]): seq[T]

proc cumtrapz*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[], dx = 1e-5): seq[T]

proc simpson*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, N = 500, optional: openArray[T] = @[]): T

proc simpson*[T](Y: openArray[T], X: openArray[float]): T

proc adaptiveSimpson*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, tol = 1e-8, optional: openArray[T] = @[]): T

proc cumsimpson*[T](Y: openArray[T], X: openArray[float]): seq[T]

proc cumsimpson*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[], dx = 1e-5): seq[T]

proc romberg*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, depth = 8, tol = 1e-8, optional: openArray[T] = @[]): T

proc romberg*[T](Y: openArray[T], X: openArray[float]): T

proc gaussQuad*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, N = 100, nPoints = 7, optional: openArray[T] = @[]): T

proc adaptiveGauss*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, tol = 1e-8, optional: openArray[T] = @[]): T
```
If you don't understand what the "T" stands for, you can replace it with "float" in your head and read up on "Generics" in Nim.

# Optimization
## Optimization methods
## 1 dimensional function optimization
So far only a few methods have been implemented:

### One Dimensional optimization methods
- `steepest_descent` - Standard method for local minimum finding over a 2D plane
- `conjugate_gradient` - iterative implementation of solving Ax = b
- `newtons` - Newton-Raphson implementation for 1-dimensional functions

## Usage
Using default parameters the methods need 3 things: the function f(t, y) = y'(t, y), the initial values and the timepoints that you want the solution y(t) at.

## Quick Tutorial

Say we have some differentiable function and we would like to find one of its roots

f = $\frac{1}{3}$x$^{3}$ - 2x$^{2}$ + 3x

$\frac{df}{dx}$ = x$^{2}$ - 4x + 3



If we translate this to code we get:

```nim
import math
import numericalnim

proc f(x:float64): float64 = (1.0 / 3.0) * x ^ 3 - 2 * x ^ 2 + 3 * x
proc df(x:float64): float64 = x ^ 2 - 4 * x + 3
```

now given a starting point (and optional precision) we can estimate a nearby root
We know for this function our actual root is 0

```nim
import numericalnim
var start = 0.5
result = newtons(f, df, start)
echo result

-1.210640218782444e-23
```
Pretty close!

# Interpolation
## Natural Cubic Splines
Cubic splines are piecewise polynomials of degree 3 ie. it is defined differently on different intervals. It passes through all the supplied points and has a continuos derivative. To find which interval a certain x-value is in we use a binary search algorithm to find it instead of looping over all intervals one after another.  
### Usage
To create a cubic spline, you have to supply two seqs/arrays with floats: `X` which is the independent variable (the input) and Y which is the dependent (the output, the function value):
```nim
let X = [0.0, 0.5, 1.7, 2.0, 5.0]
let Y = [1.0, 3.5, -4.6, 0.1, 2.3]
let spline = newCubicSpline(X, Y)
```
Now the spline is saved in the variable `spline` and it can be evaluated in multiple (but under the hood the same) ways. The easiest way is to use the `eval` proc:
```nim
echo spline.eval(1.0)
```
This will print the value of the spline evaluated at x=1. If you want to use the spline as a function without having to supply the spline you can convert it to a proc:
```nim
let splineProc = spline.toProc()
echo splineProc(1.0)
```
You can also evaluate the derivative of the spline using `derivEval`, and you can turn the derivative into a proc as well:
```nim
echo spline.derivEval(1.0)
let derivProc = spline.toDerivProc()
echo derivProc(1.0)
```
This code will print the derivative of the spline at x=1.

# Utils
I have included a few handy tools in `numericalnim/utils`.
## Vector
Hurray! Yet another vector library! This was mostly done for my own use but I figured it could come in handy if one wanted to just throw something together. It's main purpose is to enable you to solve systems of ODEs using your own types (for example arbitrary precision numbers). The `Vector` type is just a glorified seq with operator overload. No vectorization (unless the compiler does it automatically) sadly. Maybe can get OpenMP to work (or you maybe you, dear reader, can fix it :wink).
The following operators and procs are supported:

- `+` : Addition between `Vector`s and floats.
- `-` : Addition between `Vector`s and floats.
- `+=`, `-=` : inplace addition and subtraction.
- `*` : Vector-scalar multiplication or inner product between two `Vector`s.
- `/` : Vector-scalar division.
- `*=`, `/=` : inplace Vector-scalar multiplication and division.
- `*.` : Elementwise multiplication between `Vector`s. (not nested `Vector`s)
- `/.` : Elementwise multiplication between `Vector`s. (not nested `Vector`s)
- `*.=`, `/.=` : inplace elementwise multiplication and division between `Vector`s. (not nested `Vector`s)
- `-` : negation (-Vector).
- `dot` : Same as `*` between `Vector`s. It is recursive so it will not be a matrix dot product if nested `Vector`s are used.
- `[]` : Use `v[i]` to get the i:th element of the `Vector`.
- `==` : Compares two `Vector`s to see if they are equal.
- `@` : Unpacks the Vector to (nested) seqs. Works with 1, 2 and 3 dimensional Vectors. 
- `^` : Element-wise exponentiation, works with natural and floating point powers, returns a new Vector object
- `norm` : General vector norm function. norm(`Vector`, 2) is the Euclidean norm. 

A `Vector` is created using the `newVector` proc and is passed an `openArray` of the elements:
```nim
var v1 = newVector([1.0, 2.0, 3.0])
var v2 = newVector([4.0, 5.0, 6.0])
echo v1 + v2
echo v1 /. v2
echo norm(v1)
```
```nim
Vector(@[5.0, 7.0, 9.0])
Vector(@[0.25, 0.4, 0.5])
3.741657386773941
```
## linspace & arange
`linspace` and `arange` are convenient procs to generate ordered seq's of floats. 
### linspace
```nim
proc linspace*(x1, x2: float, N: int): seq[float]
```
linspace creates a seq of N evenly spaced points between two numbers x1 and x2. 
```nim
echo linspace(0.0, 10.0, 11)
```
```nim
@[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
```

### arange
```nim
proc arange*(x1, x2, dx: float, includeStart = true, includeEnd = false): seq[float]
```
arange creates a seq of float between x1 and x2 separated by dx. You can choose to include or exclude the start- and endpoint (unless the steps goes exactly to x2).
```nim
echo arange(0.0, 5.0, 0.5)
echo arange(0.0, 4.9, 0.5)
```
```nim
@[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
@[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
```

# Status
This is a hobby project of mine, so please keep that in mind. I have tried to cover most of the use-cases in the tests but there's always the risk that I have missed something. If you spot a bug (or something worse) please open an Issue. If you want to help improve this project I would very much appreciate it.

## Arraymancer support: Most should work
If you want to use Arraymancer with NumericalNim, most should work but I haven't tested all of it yet.

# TODO
- Very much!
- Comment and document code.
- Add more ODE integrators.
- Add more integration methods.
- Make the existing code more efficient and robust.
- Add parallelization of some kind to speed it up. `Vector` would probably benefit from it.
- More optimization methods!
- More interpolation methods (especially multivariate).
