# NumericalNim

![Monthly Test](https://github.com/SciNim/numericalnim/actions/workflows/ci.yml/badge.svg?event=schedule)
<a href="https://matrix.to/#/#nim-science:envs.net">
  <img src="https://img.shields.io/static/v1?message=join%20chat&color=blue&label=nim-science&logo=matrix&logoColor=gold&style=flat-square&.svg"/>
</a>
<a href="https://discord.gg/f5hA9UK3dY">
  <img src="https://img.shields.io/discord/371759389889003530?color=blue&label=nim-science&logo=discord&logoColor=gold&style=flat-square&.svg"/>
</a>

NumericalNim is a collection of numerical methods written in Nim. Currently it has support for integration, optimization, curve-fitting, interpolation and ODEs. It can operate on floats and custom structures, such as vectors and tensors (if they support a set of operators).


# Table of contents

<!-- TOC -->

- [Documentation & Tutorials](#documentation--tutorials)
- [Installation](#installation)
- [ODE](#ode)
  - [Tutorials](#tutorials)
  - [The integrators](#the-integrators)
- [Integration](#integration)
  - [Tutorials](#tutorials)
  - [The methods](#the-methods)
- [Optimization](#optimization)
  - [Tutorials:](#tutorials)
  - [One Dimensional optimization methods](#one-dimensional-optimization-methods)
  - [Multidimensional optimization methods](#multidimensional-optimization-methods)
- [Curve fitting](#curve-fitting)
  - [Tutorials](#tutorials)
  - [Levenberg-Marquardt](#levenberg-marquardt)
- [Interpolation](#interpolation)
  - [Tutorials](#tutorials)
  - [Natural Cubic Splines](#natural-cubic-splines)
  - [Cubic Hermite Splines](#cubic-hermite-splines)
  - [Radial basis function interpolation](#radial-basis-function-interpolation)
- [Utils](#utils)
  - [Documentation](#documentation)
  - [linspace](#linspace)
  - [arange](#arange)
  - [meshgrid](#meshgrid)
  - [chi2](#chi2)
- [Tips and tricks](#tips-and-tricks)
  - [Suggested compilation flags](#suggested-compilation-flags)
  - [Using enums as keys for NumContext](#using-enums-as-keys-for-numcontext)
- [Status](#status)

<!-- /TOC -->

# Documentation & Tutorials
- NumericalNim's [documentation](https://scinim.github.io/numericalnim/)
- SciNim's [getting-started](scinim.github.io/getting-started/) site has a bunch of tutorials. Specific tutorial are linked below in their respective sections.

# Installation
Install NumericalNim using Nimble:

`nimble install numericalnim`

# ODE
## Tutorials
- ODE tutorial in SciNim getting-started: https://scinim.github.io/getting-started/numerical_methods/ode.html
- [Documentation](https://scinim.github.io/numericalnim/numericalnim/ode.html)

## The integrators
These are the implemented ODE integrators:

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
- `Vern65` - Verner's "most efficient" 6th order adaptive timestep method. (https://www.sfu.ca/~jverner/)


### Dense Output
All integrators support dense output using a 3rd order Hermite interpolant. Method specific interpolants may be added in the future.


# Integration
## Tutorials
- Integration tutorial in SciNim getting-started: https://scinim.github.io/getting-started/numerical_methods/integration1d.html
- [Documentation](https://scinim.github.io/numericalnim/numericalnim/integrate.html)

## The methods
- `adaptiveGauss` - Uses Gauss-Kronrod Quadrature to adaptivly integrate function. This is the recommended proc to use! Typically this is both the fastest and most accurate of the methods. This is the only one of the methods that supports using `Inf` and `-Inf` as integration limits.
- `gaussQuad` - Uses Gauss-Legendre Quadrature to integrate functions. Choose between 20 different accuracies by setting how many function evaluations should be made on each subinterval with the `nPoints` parameter (1 - 20 is valid options).
- `trapz` - Uses the trapezoidal rule to integrate both functions and discrete points. 2nd order method.
- `simpson` - Uses Simpson's rule to integrate both functions and discrete points. 4th order method.
- `adaptiveSimpson` - Uses a adaptive Simpson's rule to subdivide the integration interval in finer pieces where the integrand is changing a lot and wider pieces in intervals where it doesn't change much. This allows it to perform the integral efficiently and still have accuracy. The error is approximated using Richardson extrapolation. So technically the values it outputs are actually not Simpson's, but Booles' method.
- `romberg` - Uses Romberg integration to integrate both functions and discrete points. __Note:__ If discrete points are provided they must be equally spaced and the number of points must be of the form 2^k + 1 ie 3, 5, 9, 17, 33, 65, 129 etc.
- `cumtrapz` - Uses the trapezoidal rule to integrate both functions and discrete points but it outputs a seq of the integral values at provided x-values. 
- `cumsimpson` - Uses Simpson's rule to integrate both functions and discrete points but it outputs a seq of the integral values at provided x-values.



# Optimization
## Tutorials:
- Optimization tutorial in SciNim getting-started: https://scinim.github.io/getting-started/numerical_methods/optimization.html
- [Documentation](https://scinim.github.io/numericalnim/numericalnim/optimize.html)

## One Dimensional optimization methods
- `steepest_descent` - Basic method for local minimum finding.
- `conjugate_gradient` - iterative implementation of solving Ax = b
- `newtons` - Newton-Raphson implementation for 1D functions.
- `secant` - The secant method for 1D functions.

## Multidimensional optimization methods
- `lbfgs`: Limited-memory BFGS, a lighter version of BFGS. This is the recommended method
for larger problems.
- `bfgs`: Broyden–Fletcher–Goldfarb–Shanno algorithm for optimization. Quasi-Newton method.
- `newton`: The classic Newton method. Fast for small problems but struggles with bigger ones.
- `steepestDescent`: The basic gradient descent method. Only use this if you have specific reasons to, as it is slower than the others for almost all problems.


# Curve fitting
## Tutorials
- Curve-fitting tutorial in SciNim getting-started: https://scinim.github.io/getting-started/numerical_methods/curve_fitting.html
- [Documentation](https://scinim.github.io/numericalnim/numericalnim/optimize.html)

## Levenberg-Marquardt
Levenberg-Marquardt (`levmarq`) is an algorithm for solving non-linear least squares problems,
like curve fitting. You provide the data points (x- & y-values), and optionally the errors in your data, along with a function describing the curve you want to fit
and a starting guess of the parameters.
`levmarq` will then find a (locally) optimal solution that minimizes the squared error
(or the χ² if you provided errors). The uncertainties of the parameters can be approximated using `paramUncertainties`. 


# Interpolation
## Tutorials
- Interpolation tutorial in SciNim getting-started: https://scinim.github.io/getting-started/numerical_methods/interpolation.html
- [Interpolation Documentation](https://scinim.github.io/numericalnim/numericalnim/interpolate.html)
- [RBF Documentation](https://scinim.github.io/numericalnim/numericalnim/rbf.html)

## Natural Cubic Splines
Cubic splines are piecewise polynomials of degree 3 ie. it is defined differently on different intervals. It passes through all the supplied points and has a continuos derivative. To find which interval a certain x-value is in we use a binary search algorithm to find it instead of looping over all intervals one after another.  

## Cubic Hermite Splines
Cubic Hermite Splines are piecewise polynomials of degree 3, the same as Natural Cubic Splines. The difference is that we can pass the the derivative at each point as well as the function value. If the derivatives are not passed, a three-point finite difference will be used instead but this will not give as accurate results compared to with derivatives. It may be better to use Natural Cubic Splines then instead. The advantage Hermite Splines have over Natural Cubic Splines in NumericalNim is that it can handle other types of y-values than floats. For example if you want to interpolate data (dependent on one variable) in Arraymancer `Tensor`s you can do it by passing those as a `seq[Tensor]`. Hermite Splines' main mission in NumericalNim is to interpolate data points you get from solving ODEs as both the function value and the derivative is known at every point in time. 

## Radial basis function interpolation

# Utils
I have included a few handy tools in `numericalnim/utils`.
## Documentation
- [Documentation](https://scinim.github.io/numericalnim/numericalnim/utils.html)

## linspace
`linspace` creates a seq of N evenly spaced points between two numbers x1 and x2. 
```nim
echo linspace(0.0, 10.0, 11)
```
```nim
@[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
```

## arange
`arange` creates a seq of float between x1 and x2 separated by dx. You can choose to include or exclude the start- and endpoint (unless the steps goes exactly to x2).
```nim
echo arange(0.0, 5.0, 0.5)
echo arange(0.0, 4.9, 0.5)
```
```nim
@[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
@[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
```

## meshgrid
Meshgrid allows you to create a Tensor containing the points in an N-dimensional grid.

## chi2
Calculates the fit of a curve to data points.

# Tips and tricks

## Suggested compilation flags

There are many optimizations the C/C++ compiler can do but doesn't do by default.
These can be activated by passing flags to the underlying compiler with `-t:<flag>`.
These can be both of the [SIMD](https://en.wikipedia.org/wiki/SIMD) variety where
multiple operations are done at the same time, or fast-math which breaks the standards of floating point arithmetics to exchange a small bit of accuracy for performance. 

To enable these, nim has to pass some flags to the C compiler.
Below is a small table, listing the flags to add when compiling with `nim c` for some widely used compilers:

Compiler     | Flags
------------ | -----
clang        | `-t:-march=native -t:-ffast-math`
gcc          | `-t:-march=native -t:-ffast-math`
icc          | `-t:-march=core-avx2 -t:-fast`
msvc         | `-t:arch:AVX2 -t:fp:fast`

## Using enums as keys for `NumContext`
If you want to avoid KeyErrors regarding mistyped keys, you can use enums for the keys instead. The enum value will be converted to a string internally so there is no constraint that all keys must be from the same enum. Here is one example of how to use it:
```nim
type
  MyKeys = enum
    key1
    key2
    key3
var ctx = newNumContext[float]()
ctx[key1] = 3.14
ctx[key2] = 6.28
```


# Status
This is a hobby project of mine, so please keep that in mind. I have tried to cover most of the use-cases in the tests but there's always the risk that I have missed something. If you spot a bug (or something worse) please open an Issue. If you want to help improve this project I would very much appreciate it.
