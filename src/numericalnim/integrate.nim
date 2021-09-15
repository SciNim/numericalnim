import math, algorithm, sequtils
import
    ./utils,
    ./common/commonTypes
import arraymancer

from ./interpolate import InterpolatorType, newHermiteSpline

type
    IntervalType[T] = object
        lower, upper: float # integration bounds
        error: float # estimated error for current interval
        value: T # estimated value for integral over current interval
    IntervalList[T] = object
        list: seq[IntervalType[T]] # contains all the intervals sorted from smallest to largest error


# N: #intervals
proc trapz*[T](f: NumContextProc[T], xStart, xEnd: float,
               N = 500, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using the trapezoidal rule.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - N: The number of subintervals to divide the integration interval into.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using the trapezoidal rule.
    if N < 1:
        raise newException(ValueError, "N must be an integer >= 1")
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    let dx = (xEnd - xStart)/N.toFloat
    result = (f(xStart, ctx) + f(xEnd, ctx)) / 2.0
    for i in 1 .. N - 1:
        result += f(xStart + dx * i.toFloat, ctx)
    result *= dx

proc trapz*[T](Y: openArray[T], X: openArray[float]): T =
    ## Calculate the integral of f using the trapezoidal rule from a set values.
    ##
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The integral evaluated from the smallest to the largest value in X
    ##     calculated using the trapezoidal rule.
    let (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
    for i in 0 .. xSorted.high - 1:
        #                ( x_i+1         -    x_i)        *   (y_i+1         +    y_i)
        result += 0.5 * (xSorted[i+1] - xSorted[i]) * (ySorted[i+1] + ySorted[i])

# discrete points
proc cumtrapz*[T](Y: openArray[T], X: openArray[float]): seq[T] =
    ## Calculate the cumulative integral of f using the trapezoidal rule from a set of values.
    ##
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The cumulative integral evaluated from the smallest to the largest
    ##     value in X calculated using the trapezoidal rule.
    let (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
    result.add(ySorted[0] - ySorted[0]) # get the right kind of zero
    var integral: T = ySorted[0] - ySorted[0]
    for i in 0 .. xSorted.high - 1:
        integral += 0.5 * (xSorted[i+1] - xSorted[i]) * (ySorted[i+1] + ySorted[i])
        result.add(integral)

# function values calculated according to the dx and then interpolated
proc cumtrapz*[T](f: NumContextProc[T], X: openArray[float],
                  ctx: NumContext[T] = nil, dx = 1e-5): seq[T] =
    ## Calculate the cumulative integral of f using the trapezoidal rule at the points in X.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - X: The x-values of the returned values.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##   - dx: The step length to use when integrating.
    ##
    ## Returns:
    ##   - The value of the integral of f from the smallest to the largest
    ##     value of X calculated using the trapezoidal rule.
    var
        times: seq[float]
        dy, y: seq[T]
        dyTemp, dyPrev: T
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    var t = min(X)
    let tEnd = max(X) + 1.0 # make sure to get the endpoint as well.
    dyTemp = f(t, ctx)
    var integral = dyTemp - dyTemp # get the right kind of zero
    times.add(t)
    dy.add(dyTemp)
    y.add(integral)
    t += dx
    while t <= tEnd:
        dyPrev = dyTemp
        dyTemp = f(t, ctx)
        integral += 0.5 * dx * (dyPrev + dyTemp)
        times.add(t)
        dy.add(dyTemp)
        y.add(integral)
        t += dx
    result = hermiteInterpolate(X, times, y, dy)


proc simpson*[T](f: NumContextProc[T], xStart, xEnd: float,
                 N = 500, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using Simpson's rule.
    ##
    ## Input:
    ##   - f: the function that is integrated. 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - N: The number of subintervals to divide the integration interval into.
    ##     Must be 2 or greater.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using Simpson's rule.
    if N < 2:
        raise newException(ValueError, "N must be an integer >= 2")
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    let dx = (xEnd - xStart)/N.toFloat
    var N = N
    var xStart = xStart
    result = f(xStart, ctx) - f(xStart, ctx) # initialize to right kind of zero
    if N mod 2 != 0:
        result += 3.0 / 8.0 * dx * (f(xStart, ctx) +
                                    3.0 * f(xStart + dx, ctx) +
                                    3.0 * f(xStart + 2.0 * dx, ctx) +
                                    f(xStart + 3.0 * dx, ctx))
        xStart = xStart + 3.0 * dx
        N = N - 3
        if N == 0:
            return result
    var resultTemp = f(xStart, ctx) + f(xEnd, ctx)
    var res1 = f(xStart, ctx) - f(xStart, ctx) # initialize to right kind of zero
    var res2 = res1.clone()
    for j in 1 .. (N / 2 - 1).toInt:
        res1 += f(xStart + dx * 2.0 * j.toFloat, ctx)
    for j in 1 .. (N / 2).toInt:
        res2 += f(xStart + dx * (2.0 * j.toFloat - 1.0), ctx)

    resultTemp += 2.0 * res1 + 4.0 * res2
    resultTemp *= dx / 3.0
    result += resultTemp

proc simpson*[T](Y: openArray[T], X: openArray[float]): T =
    ## Calculate the integral of f using Simpson's rule from a set of values.
    ##
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The integral evaluated from the smallest to the largest value in X
    ##     calculated using Simpson's rule.
    var (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
    var N = xSorted.len
    if N < 3:
        raise newException(ValueError, "X and Y must have at least 3 elements to perform Simpson")
    var alpha, beta, eta: float
    if N mod 2 == 0:
        let lastIndex = xSorted.high
        let h1 = xSorted[lastIndex - 1] - xSorted[lastIndex - 2]
        let h2 = xSorted[lastIndex] - xSorted[lastIndex - 1]
        alpha = (2.0 * h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * (h1 + h2))
        beta = (h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * h1)
        eta = -(h2 ^ 3) / (6.0 * h1 * (h1 + h2))
        result = eta * ySorted[lastIndex - 2] + beta * ySorted[lastIndex - 1] + alpha * ySorted[lastIndex]
        xSorted = xSorted[0 ..< lastIndex]
        ySorted = ySorted[0 ..< lastIndex]
        N -= 1
    for i in 0 ..< ((N-1)/2).toInt:
        let h1 = xSorted[2*i + 1] - xSorted[2*i]
        let h2 = xSorted[2*i + 2] - xSorted[2*i + 1]
        alpha = (2.0 * h2 ^ 3 - h1 ^ 3 + 3.0 * h1 * h2 ^ 2) / (6.0 * h2 * (h2 + h1))
        beta = (h2 ^ 3 + h1 ^ 3 + 3.0 * h1 * h2 * (h2 + h1)) / (6.0 * h2 * h1)
        eta = (2.0 * h1 ^ 3 - h2 ^ 3 + 3.0 * h2 * h1 ^ 2) / (6.0 * h1 * (h2 + h1))
        result += alpha * ySorted[2*i + 2] + beta * ySorted[2*i + 1] + eta * ySorted[2*i]

proc adaptiveSimpson*[T](f: NumContextProc[T], xStart, xEnd: float,
                         tol = 1e-8, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using an adaptive Simpson's rule.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using
    ##     an adaptive Simpson's rule.
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    let zero = f(xStart, ctx) - f(xStart, ctx)
    let value1 = simpson(f, xStart, xEnd, N = 2, ctx = ctx)
    let value2 = simpson(f, xStart, xEnd, N = 4, ctx = ctx)
    let error = (value2 - value1)/15
    var tol = tol
    if tol < 1e-15:
        tol = 1e-15
    if calcError(error, zero) < tol or abs(xEnd - xStart) < 1e-5:
        return value2 + error
    let m = (xStart + xEnd) / 2.0
    let newtol = tol / 2.0
    let left = adaptiveSimpson(f, xStart, m, tol = newtol, ctx = ctx)
    let right = adaptiveSimpson(f, m, xEnd, tol = newtol, ctx = ctx)
    return left + right

proc internal_adaptiveSimpson[T](f: NumContextProc[T], xStart, xEnd: float,
                         tol: float, ctx: NumContext[T], reused_points: array[3, T]): T =
    let zero = reused_points[0] - reused_points[0]
    let dx1 = (xEnd - xStart) / 2
    let dx2 = (xEnd - xStart) / 4 
    let value1 = dx1 / 3 * (reused_points[0] + 4*reused_points[1] + reused_points[2])
    let point2 = f(xStart + dx2, ctx)
    let point4 = f(xStart + 3*dx2, ctx)
    let value2 = dx2 / 3 * (reused_points[0] + 4*point2 + 2*reused_points[1] + 4*point4 + reused_points[2])
    let error = (value2 - value1)/15
    if calcError(error, zero) < tol or abs(xEnd - xStart) < 1e-6:
        return value2 + error
    let m = (xStart + xEnd) / 2.0
    let newtol = tol / 2.0
    let left = internal_adaptiveSimpson(f, xStart, m, tol = newtol, ctx = ctx, [reused_points[0], point2, reused_points[1]])
    let right = internal_adaptiveSimpson(f, m, xEnd, tol = newtol, ctx = ctx, [reused_points[1], point4, reused_points[2]])
    return left + right

proc adaptiveSimpson2*[T](f: NumContextProc[T], xStart, xEnd: float,
                         tol = 1e-8, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using an adaptive Simpson's rule.
    ##
    ## Input:
    ##   - f: the function that is integrated. 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using
    ##     an adaptive Simpson's rule.
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    var tol = tol
    if tol < 1e-15:
        tol = 1e-15
    let dx = (xEnd - xStart) / 2
    let init_points: array[3, T] = [f(xStart, ctx), f(xStart + dx, ctx), f(xEnd, ctx)]
    return internal_adaptiveSimpson(f, xStart, xEnd, tol, ctx, init_points)


proc cumsimpson*[T](Y: openArray[T], X: openArray[float]): seq[T] =
    ## Calculate the cumulative integral of f using Simpson's rule from a set of values.
    ##
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The cumulative integral evaluated from the smallest to the largest
    ##     value in X calculated using Simpson's rule.
    var (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
    var N = xSorted.len
    var alpha, beta, eta: float
    var y, dy: seq[T]
    var xs: seq[float]
    var evenN = false
    if N < 3:
        raise newException(ValueError, "X and Y must have at least 3 elements to perform Simpson, use cumtrapz instead")
    if N mod 2 == 0:
        evenN = true
        N -= 1
    var integral = ySorted[0] - ySorted[0] # get the right kind of zero
    y.add(integral)
    dy.add(ySorted[0])
    xs.add(xSorted[0])
    for i in 0 ..< ((N-1)/2).toInt:
        let h1 = xSorted[2*i + 1] - xSorted[2*i]
        let h2 = xSorted[2*i + 2] - xSorted[2*i + 1]
        alpha = (2.0 * h2 ^ 3 - h1 ^ 3 + 3.0 * h1 * h2 ^ 2) / (6.0 * h2 * (h2 + h1))
        beta = (h2 ^ 3 + h1 ^ 3 + 3.0 * h1 * h2 * (h2 + h1)) / (6.0 * h2 * h1)
        eta = (2.0 * h1 ^ 3 - h2 ^ 3 + 3.0 * h2 * h1 ^ 2) / (6.0 * h1 * (h2 + h1))
        integral += alpha * ySorted[2*i + 2] + beta * ySorted[2*i + 1] + eta * ySorted[2*i]
        y.add(integral)
        dy.add(ySorted[2*i+2])
        xs.add(xSorted[2*i+2])
    if evenN:
        let lastIndex = xSorted.high
        let h1 = xSorted[lastIndex - 1] - xSorted[lastIndex - 2]
        let h2 = xSorted[lastIndex] - xSorted[lastIndex - 1]
        alpha = (2.0 * h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * (h1 + h2))
        beta = (h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * h1)
        eta = -(h2 ^ 3) / (6.0 * h1 * (h1 + h2))
        integral += eta * ySorted[lastIndex - 2] + beta * ySorted[lastIndex - 1] + alpha * ySorted[lastIndex]
        y.add(integral)
        dy.add(ySorted[xSorted.high])
        xs.add(xSorted[xSorted.high])
    # because simpson uses multiple input-points per integral-point we must to get the integral at all input-points 
    result = hermiteInterpolate(X, xs, y, dy)

proc cumsimpson*[T](f: NumContextProc[T], X: openArray[float],
                    ctx: NumContext[T] = nil, dx = 1e-5): seq[T] =
    ## Calculate the cumulative integral of f using Simpson's rule.
    ##
    ## Input:
    ##   - f: the function that is integrated. 
    ##   - X: The x-values of the returned values.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##   - dx: The step length to use when integrating.
    ##
    ## Returns:
    ##   - The value of the integral of f from the smallest to the largest value
    ##     of X calculated using Simpson's rule.
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    var dy: seq[T]
    let t = linspace(min(X), max(X), ((max(X) - min(X)) / dx).toInt + 2)
    for x in t:
        dy.add(f(x, ctx))
    let ys = cumsimpson(dy, t)
    result = hermiteInterpolate(X, t, ys, dy)

proc romberg*[T](f: NumContextProc[T], xStart, xEnd: float,
                 depth = 8, tol = 1e-8, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using Romberg Integration.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - depth: The maximum depth of the Richardson Extrapolation.
    ##   - tol: The error tolerance that must be satisfied.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using Romberg integration.
    if depth < 2:
        raise newException(ValueError, "depth must be 2 or greater")
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    var values: seq[seq[T]]
    var firstIteration: seq[T]
    for i in 0 ..< depth:
        firstIteration.add(trapz(f, xStart, xEnd, N = 2 ^ i, ctx = ctx))
    values.add(firstIteration)
    for i in 0 ..< depth - 1:
        var newValues: seq[T]
        for j in 0 .. (values[i].high - 1):
            let val1 = values[i][j]
            let val2 = values[i][j + 1]
            newValues.add((4.0 ^ (i + 1) * val2 - val1) / (4.0 ^ (i + 1) - 1))
        let lastIteration = values[values.high]
        let error = calcError(newValues[newValues.high], lastIteration[lastIteration.high])
        if error < tol:
            return newValues[newValues.high]
        values.add(newValues)
    result = values[values.high][0]

proc romberg*[T](Y: openArray[T], X: openArray[float]): T =
    ## Calculate the integral of f using Romberg Integration from a set of values.
    ##
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The integral evaluated from the smallest to the largest value in X
    ##     calculated using Romberg Integration.
    let (xSorted, ySorted) = sortAndTrimDataset(@X, @Y)
    let N = xSorted.len
    if N < 3:
        raise newException(ValueError, "X and Y must have at least 3 elements, use trapz instead")
    elif ceil(log2((N - 1).toFloat)) != log2((N - 1).toFloat):
        raise newException(ValueError, "The length of X and Y must be of the form 2^n + 1, n is integer")
    var values: seq[seq[T]]
    var firstIteration: seq[T]
    let depth = log2((N - 1).toFloat).toInt + 1
    for i in 0 ..< depth:
        let step = ((N - 1) / (2 ^ i)).toInt
        var vals: seq[T]
        var xs: seq[float]
        var x = 0
        for j in 0 .. 2 ^ i:
            xs.add(xSorted[x])
            vals.add(ySorted[x])
            x += step
        firstIteration.add(trapz(vals, xs))
    values.add(firstIteration)
    for i in 0 ..< depth - 1:
        var newValues: seq[T]
        for j in 0 .. (values[i].high - 1):
            let val1 = values[i][j]
            let val2 = values[i][j + 1]
            newValues.add((4.0 ^ (i + 1) * val2 - val1) / (4.0 ^ (i + 1) - 1))
        values.add(newValues)
    result = values[values.high][0]


proc getGaussLegendreWeights(nPoints: int): tuple[nodes: seq[float], weights: seq[float]] {.inline.} =
    assert(0 < nPoints and nPoints <= 20, "nPoints must be an integer between 1 and 20.")
    const gaussWeights: array[1..20, (seq, seq)] = [
        (@[0.0],
         @[2.0]),
        (@[-0.5773502691896257645092, 0.5773502691896257645092],
         @[1.0, 1.0]),
        (@[-0.7745966692414833770359, 0.0, 0.7745966692414833770359],
         @[0.5555555555555555555556, 0.8888888888888888888889, 0.555555555555555555556]),
        (@[-0.861136311594052575224, -0.3399810435848562648027, 0.3399810435848562648027, 0.861136311594052575224],
         @[0.3478548451374538573731, 0.6521451548625461426269, 0.6521451548625461426269, 0.3478548451374538573731]),
        (@[0.90617984593866385,0.538469310105683,0,-0.538469310105683,-0.90617984593866385],
         @[0.23692688505618911,0.47862867049936625,0.56888888888888889,0.47862867049936625,0.23692688505618911]),
        (@[0.932469514203152,0.66120938646626448,0.23861918608319693,-0.23861918608319693,-0.66120938646626448,-0.932469514203152],
         @[0.1713244923791705,0.36076157304813866,0.467913934572691,0.467913934572691,0.36076157304813866,0.1713244923791705]),
        (@[0.94910791234275838,0.74153118559939446,0.40584515137739713,0,
           -0.40584515137739713,-0.74153118559939446,-0.94910791234275838],
         @[0.12948496616886965,0.27970539148927676,0.38183005050511903,0.41795918367346935,
           0.38183005050511903,0.27970539148927676,0.12948496616886965]),
        (@[0.9602898564975364,0.79666647741362673,0.525532409916329,0.18343464249564984,
           -0.18343464249564984,-0.525532409916329,-0.79666647741362673,-0.9602898564975364],
         @[0.10122853629037681,0.22238103445337443,0.31370664587788732,0.3626837833783621,
           0.3626837833783621,0.31370664587788732,0.22238103445337443,0.10122853629037681]),
        (@[0.96816023950762609,0.83603110732663577,0.61337143270059036,0.32425342340380897,
           0,-0.32425342340380897,-0.61337143270059036,-0.83603110732663577,-0.96816023950762609],
         @[0.081274388361574634,0.1806481606948574,0.26061069640293555,0.31234707704000292,
           0.33023935500125978,0.31234707704000292,0.26061069640293555,0.1806481606948574,0.081274388361574634]),
        (@[0.97390652851717174,0.86506336668898465,0.67940956829902444,0.43339539412924721,0.14887433898163116,
           -0.14887433898163116,-0.43339539412924721,-0.67940956829902444,-0.86506336668898465,-0.97390652851717174],
         @[0.066671344308688027,0.14945134915058056,0.21908636251598207,0.26926671930999618,0.29552422471475293,
           0.29552422471475293,0.26926671930999618,0.21908636251598207,0.14945134915058056,0.066671344308688027]),
        (@[0.97822865814605686,0.88706259976809543,0.73015200557404936,0.5190961292068117,0.2695431559523449,
           0,-0.2695431559523449,-0.5190961292068117,
           -0.73015200557404936,-0.88706259976809543,-0.97822865814605686],
         @[0.055668567116173538,0.12558036946490408,0.186290210927734,0.23319376459199023,0.26280454451024671,
           0.27292508677790062,0.26280454451024671,0.23319376459199023,
           0.186290210927734,0.12558036946490408,0.055668567116173538]),
        (@[0.98156063424671913,0.90411725637047491,0.76990267419430469,
           0.58731795428661737,0.36783149899818013,0.12523340851146886,
           -0.12523340851146886,-0.36783149899818013,-0.58731795428661737,
           -0.76990267419430469,-0.90411725637047491,-0.98156063424671913],
         @[0.0471753363865118,0.10693932599531812,0.16007832854334605,
           0.20316742672306581,0.23349253653835478,0.24914704581340286,
           0.24914704581340286,0.23349253653835478,0.20316742672306581,
           0.16007832854334605,0.10693932599531812,0.0471753363865118]),
        (@[0.98418305471858814,0.91759839922297792,0.80157809073330988,
           0.64234933944034012,0.44849275103644692,0.23045831595513477,
           0,-0.23045831595513477,-0.44849275103644692,
           -0.64234933944034012,-0.80157809073330988,-0.91759839922297792,-0.98418305471858814],
         @[0.040484004765315815,0.092121499837728438,0.13887351021978714,
           0.17814598076194568,0.20781604753688834,0.22628318026289709,
           0.2325515532308739,0.22628318026289709,0.20781604753688834,
           0.17814598076194568,0.13887351021978714,0.092121499837728438,0.040484004765315815]),
        (@[0.98628380869681243,0.92843488366357363,0.827201315069765,
           0.68729290481168537,0.5152486363581541,0.31911236892788969,0.10805494870734367,
           -0.10805494870734367,-0.31911236892788969,-0.5152486363581541,
           -0.68729290481168537,-0.827201315069765,-0.92843488366357363,-0.98628380869681243],
         @[0.035119460331752,0.080158087159760041,0.1215185706879032,
           0.15720316715819357,0.18553839747793785,0.20519846372129569,0.21526385346315768,
           0.21526385346315768,0.20519846372129569,0.18553839747793785,
           0.15720316715819357,0.1215185706879032,0.080158087159760041,0.035119460331752]),
        (@[0.98799251802048538,0.937273392400706,0.84820658341042732,
           0.72441773136017007,0.57097217260853883,0.39415134707756344,0.20119409399743454,
           0,-0.20119409399743454,-0.39415134707756344,
           -0.57097217260853883,-0.72441773136017007,-0.84820658341042732,-0.937273392400706,-0.98799251802048538],
         @[0.030753241996116634,0.070366047488108138,0.1071592204671718,
           0.13957067792615424,0.16626920581699395,0.18616100001556207,0.19843148532711161,
           0.20257824192556126,0.19843148532711161,0.18616100001556207,
           0.16626920581699395,0.13957067792615424,0.1071592204671718,0.070366047488108138,0.030753241996116634]),
        (@[0.98940093499164994,0.9445750230732326,0.86563120238783187,0.75540440835500311,
           0.61787624440264377,0.45801677765722731,0.28160355077925892,0.095012509837637482,
           -0.095012509837637482,-0.28160355077925892,-0.45801677765722731,-0.61787624440264377,
           -0.75540440835500311,-0.86563120238783187,-0.9445750230732326,-0.98940093499164994],
         @[0.027152459411754058,0.062253523938647776,0.0951585116824929,0.12462897125553393,
           0.14959598881657682,0.16915651939500256,0.18260341504492361,0.18945061045506845,
           0.18945061045506845,0.18260341504492361,0.16915651939500256,0.14959598881657682,
           0.12462897125553393,0.0951585116824929,0.062253523938647776,0.027152459411754058]),
        (@[0.99057547531441736,0.95067552176876768,0.8802391537269858,0.78151400389680137,
           0.65767115921669062,0.51269053708647694,0.35123176345387636,0.17848418149584783,
           0,-0.17848418149584783,-0.35123176345387636,-0.51269053708647694,
           -0.65767115921669062,-0.78151400389680137,-0.8802391537269858,-0.95067552176876768,-0.99057547531441736],
         @[0.024148302868547931,0.055459529373987133,0.085036148317179164,0.11188384719340388,
           0.13513636846852545,0.15404576107681039,0.1680041021564499,0.17656270536699264,
           0.17944647035620653,0.17656270536699264,0.1680041021564499,0.15404576107681039,
           0.13513636846852545,0.11188384719340388,0.085036148317179164,0.055459529373987133,0.024148302868547931]),
        (@[0.991565168420931,0.9558239495713976,0.89260246649755581,0.80370495897252314,
           0.69168704306035322,0.55977083107394754,0.41175116146284263,0.25188622569150554,
           0.084775013041735292,-0.084775013041735292,-0.25188622569150554,-0.41175116146284263,
           -0.55977083107394754,-0.69168704306035322,-0.80370495897252314,
           -0.89260246649755581,-0.9558239495713976,-0.991565168420931],
         @[0.021616013526483353,0.049714548894969283,0.0764257302548889,0.10094204410628704,
           0.12255520671147835,0.1406429146706506,0.15468467512626538,0.1642764837458327,
           0.16914238296314352,0.16914238296314352,0.1642764837458327,0.15468467512626538,
           0.1406429146706506,0.12255520671147835,0.10094204410628704,
           0.0764257302548889,0.049714548894969283,0.021616013526483353]),
        (@[0.99240684384358424,0.96020815213483,0.903155903614818,0.82271465653714282,
           0.72096617733522939,0.6005453046616811,0.464570741375961,0.31656409996362989,
           0.16035864564022534,0,-0.16035864564022534,-0.31656409996362989,
           -0.464570741375961,-0.6005453046616811,-0.72096617733522939,-0.82271465653714282,
           -0.903155903614818,-0.96020815213483,-0.99240684384358424],
         @[0.01946178822972643,0.044814226765699405,0.06904454273764106,0.091490021622449916,
           0.11156664554733405,0.12875396253933627,0.14260670217360666,0.15276604206585967,
           0.1589688433939544,0.16105444984878367,0.1589688433939544,0.15276604206585967,
           0.14260670217360666,0.12875396253933627,0.11156664554733405,0.091490021622449916,
           0.06904454273764106,0.044814226765699405,0.01946178822972643]),
        (@[0.99312859918509488,0.96397192727791392,0.912234428251326,0.83911697182221889,
           0.7463319064601508,0.63605368072651514,0.51086700195082724,0.37370608871541955,
           0.22778585114164507,0.076526521133497338,-0.076526521133497338,-0.22778585114164507,
           -0.37370608871541955,-0.51086700195082724,-0.63605368072651514,-0.7463319064601508,
           -0.83911697182221889,-0.912234428251326,-0.96397192727791392,-0.99312859918509488],
         @[0.01761400713915167,0.04060142980038705,0.06267204833410904,0.083276741576704769,
           0.1019301198172405,0.11819453196151831,0.13168863844917658,0.14209610931838215,
           0.14917298647260382,0.152753387130726,0.152753387130726,0.14917298647260382,
           0.14209610931838215,0.13168863844917658,0.11819453196151831,0.1019301198172405,
           0.083276741576704769,0.06267204833410904,0.04060142980038705,0.01761400713915167])
        ]
    return gaussWeights[nPoints]

proc gaussQuad*[T](f: NumContextProc[T], xStart, xEnd: float,
                   N = 100, nPoints = 7, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using Gaussian Quadrature.
    ## Has 20 different sets of weights, ranging from 1 to 20 function evaluations per subinterval.
    ##
    ## Input:
    ##   - f: the function that is integrated. 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - N: The number of subintervals to divide the integration interval into.
    ##   - nPoints: The number of points to evaluate f at per interval.
    ##     Choose between 1 to 20 with increasing accuracy.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The integral evaluated from the xStart to xEnd calculated using Gaussian Quadrature.
    if N < 1:
        raise newException(ValueError, "N must be an integer >= 1")
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    let dx = (xEnd - xStart)/N.toFloat
    let (nodes, weights) = getGaussLegendreWeights(nPoints)
    let zero = f(nodes[0], ctx) - f(nodes[0], ctx)
    result = zero.clone() # set result to the right kind of zero
    var tempResult = zero.clone()
    var a, b: float
    for j in 0 ..< N:
        a = xStart + dx * j.toFloat
        b = a + dx
        let c1 = (b - a)/2.0
        let c2 = (a + b)/2.0
        tempResult = zero.clone()
        for i in 0 ..< nPoints:
            tempResult += weights[i] * f(c1 * nodes[i] + c2, ctx)
        tempResult *= c1
        result += tempResult

proc calcGaussKronrod[T](f: NumContextProc[T], xStart, xEnd: float, ctx: NumContext[T] = nil, 
                            lowOrderWeights, lowOrderNodes, highOrderCommonWeights, highOrderWeights, highOrderNodes: openArray[float]): (T, T) {.inline.} =
    var lowOrderResult, highOrderResult: T
    var savedFunctionValues = newSeq[T](lowOrderNodes.len)
    let c1 = (xEnd - xStart) / 2.0
    let c2 = (xStart + xEnd) / 2.0
    for i in 0 .. lowOrderNodes.high:
        savedFunctionValues[i] = f(c1 * lowOrderNodes[i] + c2, ctx)
    lowOrderResult = lowOrderWeights[0] * savedFunctionValues[0]
    for i in 1 .. lowOrderNodes.high:
        lowOrderResult += lowOrderWeights[i] * savedFunctionValues[i]
    lowOrderResult *= c1

    highOrderResult = highOrderCommonWeights[0] * savedFunctionValues[0]
    for i in 1 .. lowOrderNodes.high:
        highOrderResult += highOrderCommonWeights[i] * savedFunctionValues[i]
    for i in 0 .. highOrderNodes.high:
        highOrderResult += highOrderWeights[i] * f(c1 * highOrderNodes[i] + c2, ctx)
    highOrderResult *= c1
    result = (highOrderResult, lowOrderResult)


proc adaptiveGaussLocal*[T](f: NumContextProc[T],
                       xStart, xEnd: float, tol = 1e-8, ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using an locally adaptive Gauss-Kronrod Quadrature.
    ##
    ## Input:
    ##   - f: the function that is integrated. 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using
    ##     an adaptive Gauss-Kronrod Quadrature.
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()

    const lowOrderWeights = [0.0666713443086881375936, 0.1494513491505805931458, 0.219086362515982043996, 0.269266719309996355091, 0.2955242247147528701739,
                             0.2955242247147528701739, 0.2692667193099963550912, 0.2190863625159820439955, 0.1494513491505805931458, 0.0666713443086881375936] # weights for low order
    const lowOrderNodes = [-0.973906528517171720078, -0.8650633666889845107321, -0.6794095682990244062343, -0.4333953941292471907993, -0.1488743389816312108848,
                           0.1488743389816312108848, 0.4333953941292471907993, 0.6794095682990244062343, 0.865063366688984510732, 0.973906528517171720078] # nodes for low order
    const highOrderCommonWeights = [0.0325581623079647274788, 0.075039674810919952767, 0.1093871588022976418992, 0.134709217311473325928, 0.1477391049013384913748,
                                    0.1477391049013384913748, 0.134709217311473325928, 0.109387158802297641899, 0.075039674810919952767, 0.032558162307964727479] # weights for high order to use with lowOrderNodes
    const highOrderWeights = [0.0116946388673718742781, 0.0547558965743519960314, 0.093125454583697605535, 0.123491976262065851078, 0.142775938577060080797,
                              0.149445554002916905665, 0.1427759385770600807971, 0.123491976262065851078, 0.093125454583697605535, 0.05475589657435199603138, 0.0116946388673718742781] # weights for high order to use with highOrderNodes
    const highOrderNodes = [-0.9956571630258080807355, -0.9301574913557082260012, -0.7808177265864168970637, -0.562757134668604683339, -0.294392862701460198131,
                            0.0, 0.2943928627014601981311, 0.562757134668604683339, 0.7808177265864168970637, 0.9301574913557082260012, 0.9956571630258080807355] # nodes for high order

    let (highOrderResult, lowOrderResult) = calcGaussKronrod(f, xStart, xEnd, ctx, lowOrderWeights, lowOrderNodes, highOrderCommonWeights, highOrderWeights, highOrderNodes)

    let zero = f(xStart, ctx) - f(xStart, ctx)
    let error = highOrderResult - lowOrderResult
    if calcError(error, zero) < tol or abs(xEnd - xStart) < 1e-8:
        return highOrderResult
    let c1 = (xEnd - xStart) / 2.0
    let c2 = (xStart + xEnd) / 2.0
    let left = adaptiveGaussLocal(f, xStart, c2, tol = tol/2, ctx = ctx)
    let right = adaptiveGaussLocal(f, c2, xEnd, tol = tol/2, ctx = ctx)
    return left + right

# Intervals

proc cmpInterval*[T](interval1, interval2: IntervalType[T]): int =
    if interval1.error < interval2.error:
        return -1
    elif interval1.error > interval2.error:
        return 1
    else:
        return 0

proc insert*[T](intervalList: var IntervalList[T], el: IntervalType[T]) {.inline.} =
    intervalList.list.insert(el, intervalList.list.lowerBound(el, cmpInterval))

proc pop*[T](intervalList: var IntervalList[T]): IntervalType[T] {.inline.} =
    result = intervalList.list.pop()

template adaptiveGaussImpl(): untyped {.dirty.} =
    var ctx = ctx
    if ctx.isNil:
        ctx = newNumContext[T]()
    var f: (proc(x: float, ctx: NumContext[T]): T) = f_in
    var xStart: float = xStart_in
    var xEnd: float = xEnd_in
    var points_transformed = @initialPoints
    if xStart == -Inf and xEnd == Inf: # Done
        # Remember to scale supplied points to new interval
        f = proc(x: float, ctx: NumContext[T]): T = (f_in((1-x)/x, ctx) + f_in(-(1-x)/x, ctx)) / (x*x)
        # if x => 0: t = 1/(1+x)
        # elif x < 0: t = 1/(1-x)
        for i in 0 .. points_transformed.high:
            let x = points_transformed[i]
            if 0 <= x:
                points_transformed[i] = 1 / (1 + x)
            else:
                points_transformed[i] = 1 / (1 - x)
        xStart = 0.0
        xEnd = 1.0
        points_transformed.add(xStart)
        points_transformed.add(xEnd)
    elif  xStart == Inf and xEnd == -Inf: # Done
        f = proc(x: float, ctx: NumContext[T]): T = -(f_in((1-x)/x, ctx) + f_in(-(1-x)/x, ctx)) / (x*x)
        for i in 0 .. points_transformed.high:
            let x = points_transformed[i]
            if 0 <= x:
                points_transformed[i] = 1 / (1 + x)
            else:
                points_transformed[i] = 1 / (1 - x)
        xStart = 0.0
        xEnd = 1.0
        points_transformed.add(xStart)
        points_transformed.add(xEnd)
    elif xStart == -Inf: # Done
        f = proc(x: float, ctx: NumContext[T]): T = f_in(xEnd_in - (1-x)/x, ctx) / (x*x)
        for i in 0 .. points_transformed.high:
            let x = points_transformed[i]
            points_transformed[i] = 1 / (1 + xEnd_in - x)
        xStart = 0.0
        xEnd = 1.0
        points_transformed.add(xStart)
        points_transformed.add(xEnd)
    elif xStart == Inf: # Done
        f = proc(x: float, ctx: NumContext[T]): T = -f_in(xEnd_in + (1-x)/x, ctx) / (x*x)
        for i in 0 .. points_transformed.high:
            let x = points_transformed[i]
            points_transformed[i] = 1 / (1 + x - xEnd_in)
        xStart = 0.0
        xEnd = 1.0
        points_transformed.add(xStart)
        points_transformed.add(xEnd)
    elif xEnd == Inf: # Done
        f = proc(x: float, ctx: NumContext[T]): T = f_in(xStart_in + (1-x)/x, ctx) / (x*x)
        for i in 0 .. points_transformed.high:
            let x = points_transformed[i]
            points_transformed[i] = 1 / (1.0 + x - xStart_in)#xStart_in + (1-x) / x # we must inverse, this is x(t) not t(x)
        xStart = 0.0
        xEnd = 1.0
        points_transformed.add(xStart)
        points_transformed.add(xEnd)
    elif xEnd == -Inf: # Done
        f = proc(x: float, ctx: NumContext[T]): T = -f_in(xStart_in - (1-x)/x, ctx) / (x*x)
        for i in 0 .. points_transformed.high:
            let x = points_transformed[i]
            points_transformed[i] = 1 / (1 + xStart_in - x)
        xStart = 0.0
        xEnd = 1.0
        points_transformed.add(xStart)
        points_transformed.add(xEnd)
    else:
        if points_transformed.len == 0:
            points_transformed = @[xStart_in, xEnd_in]
        else:
            points_transformed = @[xStart_in, xEnd_in].concat(@initialPoints)
    if xStart < xEnd:
        points_transformed.sort(Ascending)
    else:
        points_transformed.sort(Descending)
    # Trim off values outside integration domain
    while points_transformed[0] != xStart:
            points_transformed.delete(0)
    while points_transformed[points_transformed.high] != xEnd:
        points_transformed.delete(points_transformed.high)
    # Remove duplicates, typically the endpoints
    points_transformed = deduplicate(points_transformed, isSorted = true)


    const lowOrderWeights = [0.0666713443086881375936, 0.1494513491505805931458, 0.219086362515982043996, 0.269266719309996355091, 0.2955242247147528701739,
                             0.2955242247147528701739, 0.2692667193099963550912, 0.2190863625159820439955, 0.1494513491505805931458, 0.0666713443086881375936] # weights for low order
    const lowOrderNodes = [-0.973906528517171720078, -0.8650633666889845107321, -0.6794095682990244062343, -0.4333953941292471907993, -0.1488743389816312108848,
                           0.1488743389816312108848, 0.4333953941292471907993, 0.6794095682990244062343, 0.865063366688984510732, 0.973906528517171720078] # nodes for low order
    const highOrderCommonWeights = [0.0325581623079647274788, 0.075039674810919952767, 0.1093871588022976418992, 0.134709217311473325928, 0.1477391049013384913748,
                                    0.1477391049013384913748, 0.134709217311473325928, 0.109387158802297641899, 0.075039674810919952767, 0.032558162307964727479] # weights for high order to use with lowOrderNodes
    const highOrderWeights = [0.0116946388673718742781, 0.0547558965743519960314, 0.093125454583697605535, 0.123491976262065851078, 0.142775938577060080797,
                              0.149445554002916905665, 0.1427759385770600807971, 0.123491976262065851078, 0.093125454583697605535, 0.05475589657435199603138, 0.0116946388673718742781] # weights for high order to use with highOrderNodes
    const highOrderNodes = [-0.9956571630258080807355, -0.9301574913557082260012, -0.7808177265864168970637, -0.562757134668604683339, -0.294392862701460198131,
                            0.0, 0.2943928627014601981311, 0.562757134668604683339, 0.7808177265864168970637, 0.9301574913557082260012, 0.9956571630258080807355] # nodes for high order
    
    var intervals = IntervalList[T](list: newSeqOfCap[IntervalType[T]](maxintervals))

    let (initHigh, initLow) = calcGaussKronrod(f, points_transformed[0], points_transformed[1], ctx, lowOrderWeights, lowOrderNodes, highOrderCommonWeights, highOrderWeights, highOrderNodes)
    let zero = initHigh - initHigh
    if xStart_in == xEnd_in:
            raise newException(ValueError, "xStart can't be the same as xEnd")

    let initError = calcError(initHigh - initLow, zero)
    let initInterval = IntervalType[T](lower: points_transformed[0], upper: points_transformed[1], error: initError, value: initHigh)
    intervals.insert(initInterval)
    var totalValue: T = initHigh
    var totalError: float = initError
    for i in 1 .. points_transformed.high - 1:
        let (initHigh, initLow) = calcGaussKronrod(f, points_transformed[i], points_transformed[i+1], ctx, lowOrderWeights, lowOrderNodes, highOrderCommonWeights, highOrderWeights, highOrderNodes)
        let initError = calcError(initHigh - initLow, zero)
        let initInterval = IntervalType[T](lower: points_transformed[i], upper: points_transformed[i+1], error: initError, value: initHigh)
        intervals.insert(initInterval)
        totalValue += initHigh
        totalError += initError
    var currentInterval: IntervalType[T]
    var middle, error: float
    var highValue, lowValue: T

    while totalError > tol and intervals.list.len < maxintervals:
        currentInterval = intervals.pop()
        totalError -= currentInterval.error
        totalValue -= currentInterval.value
        middle = (currentInterval.upper + currentInterval.lower) / 2
        # first half
        (highValue, lowValue) = calcGaussKronrod(f, currentInterval.lower, middle, ctx, lowOrderWeights, lowOrderNodes, highOrderCommonWeights, highOrderWeights, highOrderNodes)
        error = calcError(highValue - lowValue, zero)
        intervals.insert(IntervalType[T](lower: currentInterval.lower, upper: middle, error: error, value: highValue))
        totalError += error
        totalValue += highValue
        # second half
        (highValue, lowValue) = calcGaussKronrod(f, middle, currentInterval.upper, ctx, lowOrderWeights, lowOrderNodes, highOrderCommonWeights, highOrderWeights, highOrderNodes)
        error = calcError(highValue - lowValue, zero)
        intervals.insert(IntervalType[T](lower: middle, upper: currentInterval.upper, error: error, value: highValue))
        totalError += error
        totalValue += highValue

proc adaptiveGauss*[T](f_in: NumContextProc[T],
                       xStart_in, xEnd_in: float, tol = 1e-8, initialPoints: openArray[float] = @[], maxintervals: int = 10000,  ctx: NumContext[T] = nil): T =
    ## Calculate the integral of f using an globally adaptive Gauss-Kronrod Quadrature. Inf and -Inf can be used as integration limits.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - maxintervals: maximum numbers of intervals to divide integral in before stopping.
    ##   - initialPoints: A list of known difficult points (integrable singularities, discontinouities etc) that will be used as the inital interval boundaries.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using
    ##     an adaptive Gauss-Kronrod Quadrature.
    adaptiveGaussImpl()
    return totalValue

proc cumGaussSpline*[T](f_in: NumContextProc[T],
                       xStart_in, xEnd_in: float, tol = 1e-8, initialPoints: openArray[float] = @[], maxintervals: int = 10000, ctx: NumContext[T] = nil): InterpolatorType[T] =
    ## Calculate the cumulative integral of f using an globally adaptive Gauss-Kronrod Quadrature. Inf and -Inf can be used as integration limits.
    ## Returns a Hermite spline that can be evaluated at any point between xStart and xEnd.
    ## Important: because of the much higher order of the Gauss-Kronrod quadrature (order 21) compared to the interpolating Hermite spline (order 3) you have to give it a large amount of initialPoints.
    ## Otherwise it may only use a couple of points which gives quite a bad interpolant. By default if no initial points are given, 100 uniformly spaced points are used. The more points the better interpolant but the longer it will take to run the integration.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - maxintervals: maximum numbers of intervals to divide integral in before stopping.
    ##   - initialPoints: A list of known difficult points (integrable singularities, discontinouities etc) that will be used as the inital interval boundaries. This is also the minimum number of points it will evaluate f at, if the function is too smooth it may be evaluated in too few points to give a good enough interpolation.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - A 1D spline which represents the cumulative integral of f from xStart to xEnd.
    var initialPoints = @initialPoints
    if initialPoints.len == 0:
        initialPoints = linspace(min(xStart_in, xEnd_in), max(xStart_in, xEnd_in), 100)
    adaptiveGaussImpl()
    let interval_list = intervals.list.sortedByIt(it.lower)
    var y: T = interval_list[0].value - interval_list[0].value # zero
    if interval_list.len == 1:
        return newHermiteSpline[T]([interval_list[0].lower, interval_list[0].upper], [y, interval_list[0].value])
    var xs = newSeq[float](interval_list.len + 1)
    var ys = newSeq[T](interval_list.len + 1)
    xs[0] = interval_list[0].lower
    ys[0] = y
    for i in 0..interval_list.high:
        y += interval_list[i].value
        ys[i+1] = y
        xs[i+1] = interval_list[i].upper
    result = newHermiteSpline[T](xs, ys)

proc cumGauss*[T](f_in: NumContextProc[T],
                       X: openArray[float], tol = 1e-8, initialPoints: openArray[float] = @[], maxintervals: int = 10000, ctx: NumContext[T] = nil): seq[T] =
    ## Calculate the cumulative integral of f using an globally adaptive Gauss-Kronrod Quadrature.
    ## Returns a sequence of values which is the cumulative integral of f at the points defined in X.
    ## Important: because of the much higher order of the Gauss-Kronrod quadrature (order 21) compared to the interpolating Hermite spline (order 3) you have to give it a large amount of initialPoints.
    ## Otherwise it may only use a couple of points which gives quite a bad interpolant. By default if no initial points are given, 100 uniformly spaced points are used. The more points the better interpolant but the longer it will take to run the integration.
    ##
    ## Input:
    ##   - f: the function that is integrated.
    ##   - X: the points the cumulative integral should be evaluated at.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - maxintervals: maximum numbers of intervals to divide integral in before stopping.
    ##   - initialPoints: A list of known difficult points (integrable singularities, discontinouities etc) that will be used as the inital interval boundaries. This is also the minimum number of points it will evaluate f at, if the function is too smooth it may be evaluated in too few points to give a good enough interpolation.
    ##   - ctx: A context variable that can be accessed and modified in `f`. It is a ref type so IT IS MUTABLE. It can be used to save extra information during the solving for example, or to pass in big Tensors.
    ##
    ## Returns:
    ##   - A sequence of values which is the cumulative integral of f at the points defined in X.
    let xStart = min(X)
    let xEnd = max(X)
    let totalX = concat(@X, @initialPoints)
    var initialPoints = @initialPoints
    if initialPoints.len == 0:
        initialPoints = linspace(xStart, xEnd, 100)
    let spline = cumGaussSpline(f_in, xStart, xEnd, tol=tol, initialPoints=totalX, maxintervals=maxintervals, ctx=ctx)
    result = spline.eval(X)