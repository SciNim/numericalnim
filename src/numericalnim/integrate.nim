import math
import utils
import arraymancer

# N: #intervals
proc trapz*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, N = 500, optional: openArray[T] = @[]): T =
    ## Calculate the integral of f using the trapezoidal rule.
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f). 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - N: The number of subintervals to divide the integration interval into.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using the trapezoidal rule.
    if N < 1:
        raise newException(ValueError, "N must be an integer >= 1")
    let optional = @optional
    let dx = (xEnd - xStart)/N.toFloat
    result = (f(xStart, optional) + f(xEnd, optional)) / 2.0
    for i in 1 .. N - 1:
        result += f(xStart + dx * i.toFloat, optional)
    result *= dx

proc trapz*[T](Y: openArray[T], X: openArray[float]): T =
    ## Calculate the integral of f using the trapezoidal rule from a set values.
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The integral evaluated from the smallest to the largest value in X calculated using the trapezoidal rule.
    let dataset = sortDataset(X, Y)
    result = 0.5 * (dataset[1][0] - dataset[0][0]) * (dataset[0][1] + dataset[1][1])
    for i in 1 .. dataset.high - 1:
        #                ( x_i+1         -    x_i)        *   (y_i+1         +    y_i)
        result += 0.5 * (dataset[i+1][0] - dataset[i][0]) * (dataset[i+1][1] + dataset[i][1])

# discrete points
proc cumtrapz*[T](Y: openArray[T], X: openArray[float]): seq[T] =
    ## Calculate the cumulative integral of f using the trapezoidal rule from a set of values.
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The cumulative integral evaluated from the smallest to the largest value in X calculated using the trapezoidal rule.
    let dataset = sortDataset(X, Y)
    result.add(dataset[0][1] - dataset[0][1]) # get the right kind of zero
    var integral = 0.5 * (dataset[1][0] - dataset[0][0]) * (dataset[0][1] + dataset[1][1])
    result.add(integral)
    for i in 1 .. dataset.high - 1:
        integral += 0.5 * (dataset[i+1][0] - dataset[i][0]) * (dataset[i+1][1] + dataset[i][1])
        result.add(integral)

# function values only on discrete points
proc cumtrapz*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[]): seq[T] =
    let optional = @optional
    var Y: seq[T]
    for x in X:
        Y.add(f(x, optional))
    result = cumtrapz(Y, X)

# function values calculated according to the dx and then interpolated
proc cumtrapz*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[], dx = 1e-5): seq[T] =
    ## Calculate the cumulative integral of f using the trapezoidal rule at the points in X.
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f).
    ##   - X: The x-values of the returned values.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##   - dx: The step length to use when integrating.
    ##
    ## Returns:
    ##   - The value of the integral of f from the smallest to the largest value of X calculated using the trapezoidal rule.
    var
        times: seq[float]
        dy, y: seq[T]
        dyTemp, dyPrev: T
    let optional = @optional
    var t = min(X)
    let tEnd = max(X) + 1.0 # make sure to get the endpoint as well. 
    dyTemp = f(t, optional)
    var integral = dyTemp - dyTemp # get the right kind of zero
    times.add(t)
    dy.add(dyTemp)
    y.add(integral)
    t += dx
    while t <= tEnd:
        dyPrev = dyTemp
        dyTemp = f(t, optional)
        integral += 0.5 * dx * (dyPrev + dyTemp)
        times.add(t)
        dy.add(dyTemp)
        y.add(integral)
        t += dx
    result = hermiteInterpolate(X, times, y, dy)


proc simpson*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, N = 500, optional: openArray[T] = @[]): T =
    ## Calculate the integral of f using Simpson's rule.
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f). 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - N: The number of subintervals to divide the integration interval into. Must be 2 or greater.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using Simpson's rule.
    if N < 2:
        raise newException(ValueError, "N must be an integer >= 2")
    let optional = @optional
    let dx = (xEnd - xStart)/N.toFloat
    var N = N
    var xStart = xStart
    result = f(xStart, optional) - f(xStart, optional) # initialize to right kind of zero
    if N mod 2 != 0:
        result += 3.0 / 8.0 * dx * (f(xStart, optional) + 3.0 * f(xStart + dx, optional) + 3.0 * f(xStart + 2.0 * dx, optional) + f(xStart + 3.0 * dx, optional))
        xStart = xStart + 3.0 * dx
        N = N - 3
        if N == 0:
            return result
    var resultTemp = f(xStart, optional) + f(xEnd, optional)
    var res1 = f(xStart, optional) - f(xStart, optional) # initialize to right kind of zero
    var res2 = res1.clone()
    for j in 1 .. (N / 2 - 1).toInt:
        res1 += f(xStart + dx * 2.0 * j.toFloat, optional)
    for j in 1 .. (N / 2).toInt:
        res2 += f(xStart + dx * (2.0 * j.toFloat - 1.0), optional)
    
    resultTemp += 2.0 * res1 + 4.0 * res2
    resultTemp *= dx / 3.0
    result += resultTemp

proc simpson*[T](Y: openArray[T], X: openArray[float]): T =
    ## Calculate the integral of f using Simpson's rule from a set of values.
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The integral evaluated from the smallest to the largest value in X calculated using Simpson's rule.
    var dataset = sortDataset(X, Y)
    var N = dataset.len
    if N < 3:
        raise newException(ValueError, "X and Y must have at least 3 elements to perform Simpson")
    var alpha, beta, eta: float
    if N mod 2 == 0:
        let lastIndex = dataset.high
        let h1 = dataset[lastIndex - 1][0] - dataset[lastIndex - 2][0]
        let h2 = dataset[lastIndex][0] - dataset[lastIndex - 1][0]
        alpha = (2.0 * h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * (h1 + h2))
        beta = (h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * h1)
        eta = -(h2 ^ 3) / (6.0 * h1 * (h1 + h2))
        result = eta * dataset[lastIndex - 2][1] + beta * dataset[lastIndex - 1][1] + alpha * dataset[lastIndex][1]
        dataset = dataset[0 ..< lastIndex]
        N -= 1
    for i in 0 ..< ((N-1)/2).toInt:
        let h1 = dataset[2*i + 1][0] - dataset[2*i][0]
        let h2 = dataset[2*i + 2][0] - dataset[2*i + 1][0]
        alpha = (2.0 * h2 ^ 3 - h1 ^ 3 + 3.0 * h1 * h2 ^ 2) / (6.0 * h2 * (h2 + h1))
        beta = (h2 ^ 3 + h1 ^ 3 + 3.0 * h1 * h2 * (h2 + h1)) / (6.0 * h2 * h1)
        eta = (2.0 * h1 ^ 3 - h2 ^ 3 + 3.0 * h2 * h1 ^ 2) / (6.0 * h1 * (h2 + h1))
        result += alpha * dataset[2*i + 2][1] + beta * dataset[2*i + 1][1] + eta * dataset[2*i][1]

proc adaptiveSimpson*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, tol = 1e-8, optional: openArray[T] = @[]): T =
    ## Calculate the integral of f using a adaptive Simpson's rule.
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f). 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - tol: The error tolerance that must be satisfied on every subinterval.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using an adaptive Simpson's rule.
    let zero = f(xStart, @optional) - f(xStart, @optional)
    let value1 = simpson(f, xStart, xEnd, N = 2, optional = optional)
    let value2 = simpson(f, xStart, xEnd, N = 4, optional = optional)
    let error = (value2 - value1)/15
    var tol = tol
    if tol < 1e-15:
        tol = 1e-15
    if calcError(error, zero) < tol or abs(xEnd - xStart) < 1e-5:
        return value2 + error
    let m = (xStart + xEnd) / 2.0
    let newtol = tol / 2.0
    let left = adaptiveSimpson(f, xStart, m, tol = newtol, optional = optional)
    let right = adaptiveSimpson(f, m, xEnd, tol = newtol, optional = optional)
    return left + right


proc cumsimpson*[T](Y: openArray[T], X: openArray[float]): seq[T] =
    ## Calculate the cumulative integral of f using Simpson's rule from a set of values.
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The cumulative integral evaluated from the smallest to the largest value in X calculated using Simpson's rule.
    var dataset = sortDataset(X, Y)
    var N = dataset.len
    var alpha, beta, eta: float
    var y, dy: seq[T]
    var xs: seq[float]
    var evenN = false 
    if N < 3:
        raise newException(ValueError, "X and Y must have at least 3 elements to perform Simpson, use cumtrapz instead")
    if N mod 2 == 0:
        evenN = true
        N -= 1
    var integral = dataset[0][1] - dataset[0][1] # get the right kind of zero
    y.add(integral)
    dy.add(dataset[0][1])
    xs.add(dataset[0][0])
    for i in 0 ..< ((N-1)/2).toInt:
        let h1 = dataset[2*i + 1][0] - dataset[2*i][0]
        let h2 = dataset[2*i + 2][0] - dataset[2*i + 1][0]
        alpha = (2.0 * h2 ^ 3 - h1 ^ 3 + 3.0 * h1 * h2 ^ 2) / (6.0 * h2 * (h2 + h1))
        beta = (h2 ^ 3 + h1 ^ 3 + 3.0 * h1 * h2 * (h2 + h1)) / (6.0 * h2 * h1)
        eta = (2.0 * h1 ^ 3 - h2 ^ 3 + 3.0 * h2 * h1 ^ 2) / (6.0 * h1 * (h2 + h1))
        integral += alpha * dataset[2*i + 2][1] + beta * dataset[2*i + 1][1] + eta * dataset[2*i][1]
        y.add(integral)
        dy.add(dataset[2*i+2][1])
        xs.add(dataset[2*i+2][0])
    if evenN:
        let lastIndex = dataset.high
        let h1 = dataset[lastIndex - 1][0] - dataset[lastIndex - 2][0]
        let h2 = dataset[lastIndex][0] - dataset[lastIndex - 1][0]
        alpha = (2.0 * h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * (h1 + h2))
        beta = (h2 ^ 2 + 3.0 * h1 * h2) / (6.0 * h1)
        eta = -(h2 ^ 3) / (6.0 * h1 * (h1 + h2))
        integral += eta * dataset[lastIndex - 2][1] + beta * dataset[lastIndex - 1][1] + alpha * dataset[lastIndex][1]
        y.add(integral)
        dy.add(dataset[dataset.high][1])
        xs.add(dataset[dataset.high][0])
    result = hermiteInterpolate(X, xs, y, dy)

proc cumsimpson*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[]): seq[T] =
    let optional = @optional
    var Y: seq[T]
    for x in X:
        Y.add(f(x, optional))
    result = cumsimpson(Y, X)

proc cumsimpson*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[], dx: float): seq[T] =
    ## Calculate the cumulative integral of f using Simpson's rule.
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f).
    ##   - X: The x-values of the returned values.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##   - dx: The step length to use when integrating.
    ##
    ## Returns:
    ##   - The value of the integral of f from the smallest to the largest value of X calculated using Simpson's rule.
    let optional = @optional
    var dy: seq[T]
    let t = linspace(min(X), max(X), ((max(X) - min(X)) / dx).toInt + 2)
    for x in t:
        dy.add(f(x, optional))
    let ys = cumsimpson(dy, t)
    result = hermiteInterpolate(X, t, ys, dy)

proc romberg*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, depth = 8, tol = 1e-8, optional: openArray[T] = @[]): T =
    ## Calculate the integral of f using Romberg Integration.
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f). 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - depth: The maximum depth of the Richardson Extrapolation.
    ##   - tol: The error tolerance that must be satisfied.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##
    ## Returns:
    ##   - The value of the integral of f from xStart to xEnd calculated using Romberg integration.
    if depth < 2:
        raise newException(ValueError, "depth must be 2 or greater")
    let optional = @optional
    var values: seq[seq[T]]
    var firstIteration: seq[T]
    for i in 0 ..< depth:
        firstIteration.add(trapz(f, xStart, xEnd, N = 2 ^ i, optional = optional))
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
    ## Input:
    ##   - Y: A seq of values of the integrand.
    ##   - X: A seq with the corresponding x-values.
    ##
    ## Returns:
    ##   - The integral evaluated from the smallest to the largest value in X calculated using Romberg Integration.
    let dataset = sortDataset(X, Y)
    let N = dataset.len
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
            xs.add(dataset[x][0])
            vals.add(dataset[x][1])
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
