import math, tables
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

proc cumsimpson*[T](f: proc(x: float, optional: seq[T]): T, X: openArray[float], optional: openArray[T] = @[], dx = 1e-5): seq[T] =
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

proc getGaussLegendreWeights(nPoints: int): tuple[nodes: seq[float], weights: seq[float]] {.inline.} =
    assert(0 < nPoints and nPoints <= 20, "nPoints must be an integer between 1 and 20.")
    const gaussWeights = {
        1: (@[0.0], @[2.0]),
        2: (@[-0.5773502691896257645092, 0.5773502691896257645092], @[1.0, 1.0]),
        3: (@[-0.7745966692414833770359, 0.0, 0.7745966692414833770359], @[0.5555555555555555555556, 0.8888888888888888888889, 0.555555555555555555556]),
        4: (@[-0.861136311594052575224, -0.3399810435848562648027, 0.3399810435848562648027, 0.861136311594052575224], @[0.3478548451374538573731, 0.6521451548625461426269, 0.6521451548625461426269, 0.3478548451374538573731]),
        5: (@[0.90617984593866385,0.538469310105683,0,-0.538469310105683,-0.90617984593866385], @[0.23692688505618911,0.47862867049936625,0.56888888888888889,0.47862867049936625,0.23692688505618911]),
        6: (@[0.932469514203152,0.66120938646626448,0.23861918608319693,-0.23861918608319693,-0.66120938646626448,-0.932469514203152], @[0.1713244923791705,0.36076157304813866,0.467913934572691,0.467913934572691,0.36076157304813866,0.1713244923791705]),
        7: (@[0.94910791234275838,0.74153118559939446,0.40584515137739713,0,-0.40584515137739713,-0.74153118559939446,-0.94910791234275838], @[0.12948496616886965,0.27970539148927676,0.38183005050511903,0.41795918367346935,0.38183005050511903,0.27970539148927676,0.12948496616886965]),
        8: (@[0.9602898564975364,0.79666647741362673,0.525532409916329,0.18343464249564984,-0.18343464249564984,-0.525532409916329,-0.79666647741362673,-0.9602898564975364], @[0.10122853629037681,0.22238103445337443,0.31370664587788732,0.3626837833783621,0.3626837833783621,0.31370664587788732,0.22238103445337443,0.10122853629037681]),
        9: (@[0.96816023950762609,0.83603110732663577,0.61337143270059036,0.32425342340380897,0,-0.32425342340380897,-0.61337143270059036,-0.83603110732663577,-0.96816023950762609], @[0.081274388361574634,0.1806481606948574,0.26061069640293555,0.31234707704000292,0.33023935500125978,0.31234707704000292,0.26061069640293555,0.1806481606948574,0.081274388361574634]),
        10: (@[0.97390652851717174,0.86506336668898465,0.67940956829902444,0.43339539412924721,0.14887433898163116,-0.14887433898163116,-0.43339539412924721,-0.67940956829902444,-0.86506336668898465,-0.97390652851717174], @[0.066671344308688027,0.14945134915058056,0.21908636251598207,0.26926671930999618,0.29552422471475293,0.29552422471475293,0.26926671930999618,0.21908636251598207,0.14945134915058056,0.066671344308688027]),
        11: (@[0.97822865814605686,0.88706259976809543,0.73015200557404936,0.5190961292068117,0.2695431559523449,0,-0.2695431559523449,-0.5190961292068117,-0.73015200557404936,-0.88706259976809543,-0.97822865814605686], @[0.055668567116173538,0.12558036946490408,0.186290210927734,0.23319376459199023,0.26280454451024671,0.27292508677790062,0.26280454451024671,0.23319376459199023,0.186290210927734,0.12558036946490408,0.055668567116173538]),
        12: (@[0.98156063424671913,0.90411725637047491,0.76990267419430469,0.58731795428661737,0.36783149899818013,0.12523340851146886,-0.12523340851146886,-0.36783149899818013,-0.58731795428661737,-0.76990267419430469,-0.90411725637047491,-0.98156063424671913], @[0.0471753363865118,0.10693932599531812,0.16007832854334605,0.20316742672306581,0.23349253653835478,0.24914704581340286,0.24914704581340286,0.23349253653835478,0.20316742672306581,0.16007832854334605,0.10693932599531812,0.0471753363865118]),
        13: (@[0.98418305471858814,0.91759839922297792,0.80157809073330988,0.64234933944034012,0.44849275103644692,0.23045831595513477,0,-0.23045831595513477,-0.44849275103644692,-0.64234933944034012,-0.80157809073330988,-0.91759839922297792,-0.98418305471858814], @[0.040484004765315815,0.092121499837728438,0.13887351021978714,0.17814598076194568,0.20781604753688834,0.22628318026289709,0.2325515532308739,0.22628318026289709,0.20781604753688834,0.17814598076194568,0.13887351021978714,0.092121499837728438,0.040484004765315815]),
        14: (@[0.98628380869681243,0.92843488366357363,0.827201315069765,0.68729290481168537,0.5152486363581541,0.31911236892788969,0.10805494870734367,-0.10805494870734367,-0.31911236892788969,-0.5152486363581541,-0.68729290481168537,-0.827201315069765,-0.92843488366357363,-0.98628380869681243], @[0.035119460331752,0.080158087159760041,0.1215185706879032,0.15720316715819357,0.18553839747793785,0.20519846372129569,0.21526385346315768,0.21526385346315768,0.20519846372129569,0.18553839747793785,0.15720316715819357,0.1215185706879032,0.080158087159760041,0.035119460331752]),
        15: (@[0.98799251802048538,0.937273392400706,0.84820658341042732,0.72441773136017007,0.57097217260853883,0.39415134707756344,0.20119409399743454,0,-0.20119409399743454,-0.39415134707756344,-0.57097217260853883,-0.72441773136017007,-0.84820658341042732,-0.937273392400706,-0.98799251802048538], @[0.030753241996116634,0.070366047488108138,0.1071592204671718,0.13957067792615424,0.16626920581699395,0.18616100001556207,0.19843148532711161,0.20257824192556126,0.19843148532711161,0.18616100001556207,0.16626920581699395,0.13957067792615424,0.1071592204671718,0.070366047488108138,0.030753241996116634]),
        16: (@[0.98940093499164994,0.9445750230732326,0.86563120238783187,0.75540440835500311,0.61787624440264377,0.45801677765722731,0.28160355077925892,0.095012509837637482,-0.095012509837637482,-0.28160355077925892,-0.45801677765722731,-0.61787624440264377,-0.75540440835500311,-0.86563120238783187,-0.9445750230732326,-0.98940093499164994], @[0.027152459411754058,0.062253523938647776,0.0951585116824929,0.12462897125553393,0.14959598881657682,0.16915651939500256,0.18260341504492361,0.18945061045506845,0.18945061045506845,0.18260341504492361,0.16915651939500256,0.14959598881657682,0.12462897125553393,0.0951585116824929,0.062253523938647776,0.027152459411754058]),
        17: (@[0.99057547531441736,0.95067552176876768,0.8802391537269858,0.78151400389680137,0.65767115921669062,0.51269053708647694,0.35123176345387636,0.17848418149584783,0,-0.17848418149584783,-0.35123176345387636,-0.51269053708647694,-0.65767115921669062,-0.78151400389680137,-0.8802391537269858,-0.95067552176876768,-0.99057547531441736], @[0.024148302868547931,0.055459529373987133,0.085036148317179164,0.11188384719340388,0.13513636846852545,0.15404576107681039,0.1680041021564499,0.17656270536699264,0.17944647035620653,0.17656270536699264,0.1680041021564499,0.15404576107681039,0.13513636846852545,0.11188384719340388,0.085036148317179164,0.055459529373987133,0.024148302868547931]),
        18: (@[0.991565168420931,0.9558239495713976,0.89260246649755581,0.80370495897252314,0.69168704306035322,0.55977083107394754,0.41175116146284263,0.25188622569150554,0.084775013041735292,-0.084775013041735292,-0.25188622569150554,-0.41175116146284263,-0.55977083107394754,-0.69168704306035322,-0.80370495897252314,-0.89260246649755581,-0.9558239495713976,-0.991565168420931], @[0.021616013526483353,0.049714548894969283,0.0764257302548889,0.10094204410628704,0.12255520671147835,0.1406429146706506,0.15468467512626538,0.1642764837458327,0.16914238296314352,0.16914238296314352,0.1642764837458327,0.15468467512626538,0.1406429146706506,0.12255520671147835,0.10094204410628704,0.0764257302548889,0.049714548894969283,0.021616013526483353]),
        19: (@[0.99240684384358424,0.96020815213483,0.903155903614818,0.82271465653714282,0.72096617733522939,0.6005453046616811,0.464570741375961,0.31656409996362989,0.16035864564022534,0,-0.16035864564022534,-0.31656409996362989,-0.464570741375961,-0.6005453046616811,-0.72096617733522939,-0.82271465653714282,-0.903155903614818,-0.96020815213483,-0.99240684384358424], @[0.01946178822972643,0.044814226765699405,0.06904454273764106,0.091490021622449916,0.11156664554733405,0.12875396253933627,0.14260670217360666,0.15276604206585967,0.1589688433939544,0.16105444984878367,0.1589688433939544,0.15276604206585967,0.14260670217360666,0.12875396253933627,0.11156664554733405,0.091490021622449916,0.06904454273764106,0.044814226765699405,0.01946178822972643]),
        20: (@[0.99312859918509488,0.96397192727791392,0.912234428251326,0.83911697182221889,0.7463319064601508,0.63605368072651514,0.51086700195082724,0.37370608871541955,0.22778585114164507,0.076526521133497338,-0.076526521133497338,-0.22778585114164507,-0.37370608871541955,-0.51086700195082724,-0.63605368072651514,-0.7463319064601508,-0.83911697182221889,-0.912234428251326,-0.96397192727791392,-0.99312859918509488], @[0.01761400713915167,0.04060142980038705,0.06267204833410904,0.083276741576704769,0.1019301198172405,0.11819453196151831,0.13168863844917658,0.14209610931838215,0.14917298647260382,0.152753387130726,0.152753387130726,0.14917298647260382,0.14209610931838215,0.13168863844917658,0.11819453196151831,0.1019301198172405,0.083276741576704769,0.06267204833410904,0.04060142980038705,0.01761400713915167]),
    }.toTable()
    return gaussWeights[nPoints]

proc gaussQuad*[T](f: proc(x: float, optional: seq[T]): T, xStart, xEnd: float, N = 100, nPoints = 7, optional: openArray[T] = @[]): T =
    ## Calculate the integral of f using Gaussian Quadrature. Has 20 different sets of weights, ranging from 1 to 20 function evaluations per subinterval. 
    ## Input:
    ##   - f: the function that is integrated. x is the independent variable and optional is a seq of optional parameters (must be of same type as the output of f). 
    ##   - xStart: The start of the integration interval.
    ##   - xEnd: The end of the integration interval.
    ##   - N: The number of subintervals to divide the integration interval into.
    ##   - nPoints: The number of points to evaluate f at per interval. Choose between 1 to 20 with increasing accuracy.
    ##   - optional: A seq of optional parameters that is passed to f.
    ##
    ## Returns:
    ##   - The integral evaluated from the xStart to xEnd calculated using Gaussian Quadrature.
    if N < 1:
        raise newException(ValueError, "N must be an integer >= 1")
    let optional = @optional
    let dx = (xEnd - xStart)/N.toFloat
    let (nodes, weights) = getGaussLegendreWeights(nPoints)
    let zero = f(nodes[0], optional) - f(nodes[0], optional)
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
            tempResult += weights[i] * f(c1 * nodes[i] + c2, optional)
        tempResult *= c1
        result += tempResult
