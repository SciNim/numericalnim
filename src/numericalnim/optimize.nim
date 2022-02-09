import strformat
import arraymancer
import sequtils
import math
import ./differentiate

proc steepest_descent*(deriv: proc(x: float64): float64, start: float64, gamma: float64 = 0.01, precision: float64 = 1e-5, max_iters: Natural = 1000):float64 {.inline.} =
    ## Gradient descent optimization algorithm for finding local minimums of a function with derivative 'deriv'
    ## 
    ## Assuming that a multivariable function F is defined and differentiable near a minimum, F(x) decreases fastest
    ## when going in the direction negative to the gradient of F(a), similar to how water might traverse down a hill 
    ## following the path of least resistance.
    ## can benefit from preconditioning if the condition number of the coefficient matrix is ill-conditioned
    ## Input:
    ##   - deriv: derivative of a multivariable function F
    ##   - start: starting point near F's minimum
    ##   - gamma: step size multiplier, used to control the step size between iterations
    ##   - precision: numerical precision
    ##   - max_iters: maximum iterations
    ##
    ## Returns:
    ##   - float64.
    var
        current = 0.0
        x = start

    for i in 0 .. max_iters:
        # calculate the next direction to propogate
        current = x
        x = current - gamma * deriv(current)
        
        # If we haven't moved much since the last iteration, break
        if abs(x - current) <= precision:
            break

        if i == max_iters:
            raise newException(ArithmeticError, "Maximum iterations for Steepest descent method exceeded")

    return x

proc conjugate_gradient*[T](A, b, x_0: Tensor[T], tolerance: float64): Tensor[T] =
    ## Conjugate Gradient method.
    ## Given a Symmetric and Positive-Definite matrix A, solve the linear system Ax = b
    ## Symmetric Matrix: Square matrix that is equal to its transpose, transpose(A) == A
    ## Positive Definite: Square matrix such that transpose(x)Ax > 0 for all x in R^n
    ## 
    ## Input:
    ##   - A: NxN square matrix
    ##   - b: vector on the right side of Ax=b
    ##   - x_0: Initial guess vector
    ##
    ## Returns:
    ##   - Tensor.
    
    var r = b - (A * x_0)
    var p = r
    var rsold = (r.transpose() * r)[0,0] # multiplication returns a Tensor, so need the first element
    
    result = x_0
    
    var
        Ap = A
        alpha = 0.0
        rsnew = 0.0
        Ap_p = 0.0

    for i in 1 .. b.shape[0]:
        Ap = A * p
        Ap_p = (p.transpose() * Ap)[0,0] 
        alpha = rsold / Ap_p
        result = result + alpha * p
        r = r - alpha * Ap
        rsnew = (r.transpose() * r)[0,0]
        if sqrt(rsnew) < tolerance:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    

proc newtons*(f: proc(x: float64): float64, deriv: proc(x: float64): float64, start: float64, precision: float64 = 1e-5, max_iters: Natural = 1000): float64 {.raises: [ArithmeticError].} =
    ## Newton-Raphson implementation for 1-dimensional functions

    ## Given a single variable function f and it's derivative, calcuate an approximation to f(x) = 0
    ## Input:
    ##   - f: "Well behaved" function of a single variable with a known root
    ##   - deriv: derivative of f with respect to x
    ##   - start: starting x 
    ##   - precision: numerical precision
    ##   - max_iters: maxmimum number of iterations
    ##
    ## Returns:
    ##   - float64.
    var 
        x_iter = start
        i = 0
        current_f = f(start)

    while abs(current_f) >= precision and i <= max_iters:
        current_f = f(x_iter)
        x_iter = x_iter - (current_f / deriv(x_iter))
        i += 1
        if i == max_iters:
            raise newException(ArithmeticError, "Maximum iterations for Newtons method exceeded")

    return x_iter - (current_f / deriv(x_iter))

proc secant*(f: proc(x: float64): float64, start: array[2, float64], precision: float64 = 1e-5, max_iters: Natural = 1000): float64 =
    var xLast = start[0]
    var fLast = f(xLast)
    var xCurrent = start[1]
    var fCurrent = f(xCurrent)
    var xTemp = 0.0
    var i = 0
    while abs(xLast - xCurrent) > precision:
        xTemp = xCurrent
        xCurrent = (xLast * fCurrent - xCurrent * fLast) / (fCurrent - fLast)
        xLast = xTemp
        fLast = fCurrent
        fCurrent = f(xCurrent)
        if i == max_iters:
            raise newException(ArithmeticError, "Maximum iterations for Secant method exceeded")
    return xCurrent

##############################
## Multidimensional methods ##
##############################

proc vectorNorm*[T](v: Tensor[T]): T =
    ## Calculates the norm of the vector, ie the sqrt(Σ vᵢ²)
    assert v.rank == 1, "v must be a 1d vector!"
    result = sqrt(v.dot(v))

proc eye[T](n: int): Tensor[T] =
    result = zeros[T](n, n)
    for i in 0 ..< n:
        result[i, i] = 1

proc steepestDescent*[U, T](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(0.1), tol: U = U(1e-6), fastMode: bool = false): Tensor[U] =
    ## Minimize scalar-valued function f. 
    var x = x0.clone()
    var fNorm = abs(f(x0))
    var gradient = tensorGradient(f, x0, fastMode=fastMode)
    var gradNorm = vectorNorm(gradient)
    var iters: int
    while gradNorm > tol*(1 + fNorm) and iters < 10000:
        let p = -gradient
        x += alpha * p
        let fx = f(x)
        fNorm = abs(fx)
        gradient = tensorGradient(f, x, fastMode=fastMode)
        gradNorm = vectorNorm(gradient)
        iters += 1
    if iters >= 10000:
        echo "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc levmarq*[U, T](f: proc(params: Tensor[U], x: U): T, params0: Tensor[U], xData: Tensor[U], yData: Tensor[T], alpha = U(1), tol = U(1e-6), lambda0 = U(1), fastMode = false): Tensor[U] =
    assert xData.rank == 1
    assert yData.rank == 1
    assert params0.rank == 1
    let xLen = xData.shape[0]
    let yLen = yData.shape[0]
    let paramsLen = params0.shape[0]
    assert xLen == yLen

    let residualFunc = # proc that returns the residual vector
        proc (params: Tensor[U]): Tensor[T] =
            result = map2_inline(xData, yData):
                f(params, x) - y

    var lambdaCoeff = lambda0

    var params = params0.clone()
    var gradient = tensorGradient(residualFunc, params, fastMode=fastMode)
    var residuals = residualFunc(params)
    var resNorm = vectorNorm(residuals)
    var gradNorm = vectorNorm(squeeze(gradient * residuals.reshape(xLen, 1)))
    var iters: int
    let eyeNN = eye[T](paramsLen)
    while gradNorm > tol*(1 + resNorm) and iters < 10000:
        let rhs = -gradient * residuals.reshape(xLen, 1)
        let lhs = gradient * gradient.transpose + lambdaCoeff * eyeNN
        let p = solve(lhs, rhs)
        params += p * alpha
        gradient = tensorGradient(residualFunc, params, fastMode=fastMode)
        residuals = residualFunc(params)
        let newGradNorm = vectorNorm(squeeze(gradient * residuals.reshape(xLen, 1)))
        if newGradNorm / gradNorm < 0.9: # we have improved, decrease lambda → more Gauss-Newton
            lambdaCoeff = max(lambdaCoeff / 3, 1e-4)
        elif newGradNorm / gradNorm > 1.2: # we have done worse than last ste, increase lambda → more Steepest descent
            lambdaCoeff = min(lambdaCoeff * 2, 20)
        # else: don't change anything
        gradNorm = newGradNorm
        resNorm = vectorNorm(residuals)
        iters += 1
    result = params


        
when isMainModule:
    import benchy
    # Steepest descent:
#[     proc f1(x: Tensor[float]): float =
        result = x[0]*x[0] + x[1]*x[1] - 10
    
    let x0 = [10.0, 10.0].toTensor
    echo eye[float](10)
    let sol1 = steepestDescent(f1, x0, tol=1e-10, fastMode=false)
    let sol2 = steepestDescent(f1, x0, tol=1e-10, fastMode=true)
    echo sol1
    echo sol2
    echo f1(sol1)
    echo f1(sol2)

    timeIt "slow mode":
        keep steepestDescent(f1, x0, tol=1e-10, fastMode=false)
    timeIt "fast mode":
        keep steepestDescent(f1, x0, tol=1e-10, fastMode=true) ]#

    # Lev-Marq:
    proc fFit(params: Tensor[float], x: float): float =
        params[0] + params[1] * x + params[2] * x*x
    
    let xData = arraymancer.linspace(0, 10, 100)
    let yData = 1.5 +. xData * 6.28 + xData *. xData * -5.79
    let params0 = [0.0, 0.0, 0.0].toTensor
    echo levmarq(fFit, params0, xData, yData)




