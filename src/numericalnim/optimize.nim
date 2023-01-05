import std/[strformat, sequtils, math, deques]
import arraymancer
import
    ./differentiate,
    ./utils

when not defined(nimHasEffectsOf):
  {.pragma: effectsOf.}

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
    

proc newtons*(f: proc(x: float64): float64,
              deriv: proc(x: float64): float64,
              start: float64, precision: float64 = 1e-5,
              max_iters: Natural = 1000
              ): float64{.raises: [ArithmeticError], effectsOf: [f, deriv].} =
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

# ######################## #
# Multidimensional methods #
# ######################## #

type LineSearchCriterion* = enum
    Armijo, Wolfe, WolfeStrong, NoLineSearch

type
    OptimOptions*[U, AO] = object
        tol*, alpha*: U
        fastMode*: bool
        maxIterations*: int
        lineSearchCriterion*: LineSearchCriterion
        algoOptions*: AO
    StandardOptions* = object
    LevMarqOptions*[U] = object
        lambda0*: U
    LBFGSOptions*[U] = object
        savedIterations*: int

proc optimOptions*[U](tol: U = U(1e-6), alpha: U = U(1), fastMode: bool = false, maxIterations: int = 10000, lineSearchCriterion: LineSearchCriterion = NoLineSearch): OptimOptions[U, StandardOptions] =
    ## Returns a vanilla OptimOptions
    ## - tol: The tolerance used. This is the criteria for convergence: `gradNorm < tol*(1 + fNorm)`.
    ## - alpha: The step size.
    ## - fastMode: If true, a faster first order accurate finite difference approximation of the derivative will be used. 
    ##   Else a more accurate but slowe second order finite difference scheme will be used.
    ## - maxIteration: The maximum number of iteration before returning if convergence haven't been reached.
    ## - lineSearchCriterion: Which line search method to use.
    result.tol = tol
    result.alpha = alpha
    result.fastMode = fastMode
    result.maxIterations = maxIterations
    result.lineSearchCriterion = lineSearchCriterion

proc steepestDescentOptions*[U](tol: U = U(1e-6), alpha: U = U(0.001), fastMode: bool = false, maxIterations: int = 10000, lineSearchCriterion: LineSearchCriterion = NoLineSearch): OptimOptions[U, StandardOptions] =
    ## Returns a Steepest Descent OptimOptions
    ## - tol: The tolerance used. This is the criteria for convergence: `gradNorm < tol*(1 + fNorm)`.
    ## - alpha: The step size.
    ## - fastMode: If true, a faster first order accurate finite difference approximation of the derivative will be used. 
    ##   Else a more accurate but slowe second order finite difference scheme will be used.
    ## - maxIteration: The maximum number of iteration before returning if convergence haven't been reached.
    ## - lineSearchCriterion: Which line search method to use.
    result.tol = tol
    result.alpha = alpha
    result.fastMode = fastMode
    result.maxIterations = maxIterations
    result.lineSearchCriterion = lineSearchCriterion

proc newtonOptions*[U](tol: U = U(1e-6), alpha: U = U(1), fastMode: bool = false, maxIterations: int = 10000, lineSearchCriterion: LineSearchCriterion = NoLineSearch): OptimOptions[U, StandardOptions] =
    ## Returns a Newton OptimOptions
    ## - tol: The tolerance used. This is the criteria for convergence: `gradNorm < tol*(1 + fNorm)`.
    ## - alpha: The step size.
    ## - fastMode: If true, a faster first order accurate finite difference approximation of the derivative will be used. 
    ##   Else a more accurate but slowe second order finite difference scheme will be used.
    ## - maxIteration: The maximum number of iteration before returning if convergence haven't been reached.
    ## - lineSearchCriterion: Which line search method to use.
    result.tol = tol
    result.alpha = alpha
    result.fastMode = fastMode
    result.maxIterations = maxIterations
    result.lineSearchCriterion = lineSearchCriterion

proc bfgsOptions*[U](tol: U = U(1e-6), alpha: U = U(1), fastMode: bool = false, maxIterations: int = 10000, lineSearchCriterion: LineSearchCriterion = NoLineSearch): OptimOptions[U, StandardOptions] =
    ## Returns a BFGS OptimOptions
    ## - tol: The tolerance used. This is the criteria for convergence: `gradNorm < tol*(1 + fNorm)`.
    ## - alpha: The step size.
    ## - fastMode: If true, a faster first order accurate finite difference approximation of the derivative will be used. 
    ##   Else a more accurate but slowe second order finite difference scheme will be used.
    ## - maxIteration: The maximum number of iteration before returning if convergence haven't been reached.
    ## - lineSearchCriterion: Which line search method to use.
    result.tol = tol
    result.alpha = alpha
    result.fastMode = fastMode
    result.maxIterations = maxIterations
    result.lineSearchCriterion = lineSearchCriterion

proc lbfgsOptions*[U](savedIterations: int = 10, tol: U = U(1e-6), alpha: U = U(1), fastMode: bool = false, maxIterations: int = 10000, lineSearchCriterion: LineSearchCriterion = NoLineSearch): OptimOptions[U, LBFGSOptions[U]] =
    ## Returns a LBFGS OptimOptions
    ## - tol: The tolerance used. This is the criteria for convergence: `gradNorm < tol*(1 + fNorm)`.
    ## - alpha: The step size.
    ## - fastMode: If true, a faster first order accurate finite difference approximation of the derivative will be used. 
    ##   Else a more accurate but slowe second order finite difference scheme will be used.
    ## - maxIteration: The maximum number of iteration before returning if convergence haven't been reached.
    ## - lineSearchCriterion: Which line search method to use.
    ## - savedIterations: Number of past iterations to save. The higher the value, the better but slower steps.
    result.tol = tol
    result.alpha = alpha
    result.fastMode = fastMode
    result.maxIterations = maxIterations
    result.lineSearchCriterion = lineSearchCriterion
    result.algoOptions.savedIterations = savedIterations

proc levmarqOptions*[U](lambda0: U = U(1), tol: U = U(1e-6), alpha: U = U(1), fastMode: bool = false, maxIterations: int = 10000, lineSearchCriterion: LineSearchCriterion = NoLineSearch): OptimOptions[U, LevMarqOptions[U]] =
    ## Returns a levmarq OptimOptions
    ## - tol: The tolerance used. This is the criteria for convergence: `gradNorm < tol*(1 + fNorm)`.
    ## - alpha: The step size.
    ## - fastMode: If true, a faster first order accurate finite difference approximation of the derivative will be used. 
    ##   Else a more accurate but slowe second order finite difference scheme will be used.
    ## - maxIteration: The maximum number of iteration before returning if convergence haven't been reached.
    ## - lineSearchCriterion: Which line search method to use.
    ## - lambda0: Starting value of dampening parameter
    result.tol = tol
    result.alpha = alpha
    result.fastMode = fastMode
    result.maxIterations = maxIterations
    result.lineSearchCriterion = lineSearchCriterion
    result.algoOptions.lambda0 = lambda0




proc vectorNorm*[T](v: Tensor[T]): T =
    ## Calculates the norm of the vector, ie the sqrt(Σ vᵢ²)
    assert v.rank == 1, "v must be a 1d vector!"
    result = sqrt(v.dot(v))

proc eye[T](n: int): Tensor[T] =
    result = zeros[T](n, n)
    for i in 0 ..< n:
        result[i, i] = 1

proc line_search*[U, T](alpha: var U, p: Tensor[T], x0: Tensor[U], f: proc(x: Tensor[U]): T, criterion: LineSearchCriterion, fastMode: bool = false) =
    # note: set initial alpha for the methods as 1 / sqrt(dot(grad, grad)) so first step has length 1.
    if criterion == NoLineSearch:
        return
    var gradient = tensorGradient(f, x0, fastMode=fastMode)
    let dirDerivInit = dot(gradient, p)

    if 0 < dirDerivInit:
        # p is pointing uphill, use whatever alpha we currently have.
        return

    let fInit = f(x0)
    var counter = 0
    alpha = 1
    while counter < 20:
        let x = x0 + alpha * p
        let fx = f(x)
        gradient = tensorGradient(f, x, fastMode=fastMode)
        counter += 1

        if fx > fInit + 1e-4*alpha * dirDerivInit: # c1 = 1e-4 multiply as well? Doesn't seem to work
            alpha *= 0.5
        else:
            if criterion == Armijo:
                return
            let dirDeriv = dot(gradient, p)
            if dirDeriv < 0.9 * dirDerivInit:
                alpha *= 2.1
            else:
                if criterion == Wolfe:
                    return
                if dirDeriv > -0.9*dirDerivInit:
                    alpha *= 0.5
                else:
                    return
        if alpha < 1e-3:
            alpha = 1e-3
            return
        elif alpha > 1e2:
            alpha = 1e2
            return

template analyticOrNumericGradient(analytic, f, x, options: untyped): untyped =
    if analytic.isNil:
        tensorGradient(f, x, fastMode=options.fastMode)
    else:
        analytic(x)

proc steepestDescent*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], options: OptimOptions[U, StandardOptions] = steepestDescentOptions[U](), analyticGradient: proc(x: Tensor[U]): Tensor[T] = nil): Tensor[U] =
    ## Steepest descent method for optimization.
    ## 
    ## Inputs:
    ## - f: The function to optimize. It should take as input a 1D Tensor of the input variables and return a scalar.
    ## - options: Options object (see `steepestDescentOptions` for constructing one)
    ## - analyticGradient: The analytic gradient of `f` taking in and returning a 1D Tensor. If not provided, a finite difference approximation will be performed instead.
    ## 
    ## Returns:
    ## - The final solution for the parameters. Either because a (local) minimum was found or because the maximum number of iterations was reached.
    var alpha = options.alpha
    var x = x0.clone()
    var fNorm = abs(f(x0))
    var gradient = analyticOrNumericGradient(analyticGradient, f, x0, options) #tensorGradient(f, x0, fastMode=options.fastMode)
    var gradNorm = vectorNorm(gradient)
    var iters: int
    while gradNorm > options.tol*(1 + fNorm) and iters < 10000:
        let p = -gradient
        line_search(alpha, p, x, f, options.lineSearchCriterion, options.fastMode)
        x += alpha * p
        let fx = f(x)
        fNorm = abs(fx)
        gradient = analyticOrNumericGradient(analyticGradient, f, x, options) #tensorGradient(f, x, fastMode=options.fastMode)
        gradNorm = vectorNorm(gradient)
        iters += 1
    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc newton*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], options: OptimOptions[U, StandardOptions] = newtonOptions[U](), analyticGradient: proc(x: Tensor[U]): Tensor[T] = nil): Tensor[U] =
    ## Newton's method for optimization.
    ## 
    ## Inputs:
    ## - f: The function to optimize. It should take as input a 1D Tensor of the input variables and return a scalar.
    ## - options: Options object (see `newtonOptions` for constructing one)
    ## - analyticGradient: The analytic gradient of `f` taking in and returning a 1D Tensor. If not provided, a finite difference approximation will be performed instead.
    ## 
    ## Returns:
    ## - The final solution for the parameters. Either because a (local) minimum was found or because the maximum number of iterations was reached.
    var alpha = options.alpha
    var x = x0.clone()
    var fNorm = abs(f(x))
    var gradient = analyticOrNumericGradient(analyticGradient, f, x0, options)
    var gradNorm = vectorNorm(gradient)
    var hessian = tensorHessian(f, x)
    var iters: int
    while gradNorm > options.tol*(1 + fNorm) and iters < 10000:
        let p = -solve(hessian, gradient)
        line_search(alpha, p, x, f, options.lineSearchCriterion, options.fastMode)
        x += alpha * p
        let fx = f(x)
        fNorm = abs(fx)
        gradient = analyticOrNumericGradient(analyticGradient, f, x, options)
        gradNorm = vectorNorm(gradient)
        hessian = tensorHessian(f, x)
        iters += 1
    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc bfgs_old*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(1), tol: U = U(1e-6), fastMode: bool = false, analyticGradient: proc(x: Tensor[U]): Tensor[T] = nil): Tensor[U] =
    var x = x0.clone()
    let xLen = x.shape[0]
    var fNorm = abs(f(x))
    var gradient = 0.01*tensorGradient(f, x, fastMode=fastMode)
    var gradNorm = vectorNorm(gradient)
    var hessianB = eye[T](xLen) # inverse of the approximated hessian
    var iters: int
    while gradNorm > tol*(1 + fNorm) and iters < 10000:
        #echo "Hessian iter ", iters, ": ", hessianB
        let p = -hessianB * gradient.reshape(xLen, 1)
        #echo "p iter ", iters, ": ", p
        x += alpha * p
        let newGradient = tensorGradient(f, x, fastMode=fastMode)
        let sk = alpha * p.reshape(xLen, 1)
        #gemm(1.0, hessianB, hessianB, 0.0, newGradient)
        let yk = (newGradient - gradient).reshape(xLen, 1)
        let sk_yk_dot = dot(sk.squeeze, yk.squeeze)
        let prefactor1 = (sk_yk_dot + dot(yk.squeeze, squeeze(hessianB * yk))) / (sk_yk_dot * sk_yk_dot)
        let prefactor2 = 1 / sk_yk_dot
        let m1 = (sk * sk.transpose)
        let m2_1 = (hessianB * yk) * sk.transpose
        let m2_2 = sk * (yk.transpose * hessianB)
        #echo "prefactor2: ", prefactor2
        hessianB += prefactor1 * m1
        #echo "hessian2: ", hessianB
        #echo "hessian: ", hessianB
        #echo "yk: ", yk
        hessianB -= prefactor2 * m2_1
        #echo "checkpoint1: ", (hessianB * yk)
        #echo "hessian2.5: ", hessianB
        hessianB -= prefactor2 * m2_2
        #echo "hessian3: ", hessianB

        gradient = newGradient
        let fx = f(x)
        fNorm = abs(fx)
        gradNorm = vectorNorm(gradient)
        iters += 1
    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc bfgs*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], options: OptimOptions[U, StandardOptions] = bfgsOptions[U](), analyticGradient: proc(x: Tensor[U]): Tensor[T] = nil): Tensor[U] =
    ## BFGS (Broyden–Fletcher–Goldfarb–Shanno) method for optimization.
    ## 
    ## Inputs:
    ## - f: The function to optimize. It should take as input a 1D Tensor of the input variables and return a scalar.
    ## - options: Options object (see `bfgsOptions` for constructing one)
    ## - analyticGradient: The analytic gradient of `f` taking in and returning a 1D Tensor. If not provided, a finite difference approximation will be performed instead.
    ## 
    ## Returns:
    ## - The final solution for the parameters. Either because a (local) minimum was found or because the maximum number of iterations was reached.
    # Use gemm and gemv with preallocated Tensors and setting beta = 0
    var alpha = options.alpha
    var x = x0.clone()
    let xLen = x.shape[0]
    var fNorm = abs(f(x))
    var gradient = 0.01*analyticOrNumericGradient(analyticGradient, f, x0, options)
    var gradNorm = vectorNorm(gradient)
    var hessianB = eye[T](xLen) # inverse of the approximated hessian
    var p = newTensor[U](xLen)
    var tempVector1 = newTensor[U](xLen, 1)
    var tempVector2 = newTensor[U](1, xLen)
    var iters: int
    while gradNorm > options.tol*(1 + fNorm) and iters < 10000:
        # We are using hessianB in calculating it so we are modifying it prior to its use!


        #echo "Hessian iter ", iters, ": ", hessianB
        #let p = -hessianB * gradient.reshape(xLen, 1)
        gemv(-1.0, hessianB, gradient.reshape(xLen, 1), 0.0, p)
        #echo "p iter ", iters, ": ", p
        #echo "x iter ", iters, ": ", x
        #echo "gradient iter ", iters, ": ", gradient
        line_search(alpha, p, x, f, options.lineSearchCriterion, options.fastMode)
        x += alpha * p
        let newGradient = analyticOrNumericGradient(analyticGradient, f, x, options) #tensorGradient(f, x, fastMode=options.fastMode)
        let sk = alpha * p.reshape(xLen, 1)
        
        let yk = (newGradient - gradient).reshape(xLen, 1)
        let sk_yk_dot = dot(sk.squeeze, yk.squeeze)
        # gemm(1.0, hessianB, hessianB, 0.0, newGradient)
        # Do the calculation in steps, minimizing allocations
        # sk * sk.transpose is matrix × matrix
        # how to deal with the vector.T × vector = Matrix? Use gemm? gemm() can be used as += if beta is set to 1.0
        #echo "aim: ", hessianB + (sk_yk_dot + dot(yk.squeeze, squeeze(hessianB * yk))) / (sk_yk_dot * sk_yk_dot) * (sk * sk.transpose) - ((hessianB * yk) * sk.transpose + sk * (yk.transpose * hessianB)) / sk_yk_dot
        #echo "real Hessian: ", hessianB + (sk_yk_dot + dot(yk.squeeze, squeeze(hessianB * yk))) / (sk_yk_dot * sk_yk_dot) * (sk * sk.transpose)
        let prefactor1 = (sk_yk_dot + dot(yk.squeeze, squeeze(hessianB * yk))) / (sk_yk_dot * sk_yk_dot)
        let prefactor2 = 1 / sk_yk_dot
        gemv(1.0, hessianB, yk, 0.0, tempVector1) # temp1 = hessianB * yk
        gemm(1.0, yk.transpose, hessianB, 0.0, tempVector2) # temp2 = yk.transpose * hessianB

        gemm(prefactor1, sk, sk.transpose, 1.0, hessianB) # hessianB += prefactor1 * sk * sk.transpose

        gemm(-prefactor2, tempVector1, sk.transpose, 1.0, hessianB) # hessianB -= prefactor2 * temp1 * sk.transpose

        gemm(-prefactor2, sk, tempVector2, 1.0, hessianB) # hessianB -= prefactor2 * sk * temp2
        #echo "hessian2: ", hessianB # This is correct
        
        # somewhere down here the error occurs!↓

        # reuse vector p:
        #gemv(1.0, hessianB, yk, 0.0, tempVector1) # temp1 = hessianB * yk
        #echo "checkpoint1: ", tempVector1 # this is incorrect!!!

        #
        #
        # Rewrite with transposes as gemv instead!
        # (A * B)^T = B^T * A^T → yk.transpose * hessianB = transpose(hessianB.transpose * yk)
        #

        #gemm(1.0, yk.transpose, hessianB, 0.0, p) 
        #gemv(1.0, hessianB.transpose, yk, 0.0, tempVector) # p = yk.transpose * hessianB
        #tempVector = tempVector.transpose

        
        #echo "hessian3: ", hessianB

        #hessianB += (sk_yk_dot + dot(yk.squeeze, squeeze(hessianB * yk))) / (sk_yk_dot * sk_yk_dot) * (sk * sk.transpose) - ((hessianB * yk) * sk.transpose + sk * (yk.transpose * hessianB)) / sk_yk_dot
        #tempVector = tempVector.transpose # reverse it back to column vector
        
        gradient = newGradient
        let fx = f(x)
        fNorm = abs(fx)
        gradNorm = vectorNorm(gradient)
        iters += 1
    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc lbfgs*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], options: OptimOptions[U, LBFGSOptions[U]] = lbfgsOptions[U](), analyticGradient: proc(x: Tensor[U]): Tensor[T] = nil): Tensor[U] =
    ## LBFGS (Limited-memory  Broyden–Fletcher–Goldfarb–Shanno) method for optimization.
    ## 
    ## Inputs:
    ## - f: The function to optimize. It should take as input a 1D Tensor of the input variables and return a scalar.
    ## - options: Options object (see `lbfgsOptions` for constructing one)
    ## - analyticGradient: The analytic gradient of `f` taking in and returning a 1D Tensor. If not provided, a finite difference approximation will be performed instead.
    ## 
    ## Returns:
    ## - The final solution for the parameters. Either because a (local) minimum was found or because the maximum number of iterations was reached.
    var alpha = options.alpha
    var x = x0.clone()
    let xLen = x.shape[0]
    var fNorm = abs(f(x))
    var gradient = 0.01*analyticOrNumericGradient(analyticGradient, f, x0, options)
    var gradNorm = vectorNorm(gradient)
    var iters: int
    let m = options.algoOptions.savedIterations # number of past iterations to save
    var sk_queue = initDeque[Tensor[U]](m)
    var yk_queue = initDeque[Tensor[T]](m)
    # the problem is the first iteration as the gradient is huge and no adjustments are made
    while gradNorm > options.tol*(1 + fNorm) and iters < 10000:
        #echo "grad: ", gradient
        #echo "x: ", x
        var q = gradient.clone()
        # we want to loop from latest inserted to oldest
        # → we insert at the beginning and pop at the end
        var alphas: seq[U]
        for i in 0 ..< sk_queue.len:
            let rho_i = 1 / dot(sk_queue[i], yk_queue[i])
            let alpha_i = rho_i * dot(sk_queue[i], q)
            q -= alpha_i * yk_queue[i]
            alphas.add alpha_i
        let gamma = if sk_queue.len == 0: 1.0 else: dot(sk_queue[0], yk_queue[0]) / dot(yk_queue[0], yk_queue[0])
        #echo gamma
        var z = gamma * q
        for i in countdown(sk_queue.len - 1, 0):
            let rho_i = 1 / dot(sk_queue[i], yk_queue[i])
            let beta_i = rho_i * dot(yk_queue[i], z)
            let alpha_i = alphas[i]
            z += sk_queue[i] * (alpha_i - beta_i)
        z = -z
        let p = z
        #echo "q: ", q
        line_search(alpha, p, x, f, options.lineSearchCriterion, options.fastMode)
        x += alpha * p
        sk_queue.addFirst alpha*p
        let newGradient = analyticOrNumericGradient(analyticGradient, f, x, options)
        let yk = newGradient - gradient
        yk_queue.addFirst yk
        gradient = newGradient
        let fx = f(x)
        fNorm = abs(fx)
        gradNorm = vectorNorm(gradient)
        if sk_queue.len > m: discard sk_queue.popLast
        if yk_queue.len > m: discard yk_queue.popLast
        iters += 1

    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc levmarq*[U; T: not Tensor](f: proc(params: Tensor[U], x: U): T, params0: Tensor[U], xData: Tensor[U], yData: Tensor[T], options: OptimOptions[U, LevmarqOptions[U]] = levmarqOptions[U](), yError: Tensor[T] = ones_like(yData)): Tensor[U] =
    ## Levenberg-Marquardt for non-linear least square solving. Basically it fits parameters of a function to data samples.
    ## 
    ## Input:
    ## - f: The function you want to fit the data to. The first argument should be a 1D Tensor with the values of the parameters
    ##      and the second argument is the value if the independent variable to evaluate the function at.
    ## - params0: The starting guess for the parameter values as a 1D Tensor.
    ## - yData: The measured values of the dependent variable as 1D Tensor.
    ## - xData: The values of the independent variable as 1D Tensor.
    ## - options: Object with all the options like `tol` and `lambda0`. (see `levmarqOptions`)
    ## - yError: The uncertainties of the `yData` as 1D Tensor. Ideally these should be the 1σ standard deviation.
    ## 
    ## Returns:
    ## - The final solution for the parameters. Either because a (local) minimum was found or because the maximum number of iterations was reached.
    assert xData.rank == 1
    assert yData.rank == 1
    assert params0.rank == 1
    let xLen = xData.shape[0]
    let yLen = yData.shape[0]
    let paramsLen = params0.shape[0]
    assert xLen == yLen

    let residualFunc = # proc that returns the residual vector
        proc (params: Tensor[U]): Tensor[T] =
            result = map3_inline(xData, yData, yError):
                (f(params, x) - y) / z 

    let errorFunc = # proc that returns the scalar error
        proc (params: Tensor[U]): T =
            let res = residualFunc(params)
            result = dot(res, res)

    var lambdaCoeff = options.algoOptions.lambda0

    var params = params0.clone()
    var gradient = tensorGradient(residualFunc, params, fastMode=options.fastMode)
    var residuals = residualFunc(params)
    var resNorm = vectorNorm(residuals)
    var gradNorm = vectorNorm(squeeze(gradient * residuals.reshape(xLen, 1)))
    var iters: int
    let eyeNN = eye[T](paramsLen)
    while gradNorm > options.tol*(1 + resNorm) and iters < 10000:
        let rhs = -gradient * residuals.reshape(xLen, 1)
        let lhs = gradient * gradient.transpose + lambdaCoeff * eyeNN
        let p = solve(lhs, rhs)
        params += p * options.alpha
        gradient = tensorGradient(residualFunc, params, fastMode=options.fastMode)
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
    if iters == 10000:
        echo "levmarq reached maximum number of iterations!"
    result = params


proc inv[T](t: Tensor[T]): Tensor[T] =
    result = solve(t, eye[T](t.shape[0]))

proc getDiag[T](t: Tensor[T]): Tensor[T] =
    let n = t.shape[0]
    result = newTensor[T](n)
    for i in 0 ..< n:
      result[i] = t[i,i]

proc paramUncertainties*[U; T](params: Tensor[U], fitFunc: proc(params: Tensor[U], x: U): T, xData: Tensor[U], yData: Tensor[T], yError: Tensor[T], returnFullCov = false): Tensor[T] =
    ## Returns the whole covariance matrix or only the diagonal elements for the parameters in `params`.
    ## 
    ## Inputs:
    ## - params: The parameters in a 1D Tensor that the uncertainties are wanted for.
    ## - fitFunc: The function used for fitting the parameters. (see `levmarq` for more)
    ## - xData: The values of the independent variable as 1D Tensor.
    ## - yData: The measured values of the dependent variable as 1D Tensor.
    ## - yError: The uncertainties of the `yData` as 1D Tensor. Ideally these should be the 1σ standard deviation.
    ## - returnFullConv: If true, the full covariance matrix will be returned as a 2D Tensor, else only the diagonal elements will be returned as a 1D Tensor.
    ## 
    ## Returns:
    ## 
    ## The uncertainties of the parameters in the form of a covariance matrix (or only the diagonal elements). 
    ## 
    ## Note: it is the covariance that is returned, so if you want the standard deviation you have to
    ## take the square root of it.
    proc fError(params: Tensor[U]): T =
        let yCurve = xData.map_inline:
            fitFunc(params, x)
        result = chi2(yData, yCurve, yError)
    
    let dof = xData.size - params.size
    let sigma2 = fError(params) / T(dof)
    let H = tensorHessian(fError, params)
    let cov = sigma2 * H.inv()

    if returnFullCov:
        result = cov
    else:
        result = cov.getDiag()

    

    

        
when isMainModule:
    import benchy
    # Steepest descent:
    proc f1(x: Tensor[float]): float =
        # result = -20*exp(-0.2*sqrt(0.5 * (x[0]*x[0] + x[1]*x[1]))) - exp(0.5*(cos(2*PI*x[0]) + cos(2*PI*x[1]))) + E + 20 # Ackley
        result = (1 - x[0])^2 + 100*(x[1] - x[0]^2)^2
        if x.shape[0] > 2:
            for ix in x[2 .. _]:
                result += (ix - 1)^2
    
    #let x0 = [-0.5, 2.0].toTensor
    let x0 = ones[float](100) * -1
    #[ let sol1 = steepestDescent(f1, x0, tol=1e-8, alpha=0.001, fastMode=true)
    echo sol1
    echo f1(sol1)
    echo "Newton: ", newton(f1, x0, tol=1e-8, fastMode=false)
    echo "Newton: ", newton(f1, x0, tol=1e-8, fastMode=true)
    echo "BFGS: ", bfgs(f1, x0, tol=1e-8, fastMode=false)
    echo "BFGS: ", bfgs(f1, x0, tol=1e-8, fastMode=true)
    echo "LBFGS:", lbfgs(f1, x0, tol=1e-8, fastMode=false)
    echo "BFGS: ", bfgs(f1, x0, tol=1e-8, fastMode=false)
    echo "BFGS opt: ", bfgs_optimized(f1, x0, tol=1e-8, fastMode=false)
    echo "BFGS: ", bfgs(f1, x0, tol=1e-8, fastMode=true)
    echo "BFGS opt: ", bfgs_optimized(f1, x0, tol=1e-8, fastMode=true) ]#

    #[ echo bfgs_optimized(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=None)
    echo bfgs_optimized(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=Armijo)
    echo bfgs_optimized(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=Wolfe)
    echo bfgs_optimized(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=WolfeStrong)
    echo lbfgs(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=None)
    echo lbfgs(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=Armijo)
    echo lbfgs(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=Wolfe)
    echo lbfgs(f1, x0, tol=1e-8, alpha=0.001, fastMode=false, criterion=WolfeStrong) ]#

    #[ timeIt "steepest slow mode None":
        keep bfgs_optimized(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=None)

    timeIt "steepest slow mode Armijo":
        keep bfgs_optimized(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=Armijo)

    timeIt "steepest slow mode Wolfe":
        keep bfgs_optimized(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=Wolfe)

    timeIt "steepest slow mode WolfeStrong":
        keep bfgs_optimized(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=WolfeStrong)

    timeIt "steepest slow mode None":
        keep lbfgs(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=None)

    timeIt "steepest slow mode Armijo":
        keep lbfgs(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=Armijo)

    timeIt "steepest slow mode Wolfe":
        keep lbfgs(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=Wolfe)

    timeIt "steepest slow mode WolfeStrong":
        keep lbfgs(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=WolfeStrong) ]#
#[     timeIt "newton slow mode":
        keep newton(f1, x0, tol=1e-8, fastMode=false)
    timeIt "newton fast mode":
        keep newton(f1, x0, tol=1e-8, fastMode=true)
    timeIt "bfgs slow mode":
        keep bfgs(f1, x0, tol=1e-8, fastMode=false)
    timeIt "bfgs fast mode":
        keep bfgs(f1, x0, tol=1e-8, fastMode=true)
    timeIt "lbfgs slow mode":
        keep lbfgs(f1, x0, tol=1e-8, fastMode=false)
    timeIt "lbfgs fast mode":
        keep lbfgs(f1, x0, tol=1e-8, fastMode=true)
    timeIt "optimized bfgs slow mode":
        keep bfgs_optimized(f1, x0, tol=1e-8, fastMode=false)
    timeIt "optimized bfgs fast mode":
        keep bfgs_optimized(f1, x0, tol=1e-8, fastMode=true) 
    timeIt "steepest fast mode":
        keep steepestDescent(f1, x0, tol=1e-8, alpha=0.001, fastMode=true) ]#


    # Lev-Marq:
#[     proc fFit(params: Tensor[float], x: float): float =
        params[0] + params[1] * x + params[2] * x*x
    
    let xData = arraymancer.linspace(0, 10, 100)
    let yData = 1.5 +. xData * 6.28 + xData *. xData * -5.79 + randomNormalTensor[float](xData.shape[0], 0.0, 0.1)
    let params0 = [0.0, 0.0, 0.0].toTensor
    echo levmarq(fFit, params0, xData, yData)
    timeIt "slow mode":
        keep levmarq(fFit, params0, xData, yData, fastMode=false)
    timeIt "fast mode":
        keep levmarq(fFit, params0, xData, yData, fastMode=true) ]#




