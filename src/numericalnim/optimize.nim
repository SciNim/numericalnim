import std/[strformat, sequtils, math, deques]
import arraymancer
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

type LineSearchCriterion = enum
    Armijo, Wolfe, WolfeStrong, None

proc line_search*[U, T](alpha: var U, p: Tensor[T], x0: Tensor[U], f: proc(x: Tensor[U]): T, criterion: LineSearchCriterion, fastMode: bool = false) =
    # note: set initial alpha for the methods as 1 / sqrt(dot(grad, grad)) so first step has length 1.
    if criterion == None:
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




    

proc steepestDescent*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(0.1), tol: U = U(1e-6), fastMode: bool = false, criterion: LineSearchCriterion = None): Tensor[U] =
    ## Minimize scalar-valued function f. 
    var alpha = alpha
    var x = x0.clone()
    var fNorm = abs(f(x0))
    var gradient = tensorGradient(f, x0, fastMode=fastMode)
    var gradNorm = vectorNorm(gradient)
    var iters: int
    while gradNorm > tol*(1 + fNorm) and iters < 10000:
        let p = -gradient
        line_search(alpha, p, x, f, criterion, fastMode)
        x += alpha * p
        let fx = f(x)
        fNorm = abs(fx)
        gradient = tensorGradient(f, x, fastMode=fastMode)
        gradNorm = vectorNorm(gradient)
        iters += 1
    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc newton*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(1), tol: U = U(1e-6), fastMode: bool = false): Tensor[U] =
    var x = x0.clone()
    var fNorm = abs(f(x))
    var gradient = tensorGradient(f, x, fastMode=fastMode)
    var gradNorm = vectorNorm(gradient)
    var hessian = tensorHessian(f, x)
    var iters: int
    while gradNorm > tol*(1 + fNorm) and iters < 10000:
        let p = -solve(hessian, gradient)
        x += alpha * p
        let fx = f(x)
        fNorm = abs(fx)
        gradient = tensorGradient(f, x, fastMode=fastMode)
        gradNorm = vectorNorm(gradient)
        hessian = tensorHessian(f, x)
        iters += 1
    if iters >= 10000:
        discard "Limit of 10000 iterations reached!"
    #echo iters, " iterations done!"
    result = x

proc bfgs*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(1), tol: U = U(1e-6), fastMode: bool = false): Tensor[U] =
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

proc bfgs_optimized*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(1), tol: U = U(1e-6), fastMode: bool = false, criterion: LineSearchCriterion = None): Tensor[U] =
    # Use gemm and gemv with preallocated Tensors and setting beta = 0
    var alpha = alpha
    var x = x0.clone()
    let xLen = x.shape[0]
    var fNorm = abs(f(x))
    var gradient = 0.01*tensorGradient(f, x, fastMode=fastMode)
    var gradNorm = vectorNorm(gradient)
    var hessianB = eye[T](xLen) # inverse of the approximated hessian
    var p = newTensor[U](xLen)
    var tempVector1 = newTensor[U](xLen, 1)
    var tempVector2 = newTensor[U](1, xLen)
    var iters: int
    while gradNorm > tol*(1 + fNorm) and iters < 10000:
        # We are using hessianB in calculating it so we are modifying it prior to its use!


        #echo "Hessian iter ", iters, ": ", hessianB
        #let p = -hessianB * gradient.reshape(xLen, 1)
        gemv(-1.0, hessianB, gradient.reshape(xLen, 1), 0.0, p)
        #echo "p iter ", iters, ": ", p
        #echo "x iter ", iters, ": ", x
        #echo "gradient iter ", iters, ": ", gradient
        line_search(alpha, p, x, f, criterion, fastMode)
        x += alpha * p
        let newGradient = tensorGradient(f, x, fastMode=fastMode)
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

proc lbfgs*[U; T: not Tensor](f: proc(x: Tensor[U]): T, x0: Tensor[U], alpha: U = U(1), tol: U = U(1e-6), fastMode: bool = false, m: int = 10, criterion: LineSearchCriterion = None): Tensor[U] =
    var alpha = alpha
    var x = x0.clone()
    let xLen = x.shape[0]
    var fNorm = abs(f(x))
    var gradient = 0.01*tensorGradient(f, x, fastMode=fastMode)
    var gradNorm = vectorNorm(gradient)
    var iters: int
    #let m = 10 # number of past iterations to save
    var sk_queue = initDeque[Tensor[U]](m)
    var yk_queue = initDeque[Tensor[T]](m)
    # the problem is the first iteration as the gradient is huge and no adjustments are made
    while gradNorm > tol*(1 + fNorm) and iters < 10000:
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
        line_search(alpha, p, x, f, criterion, fastMode)
        x += alpha * p
        sk_queue.addFirst alpha*p
        let newGradient = tensorGradient(f, x, fastMode=fastMode)
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

proc levmarq*[U; T: not Tensor](f: proc(params: Tensor[U], x: U): T, params0: Tensor[U], xData: Tensor[U], yData: Tensor[T], alpha = U(1), tol: U = U(1e-6), lambda0: U = U(1), fastMode = false): Tensor[U] =
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

    timeIt "steepest slow mode None":
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
        keep lbfgs(f1, x0, tol=1e-8, alpha=1, fastMode=false, criterion=WolfeStrong)
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




