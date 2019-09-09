import strformat
import arraymancer
import sequtils
import math

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
        i = 0

    while abs(x - current) > precision:
        # calculate the next direction to propogate
        current = x
        x = current - gamma * deriv(current)
        i += 1
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

    while abs(f(x_iter)) >= precision and i <= max_iters:
        x_iter = x_iter - (f(x_iter) / deriv(x_iter))
        i += 1
        if i == max_iters:
            raise newException(ArithmeticError, "Maximum iterations for Newtons method exceeded")

    return x_iter - (f(x_iter) / deriv(x_iter))




