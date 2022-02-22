import std/[unittest, math, sequtils, random] 
import arraymancer
import numericalnim

suite "1D":
    test "steepest_descent func":
        proc df(x: float): float = 4 * x^3 - 9.0 * x^2
        let start = 6.0
        let gamma = 0.01
        let precision = 0.00001
        let max_iters = 10000
        let correct = 2.24996
        let value = steepest_descent(df, start, gamma, precision, max_iters)
        check isClose(value, correct, tol = 1e-5)

    test "steepest_descent func starting at zero":
        proc df(x: float): float = 4 * x^3 - 9.0 * x^2 + 4
        let start = 0.0
        let correct = -0.59301
        let value = steepest_descent(df, start)
        check isClose(value, correct, tol = 1e-5)

    #[
    test "conjugate_gradient func":
        var A = toSeq([4.0, 1.0, 1.0, 3.0]).toTensor.reshape(2,2).astype(float64)
        var x = toSeq([2.0, 1.0]).toTensor.reshape(2,1)
        var b = toSeq([1.0,2.0]).toTensor.reshape(2,1)
        let tol = 0.001
        let correct = toSeq([0.090909, 0.636363]).toTensor.reshape(2,1).astype(float64)

        let value = conjugate_gradient(A, b, x, tol)
        check isClose(value, correct, tol = 1e-5)
    ]#
    test "Newtons 1 dimension func":
        proc f(x:float64): float64 = (1.0 / 3.0) * x ^ 3 - 2 * x ^ 2 + 3 * x
        proc df(x:float64): float64 = x ^ 2 - 4 * x + 3
        let x = 0.5
        let correct = 0.0
        let value = newtons(f, df, x, 0.000001, 1000)
        check isClose(value, correct, tol=1e-5)

    test "Newtons 1 dimension func default args":
        proc f(x:float64): float64 = (1.0 / 3.0) * x ^ 3 - 2 * x ^ 2 + 3 * x
        proc df(x:float64): float64 = x ^ 2 - 4 * x + 3
        let x = 0.5
        let correct = 0.0
        let value = newtons(f, df, x)
        check isClose(value, correct, tol=1e-5)

    test "Newtons unable to find a root":
        proc bad_f(x:float64): float64 = pow(E, x) + 1
        proc bad_df(x:float64): float64 = pow(E, x)
        expect(ArithmeticError):
            discard newtons(bad_f, bad_df, 0, 0.000001, 1000)

    test "Secant 1 dimension func":
        proc f(x:float64): float64 = (1.0 / 3.0) * x ^ 3 - 2 * x ^ 2 + 3 * x
        let x = [1.0, 0.5]
        let correct = 0.0
        let value = secant(f, x, 1e-5, 10)
        check isClose(value, correct, tol=1e-5)


###############################
## Multi-dimensional methods ##
###############################

suite "Multi-dim":
    proc bananaFunc(x: Tensor[float]): float =
        ## Function of 2 variables with minimum at (1, 1)
        ## And it looks like a banana 🍌
        result = (1 - x[0])^2 + 100*(x[1] - x[0]^2)^2
    
    let x0 = [-1.0, -1.0].toTensor()
    let correct = [1.0, 1.0].toTensor()

    test "Steepest Gradient":
        let xSol = steepestDescent(bananaFunc, x0.clone)
        for x in abs(correct - xSol):
            check x < 2e-2
    
    test "Newton":
        let xSol = newton(bananaFunc, x0.clone)
        for x in abs(correct - xSol):
            check x < 3e-10

    test "BFGS":
        let xSol = bfgs(bananaFunc, x0.clone)
        for x in abs(correct - xSol):
            check x < 3e-7

    test "L-BFGS":
        let xSol = lbfgs(bananaFunc, x0.clone)
        for x in abs(correct - xSol):
            check x < 7e-10

    let correctParams = [10.4, -0.45].toTensor()
    proc fitFunc(params: Tensor[float], x: float): float =
        params[0] * exp(params[1] * x)

    let xData = arraymancer.linspace(0.0, 10.0, 100)
    randomize(1337)
    let yData = correctParams[0] * exp(correctParams[1] * xData) + randomNormalTensor([100], 0.0, 1e-2)

    let params0 = [0.0, 0.0].toTensor()

    test "levmarq":
        let paramsSol = levmarq(fitFunc, params0, xData, yData)
        for x in abs(paramsSol - correctParams):
            check x < 1.3e-3


