import strformat
import arraymancer, sequtils
import math

proc steepest_descent*(deriv: proc(x: float64): float64, start, gamma, precision: float64, max_iters: int):(float64) {.inline.} =
    var
        current = 0.0
        x = start

    for i in 0 .. max_iters:
        current = x
        x = current - gamma * deriv(current)
        if abs(x - current) <= precision:
            break

    return x

proc conjugate_gradient*(A, b, x_0: Tensor, tolerance: float64): Tensor =

    var r = b - (A * x_0)
    var p = r
    var rsold = (r.transpose() * r)[0,0]
    var
        Ap = A
        alpha = 0.0
        rsnew = 0.0
        x = x_0
        Ap_p = 0.0

    for i in 1 .. b.shape[0]:
        Ap = A * p
        Ap_p = (p.transpose() * Ap)[0,0]
        alpha = rsold / Ap_p
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r.transpose() * r)[0,0]
        if sqrt(rsnew) < tolerance:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x
    

#var v = toSeq([1.0, 2.0, 4.4]).toTensor.reshape(3,1)
#var A = toSeq(1..9).toTensor.reshape(3,3).astype(float64)
#var b = toSeq([1.0,2.0,3.0]).toTensor.reshape(3,1)
#let tol = 0.001


