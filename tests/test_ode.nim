import unittest, math, sequtils
import arraymancer
import numericalnim

proc f(x: float, y: float): float = -0.1 * y
proc fVector(x: float, y: Vector[float]): Vector[float] = -0.1 * y
proc fTensor(x: float, y: Tensor[float]): Tensor[float] = -0.1 * y
proc correct_answer(x: float): float = exp(-0.1*x)
let oo = newODEoptions(tol=1e-8, dt=1e-6)
let ooVector = newODEoptions(tol=1e-8, dt=1e-2)
let ooTensor = newODEoptions(tol=1e-8, dt=1e-2)
let y0 = 1.0
let y0Vector = newVector(@[y0, y0, y0])
let y0Tensor = @[y0, y0, y0].toTensor()
let tspan = linspace(-10.0, 10.0, 100)
let correctY = map(tspan, correct_answer)
var correctYVectorSeq: seq[Vector[float]]
var correctYTensor: seq[Tensor[float]]
for v in correctY:
    correctYTensor.add(@[v, v, v].toTensor())
    correctYVectorSeq.add(newVector(@[v, v, v]))
let correctYVector = newVector(correctYVectorSeq)

test "DOPRI54, default":
    let (t, y) = solveODE(f, y0, tspan, integrator="dopri54")
    check t == tspan
    for i, val in y:
        check isClose(val, correctY[i], tol=1e-4)

test "DOPRI54, tol = 1e-8":
    let (t, y) = solveODE(f, y0, tspan, integrator="dopri54", options=oo)
    check t == tspan
    for i, val in y:
        check isClose(val, correctY[i], tol=1e-8)

test "RK4, default":
    let (t, y) = solveODE(f, y0, tspan, integrator="rk4")
    check t == tspan
    for i, val in y:
        check isClose(val, correctY[i], tol=1e-4)

test "RK4, dt = 1e-6":
    let (t, y) = solveODE(f, y0, tspan, integrator="rk4", options=oo)
    check t == tspan
    for i, val in y:
        check isClose(val, correctY[i], tol=1e-8)

test "DOPRI54 Vector, default":
    let (t, y) = solveODE(fVector, y0Vector, tspan, integrator="dopri54")
    check t == tspan
    for i, val in y:
        check isClose(val, correctYVector[i], tol=1e-4)

test "DOPRI54 Vector, tol = 1e-8":
    let (t, y) = solveODE(fVector, y0Vector, tspan, integrator="dopri54", options=ooVector)
    check t == tspan
    for i, val in y:
        check isClose(val, correctYVector[i], tol=1e-8)

test "RK4 Vector, default":
    let (t, y) = solveODE(fVector, y0Vector, tspan, integrator="rk4")
    check t == tspan
    for i, val in y:
        check isClose(val, correctYVector[i], tol=1e-4)

test "RK4 Vector, dt = 1e-2":
    let (t, y) = solveODE(fVector, y0Vector, tspan, integrator="rk4", options=ooVector)
    check t == tspan
    for i, val in y:
        check isClose(val, correctYVector[i], tol=1e-8)

test "DOPRI54 Tensor, default":
    let (t, y) = solveODE(fTensor, y0Tensor, tspan, integrator="dopri54")
    check t == tspan
    for i, val in y:
        check isClose(val, correctYTensor[i], tol=1e-4)

test "DOPRI54 Tensor, tol = 1e-8":
    let (t, y) = solveODE(fTensor, y0Tensor, tspan, integrator="dopri54", options=ooTensor)
    check t == tspan
    for i, val in y:
        check isClose(val, correctYTensor[i], tol=1e-8)

test "RK4 Tensor, default":
    let (t, y) = solveODE(fTensor, y0Tensor, tspan, integrator="rk4")
    check t == tspan
    for i, val in y:
        check isClose(val, correctYTensor[i], tol=1e-4)

test "RK4 Tensor, dt = 1e-2":
    let (t, y) = solveODE(fTensor, y0Tensor, tspan, integrator="rk4", options=ooTensor)
    check t == tspan
    for i, val in y:
        check isClose(val, correctYTensor[i], tol=1e-8)
