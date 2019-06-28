import algorithm, strutils, math, strformat, sequtils
import arraymancer
import utils

type 
    ODEoptions* = object
        dt*: float
        tol*: float
        dtMax*: float
        dtMin*: float
        tStart*: float



proc newODEoptions*(dt = 1e-4, tol = 1e-4, dtMax = 1e-2, dtMin = 1e-8, tStart = 0.0): ODEoptions =
    if dtMax < dtMin:
        raise newException(ValueError, "dtMin must be less than dtMax")
    result = ODEoptions(dt: abs(dt), tol: abs(tol), dtMax: abs(dtMax), dtMin: abs(dtMin), tStart: tStart)

const DEFAULT_ODEoptions = newODEoptions(dt = 1e-4, tol = 1e-4, dtMax = 1e-2, dtMin = 1e-8, tStart = 0.0)


proc RK4_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float, options: ODEoptions): (T, T, float, float) =
    var k1, k2, k3, k4: T
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5*dt, y + 0.5 * dt * k2)
    k4 = f(t +     dt, y +       dt * k3)
    let yNew = y + dt / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return (yNew, yNew, dt, 0.0)


proc DOPRI54_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float, options: ODEoptions): (T, T, float, float) =
    const
        c2 = 1.0/5.0
        c3 = 3.0/10.0
        c4 = 4.0/5.0
        c5 = 8.0/9.0
        c6 = 1.0
        c7 = 1.0
        a21 = 1.0/5.0
        a31 = 3.0/40.0
        a32 = 9.0/40.0
        a41 = 44.0/45.0
        a42 = -56.0/15.0
        a43 = 32.0/9.0
        a51 = 19372.0/6561.0
        a52 = -25360.0/2187.0
        a53 = 64448.0/6561.0
        a54 = -212.0/729.0
        a61 = 9017.0/3168.0
        a62 = -355.0/33.0
        a63 = 46732.0/5247.0
        a64 = 49.0/176.0
        a65 = -5103.0/18656.0
        a71 = 35.0/384.0
        a72 = 0.0
        a73 = 500.0/1113.0
        a74 = 125.0/192.0
        a75 = -2187.0/6784.0
        a76 = 11.0/84.0
        # Fifth order
        b1 = a71
        b2 = a72
        b3 = a73
        b4 = a74
        b5 = a75
        b6 = a76
        # Fourth order
        bHat1 = 5179.0/57600.0
        bHat2 = 0.0
        bHat3 = 7571.0/16695.0
        bHat4 = 393.0/640.0
        bHat5 = -92097.0/339200.0
        bHat6 = 187.0/2100.0
        bHat7 = 1.0/40.0
    let tol = options.tol
    let dtMax = options.dtMax
    let dtMin = options.dtMin
    var k1, k2, k3, k4, k5, k6, k7: T
    var yNew, yLow: T
    var error: float
    var limitCounter = 0
    var dt = dt
    while true and limitCounter < 2:
        k1 = FSAL
        k2 = f(t + dt*c2, y + dt * (a21 * k1))
        k3 = f(t + dt*c3, y + dt * (a31 * k1 + a32 * k2))
        k4 = f(t + dt*c4, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + dt*c5, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + dt*c6, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + dt*c7, y + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        yNew = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        yLow = y + dt * (bHat1 * k1 + bHat2 * k2 + bHat3 * k3 + bHat4 * k4 + bHat5 * k5 + bHat6 * k6 + bHat7 * k7)
        error = calcError(yNew, yLow)
        if error <= tol:
            break
        dt = 0.9 * dt * pow(tol/error, 1/5)
        if abs(dt) < dtMin:
            dt = dtMin
            limitCounter += 1
        elif dtMax < abs(dt):
            dt = dtMax
    result = (yNew, k7, dt, error)


proc adaptiveODE[T](f: proc(t: float, y: T): T, y0: T, tspan: openArray[float], options: ODEoptions = DEFAULT_ODEoptions, integrator: proc(f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float, options: ODEoptions): (T, T, float, float), useFSAL = false, order: float, adaptive = false): (seq[float], seq[T]) =
    let t0 = options.tStart
    var t = t0
    var tPositive, tNegative: seq[float]
    tPositive = tspan.filter(proc(x: float): bool = x > t0)
    tnegative = tspan.filter(proc(x: float): bool = x < t0).reversed()
    var yPositive, yNegative: seq[T]
    var y = y0.clone()
    var yZero: seq[T] = @[]
    var tZero: seq[float] = @[]
    if t0 in tspan:
        yZero.add(y)
        tZero.add(t0)
    let tol = options.tol
    let dtMax = options.dtMax
    let dtMin = options.dtMin
    var dt, dtInit: float
    if adaptive:
        dtInit = sqrt(dtMax * dtMin)
        dt = dtInit
    else:
        dtInit = options.dt
        dt = dtInit
    var useDense: bool
    var lastIter = (t: t0, y: y, dy: f(t0, y))
    if tspan.len == 2:
        useDense = false
    else:
        useDense = true
    var denseIndex = 0

    var error: float
    var FSAL = f(t0, y)
    var tEnd: float
    if 0 < tPositive.len:
        dt = dtInit
        tEnd = max(tPositive)
        while t < tEnd:
            if useDense:
                if tPositive.high < denseIndex:
                    break
                while tPositive[denseIndex] <= t:
                    if useFSAL:
                        yPositive.add(hermiteSpline(tPositive[denseIndex], lastIter.t, t, lastIter.y, y, lastIter.dy, FSAL))
                    else:
                        yPositive.add(hermiteSpline(tPositive[denseIndex], lastIter.t, t, lastIter.y, y, lastIter.dy, f(t, y)))
                    denseIndex += 1
                    if tPositive.high < denseIndex:
                        break
            dt = min(dt, tEnd - t)
            if useDense:
                if useFSAL:
                    lastIter = (t: t, y: y, dy: FSAL)
                else:
                    lastIter = (t: t, y: y, dy: f(t, y))
            (y, FSAL, dt, error) = integrator(f, t, y, FSAL, dt, options)
            t += dt
            if adaptive:
                if error == 0.0:
                    dt *= 5
                else:
                    dt = 0.9 * dt * pow(tol/error, 1.0/order)
                if dt < dtMin:
                    dt = dtMin
                elif dtMax < dt:
                    dt = dtMax
        yPositive.add(y)

    if 0 < tNegative.len:
        let g = proc(t: float, y: T): T = -f(-t, y)
        FSAL = g(-t0, y0.clone())
        dt = dtInit
        lastIter = (t: -t0, y: y0.clone(), dy: FSAL)
        tEnd = -min(tNegative)
        t = -t0
        y = y0.clone()
        denseIndex = 0
        while t < tEnd:
            if useDense:
                if tNegative.high < denseIndex:
                    break
                while -tNegative[denseIndex] <= t:
                    if useFSAL:
                        yNegative.add(hermiteSpline(-tNegative[denseIndex], lastIter.t, t, lastIter.y, y, lastIter.dy, FSAL))
                    else:
                        yNegative.add(hermiteSpline(-tNegative[denseIndex], lastIter.t, t, lastIter.y, y, lastIter.dy, g(t, y)))
                    denseIndex += 1
                    if tNegative.high < denseIndex:
                        break
            dt = min(dt, tEnd - t)
            if useDense:
                if useFSAL:
                    lastIter = (t: t, y: y, dy: FSAL)
                else:
                    lastIter = (t: t, y: y, dy: g(t, y))
            (y, FSAL, dt, error) = integrator(g, t, y, FSAL, dt, options)
            t += dt
            if adaptive:
                if error == 0.0:
                    dt *= 5
                else:
                    dt = 0.9 * dt * pow(tol/error, 1.0/order)
                if dt < dtMin:
                    dt = dtMin
                elif dtMax < dt:
                    dt = dtMax
        yNegative.add(y)
    return (tNegative.reversed().concat(tZero).concat(tPositive) , yNegative.reversed().concat(yZero).concat(yPositive))
    

proc solveODE*[T](f: proc(t: float, y: T): T, y0: T, tspan: openArray[float], options: ODEoptions = DEFAULT_ODEoptions, integrator="dopri54"): (seq[float], seq[T]) =
    case integrator.toLower():
        of "dopri54":
            return adaptiveODE(f, y0, tspan.sorted(), options, DOPRI54_step, useFSAL = true, order = 5.0, adaptive = true)
        of "rk4":
            return adaptiveODE(f, y0, tspan.sorted(), options, RK4_step, useFSAL = false, order = 4.0, adaptive = false)