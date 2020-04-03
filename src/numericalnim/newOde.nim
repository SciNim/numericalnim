type
    ODEoptions* = object
        dt*: float
        dtMax*: float
        dtMin*: float
        tStart*: float
        absTol*: float
        relTol*: float
        scaleMax*: float
        scaleMin*: float

proc newODEoptions*(dt: float = 1e-4, absTol: float = 1e-4, relTol: float = 1e-4, dtMax: float = 1e-2, dtMin: float = 1e-4, scaleMax: float = 4.0, scaleMin: float = 0.1,
                    tStart: float = 0.0): ODEoptions =
    ## Create a new ODEoptions object.
    ##
    ## Input:
    ##   - dt: The time step to use in fixed timestep integrators.
    ##   - absTol: The absolute error tolerance to use in adaptive timestep integrators.
    ##   - relTol: The relative error tolerance to use in adaptive timestep integrators.
    ##   - dtMax: The maximum timestep allowed in adaptive timestep integrators.
    ##   - dtMin: The minimum timestep allowed in adaptive timestep integrators.
    ##   - scaleMax: The maximum increase factor for the time step dt between two steps.
    ##   - ScaleMin: The minimum factor for the time step dt between two steps.
    ##   - tStart: The time to start the ODE-solver at. The time the initial
    ##     conditions are supplied at.
    ##
    ## Returns:
    ##   - ODEoptions object with the supplied parameters.
    if abs(dtMax) < abs(dtMin):
        raise newException(ValueError, "dtMin must be less than dtMax")
    if abs(scaleMax) < 1:
        raise newException(ValueError, "scaleMax must be bigger than 1")
    if 1 < abs(scaleMin):
        raise newException(ValueError, "scaleMin must be smaller than 1")
    result = ODEoptions(dt: abs(dt), absTol: abs(absTol), relTol: abs(relTol), dtMax: abs(dtMax),
                        dtMin: abs(dtMin), scaleMax: abs(scaleMax), scaleMin: abs(scaleMin), tStart: tStart)

const DEFAULT_ODEoptions = newODEoptions()

template odeLoop(useFSAL, adaptive: bool, order: float, code: untyped) {.dirty.} =
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
                    when useFSAL:
                        yPositive.add(hermiteSpline(tPositive[denseIndex], lastIter.t,
                                                    t, lastIter.y, y, lastIter.dy, FSAL))
                    else:
                        yPositive.add(hermiteSpline(tPositive[denseIndex], lastIter.t,
                                                    t, lastIter.y, y, lastIter.dy, f(t, y)))
                    denseIndex += 1
                    if tPositive.high < denseIndex:
                        break
            dt = min(dt, tEnd - t)
            if useDense:
                if useFSAL:
                    lastIter = (t: t, y: y, dy: FSAL)
                else:
                    lastIter = (t: t, y: y, dy: f(t, y))
            code
            t += dt
            when adaptive:
                if error == 0.0:
                    dt *= 5
                else:
                    dt = dt * min(4, max(0.125, 0.9 * pow(1/error, 1/order)))
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
                    when useFSAL:
                        yNegative.add(hermiteSpline(-tNegative[denseIndex], lastIter.t,
                                                    t, lastIter.y, y, lastIter.dy, FSAL))
                    else:
                        yNegative.add(hermiteSpline(-tNegative[denseIndex], lastIter.t, t,
                                                    lastIter.y, y, lastIter.dy, g(t, y)))
                    denseIndex += 1
                    if tNegative.high < denseIndex:
                        break
            dt = min(dt, tEnd - t)
            if useDense:
                if useFSAL:
                    lastIter = (t: t, y: y, dy: FSAL)
                else:
                    lastIter = (t: t, y: y, dy: g(t, y))
            code
            t += dt
            when adaptive:
                if error == 0.0:
                    dt *= 5
                else:
                    dt = dt * min(4, max(0.125, 0.9 * pow(1/error, 1/order)))
                if dt < dtMin:
                    dt = dtMin
                elif dtMax < dt:
                    dt = dtMax
        yNegative.add(y)
    return (tNegative.reversed().concat(tZero).concat(tPositive),
            yNegative.reversed().concat(yZero).concat(yPositive))


proc RK4*[T](f: proc(t: float, y: T): T, y0: T, tspan: openArray[float],
                  options: ODEoptions = DEFAULT_ODEoptions): (seq[float], seq[T]) =
    var k1, k2, k3, k4: T
    odeLoop(false, false, 4.0):
        k1 = f(t, y)
        k2 = f(t + 0.5*dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5*dt, y + 0.5 * dt * k2)
        k4 = f(t +     dt, y +       dt * k3)
        yNew = y + dt / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
        # y, yNew, dt is injected from template