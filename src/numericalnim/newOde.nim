import interpolate, ode


type
    ODESolution*[T] = ref object
        interpolant*: InterpolatorType[T]
        t*: seq[float]
        y*: seq[T]
        dy*: seq[T]

proc eval*[T](sol: ODESolution[T], x: float): T =
    sol.interpolant.eval(x)

proc eval*[T](sol: ODESolution[T], x: openArray[float]): seq[T] =
    sol.interpolant.eval(x)

template `+.`(d1, d2: float): float =
    d1 + d2

template `/.`(d1, d2: float): float =
    d1 / d2

template `*.`(d1, d2: float): float =
    d1 * d2

proc size(d: float): int {.inline.} = 1
proc sum(d: float): float {.inline.} = d

template default_between_step(): untyped {.dirty.} =
    if t > nextSaveAt or t >= tEnd:
        ys.add(yNew)
        ts.add(t)
        dys.add(f(t, yNew))
        nextSaveAt += saveEvery

template saveEvery_code(code: untyped): untyped {.dirty.} =
    if t > nextSaveAt or t >= tEnd:
        code

template odeLoop(adaptive: bool, order: float, saveEvery: float, step_code: untyped, between_step_code: untyped): untyped {.dirty.} =
    var t = t0
    var y: T = y0.clone()
    var yNew: T
    var dt, dtInit: float
    var nextSaveAt = t0 + saveEvery
    when adaptive:
        var error: T
        let absTol = options.absTol
        let relTol = options.relTol
        let dtMax = options.dtMax
        let dtMin = options.dtMin
        var limitCounter = 0
        var totalTol, err, err_square: T
        var err_size, errTot: float
    when adaptive:
        dtInit = sqrt(dtMax * dtMin)
        dt = dtInit
    else:
        dtInit = options.dt
        dt = dtInit
    while t < tEnd:
        dt = min(dt, tEnd - t)
        when adaptive: # error, y, yNew, t, dt injected
            limitCounter = 0
            while limitcounter < 2:
                step_code
                totalTol = absTol +. relTol * abs(yNew)
                err = error /. totalTol
                err_square = err *. err
                err_size = err.size.toFloat
                errTot = sqrt(sum(err_square) / err_size)

                if errTot <= 1:
                    break
                dt = dt * min(4, max(0.125, 0.9 * pow(1/errTot, 1/order)))
                if abs(dt) < dtMin:
                    dt = dtMin
                    limitCounter += 1
                elif dtMax < abs(dt):
                    dt = dtMax
        else:
            step_code
        t += dt # must update t with the successful dt before we update it
        when adaptive:
            dt = dt * min(4, max(0.125, 0.9 * pow(1/errTot, 1/order)))
            if abs(dt) < dtMin:
                dt = dtMin
                limitCounter += 1
            elif dtMax < abs(dt):
                dt = dtMax
        # Here we should have t, yNew
        #[if t > nextSaveAt or t >= tEnd:
            ys.add(yNew)
            ts.add(t)
            when useFSAL:
                dys.add(FSAL)
            else:
                dys.add(f(t, yNew))
            nextSaveAt += saveEvery]#
        # Here saving coefficients code block can be inserted
        between_step_code
        y = yNew

proc RK4*[T](f: proc(t: float, y: T): T, y0: T, tEnd: float, saveEvery: float = 0.0, options: ODEoptions = DEFAULT_ODEoptions): ODESolution[T] =
    let t0: float = options.tStart
    let dt_local = options.dt
    var ts = newSeqOfCap[float](int(abs(tEnd - t0) / dt_local))
    ts.add(t0)
    var ys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    ys.add(y0.clone())
    var dys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    dys.add(f(t0, y0))
    # Try to get above code in the template and make ts, ys, dys available outside 
    var k1, k2, k3, k4: T
    odeLoop(adaptive=false, order=4.0, saveEvery=saveEvery) do:
        k1 = f(t, y)
        k2 = f(t + 0.5*dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5*dt, y + 0.5 * dt * k2)
        k4 = f(t +     dt, y +       dt * k3)
        yNew = y + dt / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    do: default_between_step()
    var hermite = newHermiteSpline(ts, ys, dys)
    result = ODESolution[T](t: ts, y: ys, dy: dys, interpolant: hermite)

proc RK21*[T](f: proc(t: float, y: T): T, y0: T, tEnd: float, saveEvery: float = 0.0, options: ODEoptions = DEFAULT_ODEoptions): ODESolution[T] =
    let t0: float = options.tStart
    let dt_local = options.dt
    var ts = newSeqOfCap[float](int(abs(tEnd - t0) / dt_local))
    ts.add(t0)
    var ys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    ys.add(y0.clone())
    var dys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    dys.add(f(t0, y0))
    # Try to get above code in the template and make ts, ys, dys available outside 
    var k1, k2, yLow: T
    odeLoop(adaptive=true, order=2.0, saveEvery=saveEvery) do:
        k1 = f(t, y)
        k2 = f(t + dt, y + dt * k1)
        yNew = y + dt * 0.5 * (k1 + k2) # injected
        yLow = y + dt * k1
        error = yNew - yLow # injected
    do: default_between_step()
    var hermite = newHermiteSpline(ts, ys, dys)
    result = ODESolution[T](t: ts, y: ys, dy: dys, interpolant: hermite)

proc TSIT54*[T](f: proc(t: float, y: T): T, y0: T, tEnd: float, saveEvery: float = 0.0, options: ODEoptions = DEFAULT_ODEoptions): ODESolution[T] =
    let t0: float = options.tStart
    let dt_local = options.dt
    var ts = newSeqOfCap[float](int(abs(tEnd - t0) / dt_local))
    ts.add(t0)
    var ys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    ys.add(y0.clone())
    var dys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    dys.add(f(t0, y0))
    # Try to get above code in the template and make ts, ys, dys available outside 
    const
        c2 = 0.161
        c3 = 0.327
        c4 = 0.9
        c5 = 0.9800255409045097
        c6 = 1.0
        c7 = 1.0
        a21 = 0.161
        a31 = -0.008480655492356989
        a32 = 0.335480655492357
        a41 = 2.8971530571054935
        a42 = -6.359448489975075
        a43 = 4.3622954328695815
        a51 = 5.325864828439257
        a52 = -11.748883564062828
        a53 = 7.4955393428898365
        a54 = -0.09249506636175525
        a61 = 5.86145544294642
        a62 = -12.92096931784711
        a63 = 8.159367898576159
        a64 = -0.071584973281401
        a65 = -0.028269050394068383
        a71 = 0.09646076681806523
        a72 = 0.01
        a73 = 0.4798896504144996
        a74 = 1.379008574103742
        a75 = -3.290069515436081
        a76 = 2.324710524099774
        # Fifth order
        b1 = a71
        b2 = a72
        b3 = a73
        b4 = a74
        b5 = a75
        b6 = a76
        # Fourth order
        bHat1 = -0.001780011052226
        bHat2 = -0.000816434459657
        bHat3 = 0.007880878010262
        bHat4 = -0.144711007173263
        bHat5 = 0.582357165452555
        bHat6 = -0.458082105929187
        bHat7 = 1.0/66.0
    var k1, k2, k3, k4, k5, k6, k7: T
    k7 = f(t0, y0)
    odeLoop(adaptive=true, order=5.0, saveEvery=saveEvery) do:
        k1 = k7
        k2 = f(t + dt*c2, y + dt * (a21 * k1))
        k3 = f(t + dt*c3, y + dt * (a31 * k1 + a32 * k2))
        k4 = f(t + dt*c4, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + dt*c5, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + dt*c6, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + dt*c7, y + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        yNew = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        error = dt * (bHat1 * k1 + bHat2 * k2 + bHat3 * k3 + bHat4 * k4 + bHat5 * k5 + bHat6 * k6 + bHat7 * k7)
    do: 
        saveEvery_code:
            ys.add(yNew)
            ts.add(t)
            dys.add(k7) # FSAL
            nextSaveAt += saveEvery
    var hermite = newHermiteSpline(ts, ys, dys)
    result = ODESolution[T](t: ts, y: ys, dy: dys, interpolant: hermite)

proc DOPRI54*[T](f: proc(t: float, y: T): T, y0: T, tEnd: float, saveEvery: float = 0.0, options: ODEoptions = DEFAULT_ODEoptions): ODESolution[T] =
    let t0: float = options.tStart
    let dt_local = options.dt
    var ts = newSeqOfCap[float](int(abs(tEnd - t0) / dt_local))
    ts.add(t0)
    var ys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    ys.add(y0.clone())
    var dys = newSeqOfCap[T](int(abs(tEnd - t0) / dt_local))
    dys.add(f(t0, y0))
    # Try to get above code in the template and make ts, ys, dys available outside 
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
        bHat1 = b1 - 5179.0/57600.0
        bHat2 = b2 - 0.0
        bHat3 = b3 - 7571.0/16695.0
        bHat4 = b4 - 393.0/640.0
        bHat5 = b5 - -92097.0/339200.0
        bHat6 = b6 - 187.0/2100.0
        bHat7 = 0 - 1.0/40.0
    var k1, k2, k3, k4, k5, k6, k7: T
    k7 = f(t0, y0)
    odeLoop(adaptive=true, order=5.0, saveEvery=saveEvery) do:
        k1 = k7
        k2 = f(t + dt*c2, y + dt * (a21 * k1))
        k3 = f(t + dt*c3, y + dt * (a31 * k1 + a32 * k2))
        k4 = f(t + dt*c4, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + dt*c5, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + dt*c6, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + dt*c7, y + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        yNew = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        error = dt * (bHat1 * k1 + bHat2 * k2 + bHat3 * k3 + bHat4 * k4 + bHat5 * k5 + bHat6 * k6 + bHat7 * k7)
    do: 
        saveEvery_code:
            ys.add(yNew)
            ts.add(t)
            dys.add(k7) # FSAL
            nextSaveAt += saveEvery
    var hermite = newHermiteSpline(ts, ys, dys)
    result = ODESolution[T](t: ts, y: ys, dy: dys, interpolant: hermite)