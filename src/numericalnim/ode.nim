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

const fixedODE* = @["heun2", "ralston2", "kutta3", "heun3", "ralston3", "ssprk3", "ralston4", "kutta4", "rk4"]
const adaptiveODE* = @["rk21", "bs32", "dopri54", "tsit54", "vern65"]
const allODE* = fixedODE.concat(adaptiveODE)


template `+.`(d1, d2: float): float =
    d1 + d2

template `/.`(d1, d2: float): float =
    d1 / d2

template `*.`(d1, d2: float): float =
    d1 * d2

proc size(d: float): int = 1
proc sum(d: float): float = d

template commonAdaptiveMethodCode(yNew, error_y: untyped, order: int, body: untyped) =
    while limitCounter < 2:
        body
        #error = calcError(`e1`, `e2`)
        let totalTol = tol +. tol * abs(`yNew`)
        let err1 = `error_y` /. totalTol
        let err1_square = err1 *. err1
        let size = err1.size.toFloat
        error = sqrt(1 / size * sum(err1_square))
        #echo error, " ", totalTol, " ", abs(`error_y`), " ", err1_square


        if error <= 1:
            break
        dt = dt * min(4, max(0.125, 0.9 * pow(1/error, 1/order)))
        if abs(dt) < dtMin:
            dt = dtMin
            limitCounter += 1
        elif dtMax < abs(dt):
            dt = dtMax

proc newODEoptions*(dt = 1e-4, tol = 1e-4, dtMax = 1e-2, dtMin = 1e-4,
                    tStart = 0.0): ODEoptions =
    ## Create a new ODEoptions object.
    ##
    ## Input:
    ##   - dt: The time step to use in fixed timestep integrators.
    ##   - tol: The error tolerance to use in adaptive timestep integrators.
    ##   - dtMax: The maximum timestep allowed in adaptive timestep integrators.
    ##   - dtMin: The minimum timestep allowed in adaptive timestep integrators.
    ##   - tStart: The time to start the ODE-solver at. The time the initial
    ##     conditions are supplied at.
    ##
    ## Returns:
    ##   - ODEoptions object with the supplied parameters.
    if dtMax < dtMin:
        raise newException(ValueError, "dtMin must be less than dtMax")
    result = ODEoptions(dt: abs(dt), tol: abs(tol), dtMax: abs(dtMax),
                        dtMin: abs(dtMin), tStart: tStart)

const DEFAULT_ODEoptions = newODEoptions(dt = 1e-4, tol = 1e-4, dtMax = 1e-2,
                                         dtMin = 1e-8, tStart = 0.0)


proc HEUN2_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + dt, y + dt * k1)
    let yNew = y + 0.5 * dt * (k1 + k2)
    return (yNew, yNew, dt, 0.0)

proc RALSTON2_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + 2/3 * dt, y + 2/3 * dt * k1)
    let yNew = y + dt * (0.25 * k1 + 0.75 * k2)
    return (yNew, yNew, dt, 0.0)

proc KUTTA3_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    let k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
    let yNew = y + dt * (1/6 * k1 + 2/3 * k2 + 1/6 * k3)
    return (yNew, yNew, dt, 0.0)

proc HEUN3_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + 1/3 * dt, y + 1/3 * dt * k1)
    let k3 = f(t + 2/3 * dt, y + 2/3 * dt * k2)
    let yNew = y + dt * (0.25 * k1 + 0.75 * k3)
    return (yNew, yNew, dt, 0.0)

proc RALSTON3_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + 1/2 * dt, y + 1/2 * dt * k1)
    let k3 = f(t + 3/4 * dt, y + 3/4 * dt * k2)
    let yNew = y + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)
    return (yNew, yNew, dt, 0.0)

proc SSPRK3_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + dt, y + dt * k1)
    let k3 = f(t + 0.5 * dt, y + 0.25 * dt * (k1 + k2))
    let yNew = y + dt * (1/6 * k1 + 1/6 * k2 + 2/3 * k3)
    return (yNew, yNew, dt, 0.0)


proc RALSTON4_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + 0.4 * dt, y + 0.4 * dt * k1)
    let k3 = f(t + 0.45573725 * dt, y + dt * (0.29697761 * k1 + 0.15875964 * k2))
    let k4 = f(t + dt, y + dt * (0.21810040 * k1 - 3.05096516 * k2 + 3.83286476 * k3))
    let yNew = y + dt * (0.17476028 * k1 - 0.55148066 * k2 + 1.20553560 * k3 + 0.17118478 * k4)
    return (yNew, yNew, dt, 0.0)

proc KUTTA4_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let k1 = f(t, y)
    let k2 = f(t + 1/3 * dt, y + 1/3 * dt * k1)
    let k3 = f(t + 2/3 * dt, y + dt * (-1/3 * k1 + k2))
    let k4 = f(t + dt, y + dt * (k1 - k2 + k3))
    let yNew = y + dt * (1/8 * k1 + 3/8 * k2 + 3/8 * k3 + 1/8 * k4)
    return (yNew, yNew, dt, 0.0)

proc RK4_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
                 options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using RK4. Only for internal use.
    var k1, k2, k3, k4: T
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5*dt, y + 0.5 * dt * k2)
    k4 = f(t +     dt, y +       dt * k3)
    let yNew = y + dt / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return (yNew, yNew, dt, 0.0)

proc RK21_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let tol = options.tol
    let dtMax = options.dtMax
    let dtMin = options.dtMin
    var k1, k2: T
    var yNew, yLow: T
    var error: float
    var limitCounter = 0
    var dt = dt
    commonAdaptiveMethodCode(yNew, error_y, order=2):
        k1 = f(t, y)
        k2 = f(t + dt, y + dt * k1)
        
        yNew = y + dt * 0.5 * (k1 + k2)
        yLow = y + dt * k1
        let error_y = yNew - yLow
    result = (yNew, yNew, dt, error)

proc BS32_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
    options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using Heun. Only for internal use.
    let tol = options.tol
    let dtMax = options.dtMax
    let dtMin = options.dtMin
    var k1, k2, k3, k4: T
    var yNew, yLow: T
    var error: float
    var limitCounter = 0
    var dt = dt
    commonAdaptiveMethodCode(yNew, error_y, order=3):
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.75 * dt, y + 0.75 * dt * k2)
        yNew = y + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)
        k4 = f(t + dt, yNew)
        
        yLow = y + dt * (7/24 * k1 + 1/4 * k2 + 1/3 * k3 + 1/8 * k4)
        let error_y = yNew - yLow
        # error = calcError(yNew, yLow)
    result = (yNew, k4, dt, error)


proc DOPRI54_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
                     options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using DOPRI54. Only for internal use.
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
    commonAdaptiveMethodCode(yNew, error_y, order=5):
        k1 = FSAL
        k2 = f(t + dt*c2, y + dt * (a21 * k1))
        k3 = f(t + dt*c3, y + dt * (a31 * k1 + a32 * k2))
        k4 = f(t + dt*c4, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + dt*c5, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + dt*c6, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + dt*c7, y + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        yNew = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        yLow = y + dt * (bHat1 * k1 + bHat2 * k2 + bHat3 * k3 + bHat4 * k4 + bHat5 * k5 + bHat6 * k6 + bHat7 * k7)
        let error_y = yNew - yLow
        # error = calcError(yNew, yLow)
    result = (yNew, k7, dt, error)

proc TSIT54_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
                     options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using TSIT54. Only for internal use.
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
    let tol = options.tol
    let dtMax = options.dtMax
    let dtMin = options.dtMin
    var k1, k2, k3, k4, k5, k6, k7: T
    var yNew, yLow: T
    var error: float
    var limitCounter = 0
    var dt = dt
    commonAdaptiveMethodCode(yNew, error_y, order=5):
        k1 = FSAL
        k2 = f(t + dt*c2, y + dt * (a21 * k1))
        k3 = f(t + dt*c3, y + dt * (a31 * k1 + a32 * k2))
        k4 = f(t + dt*c4, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + dt*c5, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + dt*c6, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + dt*c7, y + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        yNew = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        let error_y = dt * (bHat1 * k1 + bHat2 * k2 + bHat3 * k3 + bHat4 * k4 + bHat5 * k5 + bHat6 * k6 + bHat7 * k7)
        # error = calcError(y, yLow)
    result = (yNew, k7, dt, error)
    

proc VERN65_step[T](f: proc(t: float, y: T): T, t: float, y, FSAL: T, dt: float,
                     options: ODEoptions): (T, T, float, float) =
    ## Take a single timestep using DOPRI54. Only for internal use.
    const
        c2 = 0.06
        c3 = 0.09593333333333333
        c4 = 0.1439
        c5 = 0.4973
        c6 = 0.9725
        c7 = 0.9995
        c8 = 1.0
        c9 = 1.0
        a21 = 0.06
        a31 = 0.019239962962962962
        a32 = 0.07669337037037037
        a41 = 0.035975
        a42 = 0.0
        a43 = 0.107925
        a51 = 1.3186834152331484
        a52 = 0.0
        a53 = -5.042058063628562
        a54 = 4.220674648395414
        a61 = -41.87259166432751
        a62 = 0.0
        a63 = 159.43256216313748
        a64 = -122.11921356501004
        a65 = 5.531743066200053
        a71 = -54.430156935316504
        a72 = 0.0
        a73 = 207.06725136501848
        a74 = -158.61081378459
        a75 = 6.991816585950242
        a76 = -0.01859723106220323
        a81 = -54.66374178728198
        a82 = 0.0
        a83 = 207.95280625538936
        a84 = -159.2889574744995
        a85 = 7.018743740796944
        a86 = -0.018338785905045722
        a87 = -0.0005119484997882099
        a91 = 0.03438957868357036
        a92 = 0.0
        a93 = 0.0
        a94 = 0.25826245556335037
        a95 = 0.4209371189673537
        a96 = 4.405396469669310
        a97 = -176.48311902429865
        a98 = 172.36413340141507
        # Sixth order
        b1 = 0.03438957868357036
        b2 = 0.0
        b3 = 0.0
        b4 = 0.25826245556335034
        b5 = 0.42093711896735372
        b6 = 4.4053964696693102
        b7 = -176.48311902429866
        b8 = 172.36413340141507
        # Fifth order
        bHat1 = 0.04909967648382
        bHat2 = 0.0
        bHat3 = 0.0
        bHat4 = 0.22511122295165
        bHat5 = 0.46946822530296
        bHat6 = 0.80657922499889
        bHat7 = 0.0
        bHat8 = -0.60711948917780
        bHat9 = 0.05686113944048
    let tol = options.tol
    let dtMax = options.dtMax
    let dtMin = options.dtMin
    var k1, k2, k3, k4, k5, k6, k7, k8, k9: T
    var yNew, yLow: T
    var error: float
    var limitCounter = 0
    var dt = dt
    commonAdaptiveMethodCode(yNew, error_y, order=6):
        k1 = FSAL
        k2 = f(t + dt*c2, y + dt * (a21 * k1))
        k3 = f(t + dt*c3, y + dt * (a31 * k1 + a32 * k2))
        k4 = f(t + dt*c4, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + dt*c5, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + dt*c6, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + dt*c7, y + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))
        k8 = f(t + dt*c8, y + dt * (a81 * k1 + a82 * k2 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7))
        k9 = f(t + dt*c9, y + dt * (a91 * k1 + a92 * k2 + a93 * k3 + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8))

        yNew = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8)
        yLow = y + dt * (bHat1 * k1 + bHat2 * k2 + bHat3 * k3 + bHat4 * k4 + bHat5 * k5 + bHat6 * k6 + bHat7 * k7 + bHat8 * k8 + bHat9 * k9)
        let error_y = yNew - yLow
        # error = calcError(yNew, yLow)
    result = (yNew, k9, dt, error)


proc ODESolver[T](f: proc(t: float, y: T): T, y0: T, tspan: openArray[float],
                  options: ODEoptions = DEFAULT_ODEoptions,
                  integrator: proc(f: proc(t: float, y: T): T,
                                   t: float, y, FSAL: T,
                                   dt: float, options: ODEoptions): (T, T, float, float),
                  useFSAL = false, order: float, adaptive = false): (seq[float], seq[T]) =
    ## Handles the ODE solving. Only for internal use.
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
    return (tNegative.reversed().concat(tZero).concat(tPositive),
            yNegative.reversed().concat(yZero).concat(yPositive))


proc solveODE*[T](f: proc(t: float, y: T): T, y0: T, tspan: openArray[float],
                  options: ODEoptions = DEFAULT_ODEoptions,
                  integrator="dopri54"): (seq[float], seq[T]) =
    ## Solve an ODE initial value problem.
    ##
    ## Input:
    ##   - f: the ODE function y' = f(t, y).
    ##   - y0: Initial value.
    ##   - tspan: Seq of t values that y will be returned at.
    ##   - options: ODEoptions object with ODE parameters.
    ##   - integrator: String with the integrator to use. Choices: "dopri54", "rk4".
    ##
    ## Returns:
    ##   - A tuple containing a seq of t-values and a seq of y-values (t, y).
    case integrator.toLower():
        of "dopri54":
            return ODESolver(f, y0, tspan.sorted(), options, DOPRI54_step,
                             useFSAL = true, order = 5.0, adaptive = true)
        of "rk21":
            return ODESolver(f, y0, tspan.sorted(), options, RK21_step ,
                                useFSAL = false, order = 2.0, adaptive = true)
        of "bs32":
            return ODESolver(f, y0, tspan.sorted(), options, BS32_step,
                                useFSAL = true, order = 3.0, adaptive = true)
        of "rk4":
            return ODESolver(f, y0, tspan.sorted(), options, RK4_step,
                             useFSAL = false, order = 4.0, adaptive = false)
        of "heun2":
            return ODESolver(f, y0, tspan.sorted(), options, HEUN2_step,
                             useFSAL = false, order = 2.0, adaptive = false)
        of "ralston2":
            return ODESolver(f, y0, tspan.sorted(), options, RALSTON2_step,
                             useFSAL = false, order = 2.0, adaptive = false)
        of "kutta3":
            return ODESolver(f, y0, tspan.sorted(), options, KUTTA3_step,
                                useFSAL = false, order = 3.0, adaptive = false)
        of "heun3":
            return ODESolver(f, y0, tspan.sorted(), options, HEUN3_step,
                                useFSAL = false, order = 3.0, adaptive = false)
        of "ralston3":
            return ODESolver(f, y0, tspan.sorted(), options, RALSTON3_step,
                                useFSAL = false, order = 3.0, adaptive = false)
        of "ssprk3":
            return ODESolver(f, y0, tspan.sorted(), options, SSPRK3_step,
                                useFSAL = false, order = 3.0, adaptive = false)
        of "ralston4":
            return ODESolver(f, y0, tspan.sorted(), options, RALSTON4_step,
                                useFSAL = false, order = 4.0, adaptive = false)
        of "kutta4":
            return ODESolver(f, y0, tspan.sorted(), options, KUTTA4_step,
                                useFSAL = false, order = 4.0, adaptive = false)
        of "vern65":
            return ODESolver(f, y0, tspan.sorted(), options, VERN65_step,
                                useFSAL = true, order = 6.0, adaptive = true)
        of "tsit54":
            return ODESolver(f, y0, tspan.sorted(), options, TSIT54_step,
                             useFSAL = true, order = 5.0, adaptive = true)
        else:
            raise newException(ValueError, &"{integrator} is not a valid integrator")
