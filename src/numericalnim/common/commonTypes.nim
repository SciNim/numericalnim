import tables

type
  NumContext*[T; U] = ref object
    fValues*: Table[string, U]
    tValues*: Table[string, T]

  ODEProc*[T; U] = proc(t: U, y: T, ctx: NumContext[T, U]): T

  NumContextProc*[T; U] = proc(x: U, ctx: NumContext[T, U]): T

  InterpolatorProc*[T] = proc(x: float): T

proc newNumContext*[T; U](fValues: Table[string, U] = initTable[string, U](), tValues: Table[string, T] = initTable[string, T]()): NumContext[T, U] =
  NumContext[T, U](fValues: fValues, tValues: tValues)

proc newNumContext*[T](fValues: Table[string, float] = initTable[string, float](),
                       tValues: Table[string, T] = initTable[string, T]()): NumContext[T, float] =
  NumContext[T, float](fValues: fValues, tValues: tValues)

proc `[]`*[T; U](ctx: NumContext[T, U], key: string): T =
  ctx.tValues[key]

proc `[]`*[T; U](ctx: NumContext[T, U], key: enum): T =
  ctx.tValues[$key]

proc `[]=`*[T; U](ctx: NumContext[T, U], key: string, val: T) =
  ctx.tValues[key] = val

proc `[]=`*[T; U](ctx: NumContext[T, U], key: enum, val: T) =
  ctx.tValues[$key] = val

proc getF*[T; U](ctx: NumContext[T, U], key: string): U =
  ctx.fValues[key]

proc setF*[T; U](ctx: NumContext[T, U], key: string, val: U) =
  ctx.fValues[key] = val

proc getF*[T; U](ctx: NumContext[T, U], key: enum): U =
  ctx.fValues[$key]

proc setF*[T; U](ctx: NumContext[T, U], key: enum, val: U) =
  ctx.fValues[$key] = val

when isMainModule:
  var a = newNumContext[int, float]()
  a["hej"] = 1
  a["då"] = 2
  echo a[]

  type
    e = enum
      val1
      val2
      val3

  var b = newNumContext[float, float]()
  b[val1] = 1.0
  b[val2] = 2.0
  b[val3] = 3.0
  echo b[val2]



#[
proc solveODE*[T](f: ODEProc[T], y0: T, tspan: openArray[float], ctx: var NumContext): (seq[float], seq[T]) =
  # code... bla bla
    dy = f(t, y, ctx)
  # more code bla bla

proc f(t: float, y: Tensor[float], ctx: var NumContext[Tensor[float]]): Tensor[float] =
  let A = ctx.tValues[0]
  result = A * y

when isMainModule:
  let A = @[@[1.0, 2.0], @[3.0, 4.0]].toTensor
  var ctx = newNumContext(tValues = @[A])
  let (ys, ts) = solveODE(f, ...., ctx)
  plot(ts, ys) # or do something with them
]#
