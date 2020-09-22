import tables

type
  NumContext*[T] = ref object
    fValues*: Table[string, float]
    tValues*: Table[string, T]

  ODEProc*[T] = proc(t: float, y: T, ctx: NumContext[T]): T

proc newNumContext*[T](fValues: Table[string, float] = initTable[string, float](), tValues: Table[string, T] = initTable[string, T]()): NumContext[T] =
  NumContext[T](fValues: fValues, tValues: tValues)

proc `[]`*[T](ctx: NumContext[T], key: string): T =
  ctx.tValues[key]

proc `[]`*[T](ctx: NumContext[T], key: enum): T =
  ctx.tValues[$key]

proc `[]=`*[T](ctx: NumContext[T], key: string, val: T) =
  ctx.tValues[key] = val

proc `[]=`*[T](ctx: NumContext[T], key: enum, val: T) =
  ctx.tValues[$key] = val

proc getF*[T](ctx: NumContext[T], key: string): float =
  ctx.fValues[key]

proc setF*[T](ctx: NumContext[T], key: string, val: float) =
  ctx.fValues[key] = val

proc getF*[T](ctx: NumContext[T], key: enum): float =
  ctx.fValues[$key]

proc setF*[T](ctx: NumContext[T], key: enum, val: float) =
  ctx.fValues[$key] = val

when isMainModule:
  var a = newNumContext[int]()
  a["hej"] = 1
  a["då"] = 2
  echo a[]

  type
    e = enum
      val1
      val2
      val3

  var b = newNumContext[float]()
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
