import std / macros
proc checkArgNumContext(fn: NimNode) =
  ## Checks the first argument of the given proc is indeed a `NumContextProc` argument.
  let params = fn.params
  # FormalParams                 <- `.params`
  #   Ident "T"
  #   IdentDefs                  <- `params[1]`
  #     Sym "f"
  #     BracketExpr              <- `params[1][1]`
  #       Sym "NumContextProc"   <- `params[1][1][0]`
  #       Ident "T"
  #       Sym "float"
  #     Empty
  expectKind params, nnkFormalParams
  expectKind params[1], nnkIdentDefs
  expectKind params[1][1], nnkBracketExpr
  expectKind params[1][1][0], {nnkSym, nnkIdent}
  if params[1][1][0].strVal != "NumContextProc":
    error("The function annotated with `{.genInterp.}` does not take a `NumContextProc` as the firs argument.")

proc replaceNumCtxArg(fn: NimNode): NimNode =
  ## Checks the first argument of the given proc is indeed a `NumContextProc` argument.
  ## MUST run `checkArgNumContext` on `fn` first.
  ##
  ## It returns the identifier of the first argument.
  var params = fn.params # see `checkArgNNumContext`
  expectKind params[1][0], {nnkSym, nnkIdent}
  result = ident(params[1][0].strVal)
  params[1] = nnkIdentDefs.newTree(
    result,
    nnkBracketExpr.newTree(
      ident"InterpolatorType",
      ident"T"
    ),
    newEmptyNode()
  )
  fn.params = params

proc untype(n: NimNode): NimNode =
  case n.kind
  of nnkSym: result = ident(n.strVal)
  of nnkIdent: result = n
  else:
    error("Cannot untype the argument: " & $n.treerepr)

proc genOriginalCall(fn: NimNode, ncp: NimNode): NimNode =
  ## Generates a call to the original procedure `fn` with `ncp`
  ## as the first argument
  let fnName = fn.name
  let params = fn.params
  # extract all arguments we need to pass from `params`
  var p = newSeq[NimNode]()
  p.add ncp
  for i in 2 ..< params.len: # first param is return type, second is parameter we replace
    expectKind params[i], nnkIdentDefs
    if params[i].len in 0 .. 2:
      error("Invalid parameter: " & $params[i].treerepr)
    else: # one or more arg of this type
      # IdentDefs          <- Example with 2 arguments of the same type
      #   Ident "xStart"   <- index `0`
      #   Ident "xEnd"     <- index `len - 3 = 4 - 3 = 1`
      #   Ident "float"
      #   Empty
      for j in 0 .. params[i].len - 3:
        p.add untype(params[i][j])
  # generate the call
  result = nnkCall.newTree(fnName)
  for el in p:
    result.add el

macro genInterp*(fn: untyped): untyped =
  ## Takes a `proc` with a `NumContextProc` parameter as the first argument
  ## and returns two procedures:
  ## 1. The original proc
  ## 2. An overload, which converts an `InterpolatorType[T]` argument to a
  ##    `NumContextProc[T, float]` using the conversion proc.
  doAssert fn.kind in {nnkProcDef, nnkFuncDef}
  result = newStmtList(fn)
  # 1. check arg
  checkArgNumContext(fn)
  # 2. generate overload
  var new = fn.copyNimTree()
  # 2a. replace first argument by `InterpolatorType[T]`
  let arg = new.replaceNumCtxArg()
  # 2b. add body with NumContextProc
  let ncpIdent = ident"ncp"
  new.body = quote do:
    mixin eval # defined in `interpolate`, but macro used in `integrate`
    let `ncpIdent` = proc(x: float, ctx: NumContext[T, float]): T = eval(`arg`, x)
  # 2c. add call to original proc
  new.body.add genOriginalCall(fn, ncpIdent)
  # 3. finalize
  result.add new
