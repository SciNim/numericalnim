import std / [math, algorithm, tables, sequtils]
import arraymancer
import ./utils


type
  RbfFunc* = proc (r: Tensor[float], epsilon: float): Tensor[float]
  RbfType*[T] = object
    points*: Tensor[float] # (n_points, n_dim)
    values*: Tensor[T] # (n_points, n_values)
    coeffs*: Tensor[float] # (n_points, n_values)
    epsilon*: float
    f*: RbfFunc

  RbfGrid*[T] = object
    indices*: seq[seq[int]]
    values*: Tensor[T]
    points*: Tensor[float]
    gridSize*, gridDim*: int
    gridDelta*: float

  RbfPUType*[T] = object
    limits*: tuple[upper: Tensor[float], lower: Tensor[float]]
    grid*: RbfGrid[RbfType[T]]
    nValues*: int

template km(point: Tensor[float], index: int, delta: float): int =
  int(ceil(point[0, index] / delta))

iterator neighbours*[T](grid: RbfGrid[T], k: int): int =
  discard

iterator neighboursIncludingCenter*[T](grid: RbfGrid[T], k: int): int =
  yield k
  for x in grid.neighbours(k):
    yield x

proc findIndex*[T](grid: RbfGrid[T], point: Tensor[float]): int =
  result = km(point, grid.gridDim - 1, grid.gridDelta) - 1
  for i in 0 ..< grid.gridDim - 1:
    result += (km(point, i, grid.gridDelta) - 1) * grid.gridSize ^ (grid.gridDim - i - 1)

proc constructMeshedPatches*[T](grid: RbfGrid[T]): Tensor[float] =
  meshgrid(@[arraymancer.linspace(0 + grid.gridDelta / 2, 1 - grid.gridDelta / 2, grid.gridSize)].cycle(grid.gridDim))

template dist2(p1, p2: Tensor[float]): float =
  var result = 0.0
  for i in 0 ..< p1.shape[1]:
    let diff = p1[0, i] - p2[0, i]
    result += diff * diff
  result

proc findAllWithin*[T](grid: RbfGrid[T], x: Tensor[float], rho: float): seq[int] =
  assert x.shape.len == 2 and x.shape[0] == 1
  let index = grid.findIndex(x)
  for k in grid.neighboursIncludingCenter(index):
    for i in grid.indices[k]:
      if dist2(x, grid.points[i, _]) <= rho*rho:
        result.add i

proc newRbfGrid*[T](points: Tensor[float], values: Tensor[T], gridSize: int = 0): RbfGrid[T] =
  let nPoints = points.shape[0]
  let nDims = points.shape[1]
  let gridSize =
    if gridSize > 0:
      gridSize
    else:
      max(int(round(pow(nPoints.float, 1 / nDims) / 2)), 1)
  let delta = 1 / gridSize
  result = RbfGrid[T](gridSize: gridSize, gridDim: nDims, gridDelta: delta, indices: newSeq[seq[int]](gridSize ^ nDims))
  for row in 0 ..< nPoints:
    let index = result.findIndex(points[row, _])
    result.indices[index].add row
  result.values = values
  result.points = points

# Idea: blocked distance matrix for better cache friendliness
proc distanceMatrix(p1, p2: Tensor[float]): Tensor[float] =
  ## Returns distance matrix of shape (n_points, n_points)
  let n_points1 = p1.shape[0]
  let n_points2 = p2.shape[0]
  let n_dims = p1.shape[1]
  result = newTensor[float](n_points2, n_points1)
  for i in 0 ..< n_points2:
    for j in 0 ..< n_points1:
      var r2 = 0.0
      for k in 0 ..< n_dims:
        let diff = p2[i,k] - p1[j,k]
        r2 += diff * diff
      result[i, j] = sqrt(r2)

template compactRbfFuncScalar*(r: float, epsilon: float): float =
  (1 - r/epsilon) ^ 4 * (4*r/epsilon + 1) * float(r < epsilon)

proc compactRbfFunc*(r: Tensor[float], epsilon: float): Tensor[float] =
  result = map_inline(r):
    (1 - x/epsilon) ^ 4 * (4*x/epsilon + 1) * float(x < epsilon)

proc newRbf*[T](points: Tensor[float], values: Tensor[T], rbfFunc: RbfFunc = compactRbfFunc, epsilon: float = 1): RbfType[T] =
  assert points.shape[0] == values.shape[0]
  let dist = distanceMatrix(points, points)
  let A = rbfFunc(dist, epsilon)
  let coeffs = solve(A, values)
  result = RbfType[T](points: points, values: values, coeffs: coeffs, epsilon: epsilon, f: rbfFunc)

proc eval*[T](rbf: RbfType[T], x: Tensor[float]): Tensor[T] =
  let dist = distanceMatrix(rbf.points, x)
  echo dist
  let A = rbf.f(dist, rbf.epsilon)
  echo A
  echo "---------------"
  result = A * rbf.coeffs

proc scalePoint*(x: Tensor[float], limits: tuple[upper: Tensor[float], lower: Tensor[float]]): Tensor[float] =
  let lower = limits.lower -. 0.01
  let upper = limits.upper +. 0.01
  (x -. lower) /. (upper - lower)

proc newRbfPu*[T](points: Tensor[float], values: Tensor[T], gridSize: int = 0, rbfFunc: RbfFunc = compactRbfFunc, epsilon: float = 1): RbfPUType[T] =
  assert points.shape[0] == values.shape[0]
  assert points.shape.len == 2 and values.shape.len == 2
  let upperLimit = max(points, 0)
  let lowerLimit = min(points, 0)
  let limits = (upper: upperLimit, lower: lowerLimit) # move this buff to scalePoint
  let scaledPoints = points.scalePoint(limits)
  let dataGrid = newRbfGrid(scaledPoints, values, gridSize)
  let patchPoints = dataGrid.constructMeshedPatches()
  let nPatches = patchPoints.shape[0]
  var patchRbfs: seq[RbfType[T]] #= newTensor[RbfType[T]](nPatches, 1)
  var patchIndices: seq[int]
  for i in 0 ..< nPatches:
    let indices = dataGrid.findAllWithin(patchPoints[i, _], dataGrid.gridDelta)
    if indices.len > 0:
      patchRbfs.add newRbf(dataGrid.points[indices,_], values[indices, _], epsilon=dataGrid.gridDelta)
      patchIndices.add i

  let patchGrid = newRbfGrid(patchPoints[patchIndices, _], patchRbfs.toTensor.unsqueeze(1), gridSize)
  result = RbfPUType[T](limits: limits, grid: patchGrid, nValues: values.shape[1])

proc eval*[T](rbf: RbfPUType[T], x: Tensor[float]): Tensor[T] =
  assert x.shape.len == 2
  assert (not ((x <=. rbf.limits.upper) and (x >=. rbf.limits.lower))).astype(int).sum() == 0, "Some of your points are outside the allowed limits" 
  let nPoints = x.shape[0] 
  let x = x.scalePoint(rbf.limits)
  result = newTensor[T](nPoints, rbf.nValues)
  for row in 0 ..< nPoints:
    let p = x[row, _]
    let indices = rbf.grid.findAllWithin(p, rbf.grid.gridDelta)
    if indices.len > 0:
      var c = 0.0
      for i in indices:
        let center = rbf.grid.points[i, _]
        let r = sqrt(dist2(p, center))
        let ci = compactRbfFuncScalar(r, rbf.grid.gridDelta)
        c += ci
        let val = rbf.grid.values[i, 0].eval(p)
        result[row, _] = result[row, _] + ci * val
      result[row, _] = result[row, _] / c
    else:
      result[row, _] = T(Nan) # allow to pass default value to newRbfPU?

when isMainModule:
  let x1 = @[@[0.01, 0.01, 0.01], @[1.0, 1.0, 0.0], @[1.0, 2.0, 4.0]].toTensor
  let x2 = @[@[0.0, 0.0, 1.0], @[1.0, 1.0, 2.0], @[1.0, 2.0, 3.0]].toTensor
  #echo distanceMatrix(x1, x1)
  let values = @[0.1, 1.0, 2.0].toTensor.unsqueeze(1)
  #echo newRbf(x1, values, epsilon=10)

  import benchy

  #let pos = randomTensor(5000, 3, 1.0)
  #let vals = randomTensor(5000, 1, 1.0)
  # timeIt "Rbf":
  #  keep newRbf(pos, vals)
  let rbfPu = newRbfPu(x1, values, 3)
  echo rbfPu.grid.values[1, 0]
  #echo rbfPu.eval(x1[[2, 1, 0], _])

  echo rbfPu.eval(sqrt x1)
  echo "----------------"
  #let xGrid = [[0.1, 0.1], [0.2, 0.3], [0.9, 0.9], [0.4, 0.4]].toTensor
  #let valuesGrid = @[0, 1, 9, 5].toTensor.reshape(4, 1)
  #echo newRbfGrid(xGrid, valuesGrid, 3)
