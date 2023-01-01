import std / [math, algorithm, tables, sequtils, strutils]
import arraymancer
import ./utils


type
  RbfFunc* = proc (r: Tensor[float], epsilon: float): Tensor[float]
  RbfBaseType*[T] = object
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

  RbfType*[T] = object
    limits*: tuple[upper: Tensor[float], lower: Tensor[float]]
    grid*: RbfGrid[RbfBaseType[T]]
    nValues*: int

template km(point: Tensor[float], index: int, delta: float): int =
  int(ceil(point[0, index] / delta))

iterator neighbours*[T](grid: RbfGrid[T], k: int, searchLevels: int = 1): int =
  # TODO: Create product iterator that doesn't need to allocate 3^gridDim seqs
  let directions = @[toSeq(-searchLevels .. searchLevels)].cycle(grid.gridDim)
  for dir in product(directions):
    block loopBody:
      var kNeigh = k
      for i, x in dir:
        let step = grid.gridSize ^ (grid.gridDim - i - 1)
        for level in 1 .. searchLevels:
          if (k div step) mod grid.gridSize == level - 1 and x <= -level:
            break loopBody
          elif (k div step) mod grid.gridSize == grid.gridSize - level and x >= level:
            break loopBody
        kNeigh += x * step
      if kNeigh >= 0 and kNeigh < grid.gridSize ^ grid.gridDim:
        yield kNeigh


iterator neighboursExcludingCenter*[T](grid: RbfGrid[T], k: int): int =
  for x in grid.neighbours(k):
    if x != k:
      yield x

proc findIndex*[T](grid: RbfGrid[T], point: Tensor[float]): int =
  result = km(point, grid.gridDim - 1, grid.gridDelta) - 1
  for i in 0 ..< grid.gridDim - 1:
    result += (km(point, i, grid.gridDelta) - 1) * grid.gridSize ^ (grid.gridDim - i - 1)

proc constructMeshedPatches*[T](grid: RbfGrid[T]): Tensor[float] =
  if grid.gridSize == 1:
    @[@[0.5].cycle(grid.gridDim)].toTensor
  else:
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
  let searchLevels = (rho / grid.gridDelta).ceil.int
  for k in grid.neighbours(index, searchLevels):
    for i in grid.indices[k]:
      if dist2(x, grid.points[i, _]) <= rho*rho:
        result.add i

proc findAllBetween*[T](grid: RbfGrid[T], x: Tensor[float], rho1, rho2: float): seq[int] =
  assert x.shape.len == 2 and x.shape[0] == 1
  assert rho2 > rho1
  let index = grid.findIndex(x)
  let searchLevels = (rho2 / grid.gridDelta).ceil.int
  for k in grid.neighbours(index, searchLevels):
    for i in grid.indices[k]:
      let d = dist2(x, grid.points[i, _])
      if rho1*rho1 <= d and d <= rho2*rho2:
        result.add i

proc calcGridSize(nPoints, nDims: int, gridSize: int): int =
  if gridSize > 0:
    gridSize
  else:
    max(int(round(pow(nPoints.float, 1 / nDims) / 2)), 1)

proc newRbfGrid*[T](points: Tensor[float], values: Tensor[T], gridSize: int = 0): RbfGrid[T] =
  let nPoints = points.shape[0]
  let nDims = points.shape[1]
  let gridSize = calcGridSize(nPoints, nDims, gridSize)
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
    let xeps = x / epsilon
    let temp = (1 - xeps)
    let temp2 = temp * temp
    temp2*temp2 * (4*xeps + 1) * float(xeps < 1)

proc newRbfBase*[T](points: Tensor[float], values: Tensor[T], rbfFunc: RbfFunc = compactRbfFunc, epsilon: float = 1): RbfBaseType[T] =
  assert points.shape[0] == values.shape[0]
  let dist = distanceMatrix(points, points)
  let A = rbfFunc(dist, epsilon)
  let coeffs = solve(A, values)
  result = RbfBaseType[T](points: points, values: values, coeffs: coeffs, epsilon: epsilon, f: rbfFunc)

proc eval*[T](rbf: RbfBaseType[T], x: Tensor[float]): Tensor[T] =
  let dist = distanceMatrix(rbf.points, x)
  let A = rbf.f(dist, rbf.epsilon)
  result = A * rbf.coeffs

proc scalePoint*(x: Tensor[float], limits: tuple[upper: Tensor[float], lower: Tensor[float]]): Tensor[float] =
  let lower = limits.lower -. 0.01
  let upper = limits.upper +. 0.01
  (x -. lower) /. (upper - lower)

proc newRbf*[T](points: Tensor[float], values: Tensor[T], gridSize: int = 0, rbfFunc: RbfFunc = compactRbfFunc, epsilon: float = 1): RbfType[T] =
  ## Construct a Radial basis function interpolator using Partition of Unity.
  ## points: The positions of the data points. Shape: (nPoints, nDims)
  ## values: The values at the points. Can be multivalued. Shape: (nPoints, nValues)
  ## gridSize: The number of cells along each dimension. Setting it to the default 0 will automatically choose a value based on the number of points.
  ## rbfFunc: The RBF function that accepts shape parameter. Default is a C^2 compactly supported function.
  ## epsilon: shape parameter. Default 1.
  assert points.shape[0] == values.shape[0]
  assert points.shape.len == 2 and values.shape.len == 2

  let upperLimit = max(points, 0)
  let lowerLimit = min(points, 0)
  let limits = (upper: upperLimit, lower: lowerLimit)
  let scaledPoints = points.scalePoint(limits)

  let nPoints = points.shape[0]
  let nDims = points.shape[1]
  let gridSize = calcGridSize(nPoints, nDims, gridSize)

  let dataGrid = newRbfGrid(scaledPoints, values, gridSize)
  let patchPoints = dataGrid.constructMeshedPatches()
  let nPatches = patchPoints.shape[0]

  var patchRbfs: seq[RbfBaseType[T]] #= newTensor[RbfBaseType[T]](nPatches, 1)
  var patchIndices: seq[int]
  for i in 0 ..< nPatches:
    let indices = dataGrid.findAllWithin(patchPoints[i, _], dataGrid.gridDelta)
    if indices.len > 0:
      patchRbfs.add newRbfBase(dataGrid.points[indices,_], values[indices, _], epsilon=epsilon)
      patchIndices.add i

  let patchGrid = newRbfGrid(patchPoints[patchIndices, _], patchRbfs.toTensor.unsqueeze(1), gridSize)
  result = RbfType[T](limits: limits, grid: patchGrid, nValues: values.shape[1])

proc eval*[T](rbf: RbfType[T], x: Tensor[float]): Tensor[T] =
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
      result[row, _] = T(Nan) # allow to pass default value to newRbf?

proc evalAlt*[T](rbf: RbfType[T], x: Tensor[float]): Tensor[T] =
  assert x.shape.len == 2
  assert (not ((x <=. rbf.limits.upper) and (x >=. rbf.limits.lower))).astype(int).sum() == 0, "Some of your points are outside the allowed limits"

  let nPoints = x.shape[0] 
  let x = x.scalePoint(rbf.limits)
  result = newTensor[T](nPoints, rbf.nValues)
  var c = newTensor[float](nPoints, 1)
  var isSet = newTensor[bool](nPoints, 1)
  let nPatches = rbf.grid.points.shape[0]
  let pointGrid = newRbfGrid(x, x, rbf.grid.gridSize)
  for row in 0 ..< nPatches:
    let center = rbf.grid.points[row, _]
    let indices = pointGrid.findAllWithin(center, rbf.grid.gridDelta)
    if indices.len > 0:
      let vals = rbf.grid.values[row, 0].eval(x[indices, _])
      for i, index in indices:
        let r = sqrt(dist2(center, x[index, _]))
        let ci = compactRbfFuncScalar(r, rbf.grid.gridDelta)
        result[index, _] = result[index, _] + ci * vals[i, _]
        c[index] += ci
        isSet[index, 0] = true

  result /.= c
  result[not isSet, _] = T(NaN)
