import std / [math, algorithm, tables]
import arraymancer


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
    # Don't hard-code in the patches, this is just for lookup to construct the patches
    # 
    grid*: seq[RbfType[T]]

template km(point: Tensor[float], index: int, delta: float): int =
  int(ceil(point[0, index] / delta))

iterator neighbours*[T](grid: RbfGrid[T], k: int): int =
  discard

proc findIndex*[T](grid: RbfGrid[T], point: Tensor[float]): int =
  result = km(point, grid.gridDim - 1, grid.gridDelta) - 1
  for i in 0 ..< grid.gridDim - 1:
    result += (km(point, i, grid.gridDelta) - 1) * grid.gridSize ^ (grid.gridDim - i - 1)

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
  let A = rbf.f(dist, rbf.epsilon)
  result = A * rbf.coeffs

proc scalePoint*(x: Tensor[float], limits: tuple[upper: Tensor[float], lower: Tensor[float]]): Tensor[float] =
  (x -. limits.lower) /. (limits.upper - limits.lower)

proc gridIndex*(limits: tuple[upper: Tensor[float], lower: Tensor[float]], gridSide: int): int =
  discard

proc newRbfPu*[T](points: Tensor[float], values: Tensor[T], gridSide: int, rbfFunc: RbfFunc = compactRbfFunc, epsilon: float = 1): RbfType[T] =
  let upperLimit = max(points, 0)
  let lowerLimit = min(points, 0)
  let limits = (upper: upperLimit, lower: lowerLimit)
  let scaledPoints = points.scalePoint(limits)
  echo scaledPoints
  echo limits

when isMainModule:
  let x1 = @[@[0.0, 0.0, 0.0], @[1.0, 1.0, 0.0], @[1.0, 2.0, 4.0]].toTensor
  let x2 = @[@[0.0, 0.0, 1.0], @[1.0, 1.0, 2.0], @[1.0, 2.0, 3.0]].toTensor
  echo distanceMatrix(x1, x1)
  let values = @[0.0, 1.0, 2.0].toTensor
  echo newRbf(x1, values, epsilon=10)

  import benchy

  let pos = randomTensor(5000, 3, 1.0)
  let vals = randomTensor(5000, 1, 1.0)
  # timeIt "Rbf":
  #  keep newRbf(pos, vals)
  let rbfPu = newRbfPu(x1, values, 3)

  echo "----------------"
  let xGrid = [[0.1, 0.1], [0.2, 0.3], [0.9, 0.9], [0.4, 0.4]].toTensor
  let valuesGrid = @[0, 1, 9, 5].toTensor.reshape(4, 1)
  echo newRbfGrid(xGrid, valuesGrid, 3)
