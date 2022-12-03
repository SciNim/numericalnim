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

when isMainModule:
  let x1 = @[@[0.0, 0.0, 0.0], @[1.0, 1.0, 0.0], @[1.0, 2.0, 0.0]].toTensor
  let x2 = @[@[0.0, 0.0, 1.0], @[1.0, 1.0, 2.0], @[1.0, 2.0, 3.0]].toTensor
  echo distanceMatrix(x1, x1)
  let values = @[0.0, 1.0, 2.0].toTensor
  echo newRbf(x1, values, epsilon=10)

  import benchy

  let pos = randomTensor(5000, 3, 1.0)
  let vals = randomTensor(5000, 1, 1.0)
  timeIt "Rbf":
    keep newRbf(pos, vals)

