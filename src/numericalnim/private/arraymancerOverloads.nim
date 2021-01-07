import arraymancer

template matrixMultImpl[T](t1, t2, zero): untyped =
  assert t1.shape[1] == t2.shape[0], "Dimensions must match for matrix multiplication"
  result = newTensor[T](t1.shape[0], t2.shape[1])
  let equalSide = t1.shape[1]
  for i in 0 ..< result.shape[0]:
    for j in 0 ..< result.shape[1]:
      var temp: T = zero
      for k in 0 ..< equalSide:
        temp = temp + t1[i, k] * t2[k, j]
      result[i, j] = temp

proc `*`*[T: not SomeNumber](t1: Tensor[float], t2: Tensor[T]): Tensor[T] =
  assert t1.rank == 2 and t2.rank == 2, "Only matricies are supported"
  let zero: T = t2[0, 0] - t2[0, 0]
  matrixMultImpl[T](t1, t2, zero)


proc `*`*[T: not SomeNumber](t1: Tensor[T], t2: Tensor[float]): Tensor[T] =
  assert t1.rank == 2 and t2.rank == 2, "Only matricies are supported"
  let zero: T = t1[0, 0] - t1[0, 0]
  matrixMultImpl[T](t1, t2, zero)

proc `*`*[T: not SomeNumber](t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
  assert t1.rank == 2 and t2.rank == 2, "Only matricies are supported"
  let zero: T = t2[0, 0] - t2[0, 0]
  matrixMultImpl[T](t1, t2, zero)

proc `+`*[T: not SomeNumber](t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
  assert t1.shape == t2.shape
  result = newTensor[T](t1.shape)
  apply3_inline(result, t1, t2):
    y + z

proc dot*[T: not SomeNumber](t1: Tensor[float], t2: Tensor[T]): T =
  let t2 = t2.squeeze
  let t1 = t1.squeeze
  assert t2.rank == 1 and t1.rank == 1, "Only 1D Tensors are supported"
  assert t1.shape[0] == t2.shape[0], "Tensors must have same length"
  result = t2[0] - t2[0] # zero
  for x, y in zip(t1, t2):
    result += x * y
