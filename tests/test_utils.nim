import unittest, math, sequtils, algorithm, random
import arraymancer
import numericalnim

test "isClose float":
  let a = 1.0
  let b = a + 1e-4
  check isClose(a, b, tol=1e-3) == true

test "isClose float failure":
  let a = 1.0
  let b = a + 1e-2
  check isClose(a, b, tol=1e-3) == false

test "linspace 0.0 to 10.0":
  let t = linspace(0.0, 10.0, 11)
  let ans = @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  check t == ans

test "linspace 10.0 to 0.0":
  let t = linspace(10.0, 0.0, 11)
  let ans = @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].reversed()
  check t == ans

test "arange 0.0 to 10.0, dx = 1.0":
  let t = arange(0.0, 10.0, 1.0)
  let ans = @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  check t == ans

test "arange 10.0 to 0.0, dx = 1.0":
  let t = arange(10.0, 0.0, 1.0)
  let ans = @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].reversed()
  check t == ans

test "Delete multiple indices":
  var original = @[0, 1, 2, 3, 4, 5, 6]
  let idx = @[0, 5, 3, 4]
  original.delete(idx)
  check original == @[1, 2, 6]

test "getIndexTable":
  let s = @[0,1,2,3,4,0,2,4,0]
  let idxTable = s.getIndexTable
  check idxTable == {0: @[0, 5, 8], 1: @[1], 2: @[2, 6], 3: @[3], 4: @[4, 7]}.toTable

test "findDuplicates":
  let s = @[0,1,2,3,4,0,2,4,0]
  let indices = findDuplicates(s)
  for idx in @[@[0, 5, 8], @[2, 6], @[4, 7]]:
    check idx in indices

test "removeDuplicates":
  var x = @[0,1,2,0,1,2]
  var ys = @[@[0.0,1,2,0,1,2], @[10.0,11,12,10,11,12]]
  let (xdeDup, ydeDup) = removeDuplicates(x, ys)
  check xdeDup == @[0,1,2]
  check ydeDup == @[@[0.0,1,2], @[10.0,11,12]]

test "removeDuplicates impure duplicates":
  var x = @[0, 1, 2, 0, 1, 2]
  var ys = @[@[0.0, 1, 2, 0, 1, 3]] # 2 != 3
  expect ValueError:
    let (xdeDup, ydeDup) = removeDuplicates(x, ys)

test "sortDataset":
  let x = @[0,1,2,3,4]
  let y1 = @[0.0,1,2,3,4]
  let y2 = @[10.0,11,12,13,14]
  var (xSorted, ysSorted) = sortDataset(x, @[y1, y2])
  check xSorted == x
  check ysSorted[0] == y1
  check ysSorted[1] == y2

  let x2 = x.reversed
  (xSorted, ysSorted) = sortDataset(x2, @[y1, y2])
  check xSorted == x
  check ysSorted[0] == y1.reversed
  check ysSorted[1] == y2.reversed

test "meshgridFlat":
  let x = [0, 1, 2].toTensor
  let y = [3, 4, 5].toTensor
  let (gridX, gridY) = meshgridFlat(x, y)
  check gridX == [0, 1, 2, 0, 1, 2, 0, 1, 2].toTensor
  check gridY == [3, 3, 3, 4, 4, 4, 5, 5, 5].toTensor

test "meshgrid":
  let x = [0, 1].toTensor
  let y = [2, 3].toTensor
  let z = [4, 5].toTensor
  let grid = numericalnim.meshgrid(x, y, z)
  check grid == [[0, 2, 4], [1, 2, 4], [0, 3, 4], [1, 3, 4], [0, 2, 5], [1, 2, 5], [0, 3, 5], [1, 3, 5]].toTensor

test "chi2 Tensor":
  randomize(1337)
  let N = 1000
  let sigma = 1.23
  let yMeasure = newSeqWith(N, gauss(0.0, sigma)).toTensor
  let yCorrect = zeros[float](N)
  let yError = ones[float](N) * sigma
  let chi = chi2(yMeasure, yCorrect, yError)
  # Check that the mean χ² is around 1 
  check chi / N.float in 0.90 .. 1.1

test "chi2 Seq":
  randomize(1337)
  let N = 1000
  let sigma = 1.23
  let yMeasure = newSeqWith(N, gauss(0.0, sigma))
  let yCorrect = newSeqWith(N, 0.0)
  let yError = newSeqWith(N, sigma)
  let chi = chi2(yMeasure, yCorrect, yError)
  # Check that the mean χ² is around 1 
  check chi / N.float in 0.90 .. 1.1


