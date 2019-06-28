import unittest, math, sequtils, algorithm
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