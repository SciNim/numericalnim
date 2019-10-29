import unittest, math
import arraymancer
import numericalnim

test "Vector creation from array":
    let c = [0.0, 1.0, 2.0, 3.3]
    let v = newVector(c)
    check v.components == @c

test "Vector creation from seq":
    let c = @[0.0, 1.0, 2.0, 3.3]
    let v = newVector(c)
    check v.components == c

test "Vector equality ==":
    let v1 = newVector([0.0, 1.0, 2.2, 100.0])
    let v2 = newVector([0.0, 1.0, 2.2, 100.0])
    check (v1 == v2) == true
    check (v1 == v1) == true

test "Vector negate (-Vector)":
    let v1 = newVector([1.0, 2.5, -3.34])
    let correct = newVector([-1.0, -2.5, 3.34])
    check correct == -v1

test "Vector addition":
    let v1 = newVector([1.1, 2.2, 3.3])
    let v2 = newVector([3.3, 2.2, 1.0])
    let v3 = v1 + v2
    check v3.components == @[1.1+3.3, 2.2+2.2, 3.3+1.0]

test "Vector-scalar addition":
    let v1 = newVector([1.1, 2.2, 3.3])
    let d = 8.98
    let v2 = v1 + d
    let v3 = d + v1
    let correct = newVector([1.1+d, 2.2+d, 3.3+d])
    check v2 == v3
    check v2 == correct

test "Vectors of different sizes":
    let v1 = newVector([1.0, 2.0, 4.0, 1.34, 9.9])
    let v2 = newVector([3.3, 2.2, 1.1, 5.67])
    expect(ValueError):
        discard v1 + v2

test "Vector subtraction":
    let v1 = newVector([1.1, 2.2, 3.3])
    let v2 = newVector([3.3, 2.2, 1.0])
    let v3 = v1 - v2
    let correct = newVector([1.1-3.3, 2.2-2.2, 3.3-1.0])
    check v3 == correct

test "Vector-scalar subtraction":
    let v1 = newVector([1.1, 2.2, 3.3])
    let d = 8.98
    let v2 = v1 - d
    let v3 = d - v1
    let correct2 = newVector([1.1-d, 2.2-d, 3.3-d])
    let correct3 = newVector([d-1.1, d-2.2, d-3.3])
    check v2 == correct2
    check v3 == correct3

test "Vector +=":
    var v1 = newVector([0.0, 0.0, 0.0])
    let v2 = newVector([1.0, 2.0, 3.0])
    v1 += v2
    check v1 == v2

test "Vector-scalar +=":
    var v1 = newVector([1.1, 2.2, 3.3])
    let d = 6.78
    v1 += d
    let correct = newVector([1.1+d, 2.2+d, 3.3+d])
    check v1 == correct

test "Vector -=":
    var v1 = newVector([0.0, 0.0, 0.0])
    let v2 = newVector([1.0, 2.0, 3.0])
    v1 -= v2
    check v1 == -v2

test "Vector-scalar -=":
    var v1 = newVector([1.1, 2.2, 3.3])
    let d = 6.78
    v1 -= d
    let correct = newVector([1.1-d, 2.2-d, 3.3-d])
    check v1 == correct

test "Vector-scalar division":
    let v1 = newVector([0.0, 1.0, 2.0, 3.0])
    let d = 2.0
    let v2 = v1 / d
    let correct = newVector([0.0/d, 1.0/d, 2.0/d, 3.0/d])
    check v2 == correct

test "Vector-scalar multiplication":
    let v1 = newVector([0.0, 1.0, 2.0, 3.0])
    let d = 2.0
    let v2 = v1 * d
    let correct = newVector([0.0*d, 1.0*d, 2.0*d, 3.0*d])
    check v2 == correct

test "Vector-scalar multiplication reverse order":
    let v1 = newVector([0.0, 1.0, 2.0, 3.0])
    let d = 2.0
    let v2 = d * v1
    let correct = newVector([0.0*d, 1.0*d, 2.0*d, 3.0*d])
    check v2 == correct

test "Vector-Vector dot product":
    let v1 = newVector([0.0, 1.0, 2.0, 3.0])
    let v2 = newVector([1.0, 2.2, 6.3, 0.0])
    let p1 = v1 * v2
    let p2 = v1.dot(v2)
    let correct = 0.0*1.0 + 1.0*2.2 + 2.0*6.3 + 3.0*0.0
    check p1 == p2
    check p1 == correct

test "Vector elemetwise multiplication":
    let v1 = newVector([1.0, 2.0, 4.0, 1.34])
    let v2 = newVector([3.3, 2.2, 1.1, 5.67])
    let v3 = v1 *. v2
    let correct = newVector([1.0*3.3, 2.0*2.2, 4.0*1.1, 1.34*5.67])
    check v3 == correct

test "Vector elemetwise division":
    let v1 = newVector([1.0, 2.0, 4.0, 1.34])
    let v2 = newVector([3.3, 2.2, 1.1, 5.67])
    let v3 = v1 /. v2
    let correct = newVector([1.0/3.3, 2.0/2.2, 4.0/1.1, 1.34/5.67])
    check v3 == correct

test "Vector *=":
    var v1 = newVector([2.0, 3.0, 5.0])
    let d = 1.25
    v1 *= d
    let correct = newVector([2.0*1.25, 3.0*1.25, 5.0*1.25])
    check v1 == correct

test "Vector /=":
    var v1 = newVector([2.0, 3.0, 5.0])
    let d = 1.25
    v1 /= d
    let correct = newVector([2.0/1.25, 3.0/1.25, 5.0/1.25])
    check v1 == correct

test "Vector *.=":
    var v1 = newVector([2.0, 3.0, 5.0])
    let v2 = newVector([3.3, 2.2, 1.0])
    v1 *.= v2
    let correct = newVector([2.0*3.3, 3.0*2.2, 5.0*1.0])
    check v1 == correct

test "Vector *.=":
    var v1 = newVector([2.0, 3.0, 5.0])
    let v2 = newVector([3.3, 2.2, 1.0])
    v1 /.= v2
    let correct = newVector([2.0/3.3, 3.0/2.2, 5.0/1.0])
    check v1 == correct

test "Vector abs":
    let v1 = newVector([1.0, 2.0, 3.0, 4.0])
    let v1_abs = abs(v1)
    let correct = sqrt(1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0)
    check v1_abs == correct

test "Vector non-nested @ unpacking":
    let correct = @[1.0, 2.0, 4.0, 3.0]
    let v = newVector(correct)
    let vSeq = @v
    check correct == vSeq

test "Vector 1 nested @ unpacking":
    let c1 = @[1.0, 2.0, 3.0]
    let c2 = @[4.0, 5.0, 6.0]
    let c3 = @[7.0, 8.0, 9.0]
    let correct = @[c1, c2, c3]
    let v1 = newVector(c1)
    let v2 = newVector(c2)
    let v3 = newVector(c3)
    let vNested = newVector([v1, v2, v3])
    let vSeq = @vNested
    check correct == vSeq

test "Vector 2 nested @ unpacking":
    let c1 = @[1.0, 2.0, 3.0]
    var c2 = @[4.0, 5.0, 6.0]
    let c3 = @[7.0, 8.0, 9.0]
    let c = @[c1, c2, c3]
    let correct = @[c, c, c]
    var v1 = newVector(c1)
    var v2 = newVector(c2)
    let v3 = newVector(c3)
    let v = newVector([v1, v2, v3])
    let vNested = newVector([v, v, v])
    let vSeq = @vNested
    check correct == vSeq

test "Vector iterator":
    let c = @[0.0, 0.2, 0.4, 0.6, 0.9]
    let v1 = newVector(c)
    for i, val in v1:
        check val == c[i]

test "Vector Powers (integer power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = 2
    let correct = newVector(@[1.0, 4.0, 9.0, 16.0])
    check correct == (a ^ p)

test "Vector Powers (zero power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = 0
    let correct = newVector(@[1.0, 1.0, 1.0, 1.0])
    check correct == (a ^ p)

test "Vector Powers (large integer power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = 511
    let correct = newVector(@[1.0, 6.703903964971299e+153, 6.44111661076297e+243, 4.49423283715579e+307])
    let value = a ^ p
    check isClose(value, correct, tol = 1e-4)

test "Vector Powers (float power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = 2.0
    let correct = newVector(@[1.0, 4.0, 9.0, 16.0])
    check correct == (a ^ p)

test "Vector Powers (float power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = 2.5
    let correct = newVector(@[1.0, 5.656854249492381, 15.5884572681199, 32.0])
    let value = a ^ p
    check isClose(value, correct, tol = 1e-4)

test "Vector Powers (negative power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = -2.5
    let correct = newVector(@[pow(1.0, -2.5), pow(2.0, -2.5), pow(3.0, -2.5), pow(4.0, -2.5)])
    let value = a ^ p
    check isClose(value, correct, tol = 1e-4)

test "Vector Powers (float power)":
    let a = newVector([1.0, 2.0, 3.0, 4.0])
    let p = 0.5
    let correct = newVector(@[1.0, 1.414213562373095, 1.732050807568877, 2.0])
    let value = a ^ p
    check isClose(value, correct, tol = 1e-4)

test "Vector Norms (infinity norm)":
    let vec = newVector([1.0,2.0,3.0,4.0,5.0,5.0,5.0])
    let value = norm(vec, 0)
    let correct = 5.0
    check correct == norm(vec, 0)
    check isClose(value, correct, tol = 1e-4)

test "Vector Norms (1 norm)":
    let vec = newVector([1.0,2.0,3.0,4.0,5.0,5.0,5.0])
    let value = norm(vec, 1)
    let correct = 25.0
    check isClose(value, correct, tol = 1e-4)

test "Vector Norms (2 norm)":
    let vec = newVector([1.0,2.0,3.0,4.0,5.0,5.0,5.0])
    let value = norm(vec, 2)
    let correct = 10.2469507659596
    check isClose(value, correct, tol = 1e-4)

test "Vector Norms (2 norm) equivalent to abs":
    let vec = newVector([1.0,2.0,3.0,4.0,5.0,5.0,5.0])
    let value = norm(vec, 2)
    let correct = vec.abs()
    check isClose(value, correct, tol = 1e-4)

test "Vector Norms (p norm)":
    let vec = newVector([1.0,2.0,3.0,4.0,5.0,5.0,5.0])
    let value = norm(vec, 4)
    let correct = 6.871119744004721
    check isClose(value, correct, tol = 1e-4)

test "Vector Norms (p norm)":
    let vec = newVector([1.0,2.0,3.0,4.0,5.0,5.0,5.0])
    let value = norm(vec, 15)
    let correct = 5.384187447546214
    check isClose(value, correct, tol = 1e-4)
    
