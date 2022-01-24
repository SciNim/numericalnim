# Package Information
version = "0.6.3"
author = "Hugo Granström"
description = "A collection of numerical methods written in Nim. Current features: integration, ode, optimization."
license = "MIT"
srcDir = "src"

# Dependencies
requires "nim >= 1.0"
requires "arraymancer >= 0.5.0"
requires "https://github.com/HugoGranstrom/cdt#head"

task testDeps, "Install external dependencies required for tests":
  ## installs all required external dependencies that are only used
  ## for tests
  exec "nimble install https://github.com/SciNim/Measuremancer.git"

task test, "Run all tests":
  exec "nim c -r --experimental:unicodeOperators tests/test_integrate.nim"
  exec "nim c -r tests/test_interpolate.nim"
  exec "nim c -r tests/test_ode.nim"
  exec "nim c -r tests/test_optimize.nim"
  exec "nim c -r tests/test_utils.nim"
  exec "nim c -r tests/test_vector.nim"
