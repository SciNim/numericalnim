# Package Information
version = "0.8.6"
author = "Hugo Granström"
description = "A collection of numerical methods written in Nim. Current features: integration, ode, optimization."
license = "MIT"
srcDir = "src"

# Dependencies
requires "nim >= 1.0"
requires "arraymancer >= 0.5.0"
requires "https://github.com/HugoGranstrom/cdt#head"

template installTestDeps() =
  exec "nimble install -y https://github.com/SciNim/Measuremancer.git"

task testDeps, "Install external dependencies required for tests":
  ## installs all required external dependencies that are only used
  ## for tests
  installTestDeps()

task nimCI, "Tests that should be run by the Nim CI":
  installTestDeps()
  exec "nim c -r --gc:refc tests/test_integrate.nim"
  exec "nim c -r --gc:orc tests/test_integrate.nim"

task test, "Run all tests":
  exec "nim c -r tests/test_integrate.nim"
  exec "nim c -r tests/test_interpolate.nim"
  exec "nim c -r tests/test_ode.nim"
  exec "nim c -r tests/test_optimize.nim"
  exec "nim c -r tests/test_utils.nim"
  exec "nim c -r tests/test_vector.nim"

task docs, "Generate documentation":
  # Based on Nico's script
  exec "nim doc --project --index:on --git.url:https://github.com/SciNim/numericalnim --git.commit:master --outdir:docs src/numericalnim.nim"
  exec "echo \"<meta http-equiv=\\\"Refresh\\\" content=\\\"0; url='theindex.html'\\\" />\" >> docs/index.html"
