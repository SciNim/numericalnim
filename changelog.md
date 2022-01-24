# v0.7.0 - 25.01.2022

This is a *breaking* release, due to the changes in PR #25.

`NumContext` (and types taking `NumContext` as an argument) are now
two-fold generic. The floating point like type used during
computation may now be overwritten.

This is a breaking change, as the `newNumContext` procedure must now
be given two generic arguments. For most procedures the signature
was only extended to use `float` as the secondary type, leaving them
as taking single generic arguments.
`adapdiveGauss` is an exception and thus now requires the user to
hand *both* types.

- transition for `adaptiveGauss`:
  Calling as: `adaptiveGauss[T, float](...)` will produce the old
  behavior. In the future a nicer interface may be designed.
- transition for `newNumContext`:
  Calling as: `newNumContext[T, float]` will produce the old behavior.

This change was a step towards a more (likely concept based) interface
for SciNim libraries for better interop. It allows for example to
integrate over a `Measurement`.

# v0.6.3

- fixes an issue that might arise if 2D interpolation is used together
  with multithreading where the Nim compiler gets confused about GC
  unsafety (#23)
- Added `linear1d`
- Added `barycentric2d` which works on non-gridded data.
