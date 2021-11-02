# v0.6.3

- fixes an issue that might arise if 2D interpolation is used together
  with multithreading where the Nim compiler gets confused about GC
  unsafety (#23)
- Added `linear1d`
- Added `barycentric2d` which works on non-gridded data.
