# Package

version       = "0.3.1"
author        = "Sebastian Schmidt"
description   = "A wrapper of the C library NLOPT for non-linear optimization"
license       = "MIT"
srcDir        = "src"
skipDirs      = @["examples, c_headers"]
skipExt       = @["nim~"]


# Dependencies

requires "nim >= 0.18.0"

task test, "Runs all tests":
  exec "nim c -r tests/tsimple.nim"
