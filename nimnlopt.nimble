# Package

version       = "0.1.0"
author        = "Sebastian Schmidt"
description   = "A wrapper of the C library NLOPT for non-linear minimization"
license       = "LGPL"
srcDir        = "nlopt"
skipDirs      = @["examples, c_headers"]
skipExt       = @["nim~"]


# Dependencies

requires "nim >= 0.17.0"

