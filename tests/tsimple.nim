import unittest
import sequtils
import math
import times
import nimnlopt
import tables
import strutils

# define an object to store the data for the eccentricity proc
type
  FitObject = object
    cluster: seq[tuple[a, b: int]]
    xy: tuple[a, b: float]

  FuncProto = proc (p: seq[float], varargs[seq[float]]): float

  VarsStruct = ref object
    userFunc: FuncProto

proc optimizeImpl(n: cuint, p: array[1, cdouble], grad: var array[1, cdouble], func_data: var pointer): cdouble {.cdecl.} =
  # func_data contains the actual function, which we fit
  let ff = cast[VarsStruct](func_data)
  let vecs = ff.

#proc exc(p: seq[float], cl: seq[float]): float =
  

# define a helper proc, which will be optimized for
proc excentricity(n: cuint, p: array[1, cdouble], grad: var array[1, cdouble], func_data: var pointer): cdouble {.cdecl.} =
  # this function calculates the excentricity of a found pixel cluster using nimnlopt.
  # Since no proper high level library is yet available, we need to pass a var pointer
  # of func_data, which contains the x and y arrays in which the data is stored, in
  # order to calculate the RMS variables

  # first recover the data from the pointer to func_data, by casting the
  # raw pointer to a Cluster object
  let fit = cast[FitObject](func_data)
  let c = fit.cluster

  let (x, y) = fit.xy
  var
    sum_x: float = 0
    sum_y: float = 0
    sum_x2: float = 0
    sum_y2: float = 0

  for i in 0..<len(c):
    let
      new_x = cos(p[0]) * (float(c[i].a) - x) * 0.055 - sin(p[0]) * (float(c[i].b) - y) * 0.055
      new_y = sin(p[0]) * (float(c[i].a) - x) * 0.055 + cos(p[0]) * (float(c[i].b) - y) * 0.055
    sum_x += new_x
    sum_y += new_y
    sum_x2 += (new_x * new_x)
    sum_y2 += (new_y * new_y)
  
  let
    n_elements: float = float(len(c))
    rms_x: float = sqrt( (sum_x2 / n_elements) - (sum_x * sum_x / n_elements / n_elements))
    rms_y: float = sqrt( (sum_y2 / n_elements) - (sum_y * sum_y / n_elements / n_elements))    

  let exc = cdouble(rms_x / rms_y)
  # need to check whether grad is nil. Only used for some algorithms, otherwise a
  # NULL pointer is handed in C
  if addr(grad) != nil:
    # normally we'd calculate the gradient for the current parameters, but we're
    # not going to use it. Can also remove this whole if statement
    discard
  result = -exc

template time_block(actions: untyped) {.dirty.} =
  # just a benchmark template
  let t0 = epochTime()
  for _ in 0..<10000:
    actions
  echo "Block took $# to execute" % $(epochTime() - t0)

when isMainModule:    
  let opt_name = "LN_COBYLA"
  # create new NloptOpt object, choosing an algorithm and already
  # setting upper and lower bounds
  var opt: NloptOpt = newNloptOpt(opt_name, (-3.0, 3.0))

  # check whether setting an algorithm works
  let opt_name_tab = getNloptAlgorithmTable()
  check: nlopt_get_algorithm(opt.optimizer) == opt_name_tab[opt_name]

  # check if bound setting works
  var
    x_l: cdouble = 0
    x_u: cdouble = 0
    status: nlopt_result = NLOPT_SUCCESS
    
  status = nlopt_get_lower_bounds(opt.optimizer, addr(x_l))
  status = nlopt_get_upper_bounds(opt.optimizer, addr(x_u))
  if status == NLOPT_SUCCESS:
    check: x_l == opt.l_bound
    check: x_u == opt.u_bound

  # given two sequences from 0..99, one for x, the other for y,
  # defines a line from (0, 0) to (99, 99), i.e. a line with
  # slope 1 and thus, if interpreted as an ellipse, an ellipse
  # which is rotated by 45 degrees and an eccentricity, which
  # approaches Inf
  let
    x = toSeq(0..99)
    y = toSeq(0..99)
    xy = (45.0, 45.0)
    zz = zip(x, y)

  var fobj = FitObject(cluster: zz, xy: xy)

  opt.setFunction(excentricity, fobj)
  # TODO: include a simple fit of a known distribution to check
  # library is working
  # check: 

  # time the optimization procedure over 10_000 iterations, just for
  # curiosity's sake
  echo "Performing 10_000 iterations of full optimization..."
  time_block:
    var p: seq[float] = @[0.0]
    # set initial step size
    opt.initial_step = 0.1
    # set some stopping criteria
    opt.ftol_rel = 1e-9
    opt.xtol_rel = 1e-9
    
    let (params, min_val) = opt.optimize(p)
    #echo "Optimization resulted in f(p[0]) = $# at p[0] = $#" % [$min_val, $params[0]]
    # the following check is only very rough, since we do not want to
    # start comparing float values. Instead we simply round to the next
    # integer and check whether it's 45, as we expect
    check: abs(round(radToDeg(params[0]))) == 45
    

  # finally destroy the optimizer
  nlopt_destroy(opt.optimizer)
