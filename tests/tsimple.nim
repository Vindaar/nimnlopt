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
    center: tuple[a, b: float]

  FitGradObj = object
    a: float
    b: float

# define a helper proc, which will be optimized for
proc excentricity(p: seq[float], func_data: FitObject): float = 
  # this function calculates the excentricity of a found pixel cluster using nimnlopt.
  # Since no proper high level library is yet available, we need to pass a var pointer
  # of func_data, which contains the x and y arrays in which the data is stored, in
  # order to calculate the RMS variables

  # first recover the data from the pointer to func_data, by casting the
  # raw pointer to a Cluster object
  let fit = func_data
  let c = fit.cluster

  let (centerX, centerY) = fit.center
  var
    sum_x: float = 0
    sum_y: float = 0
    sum_x2: float = 0
    sum_y2: float = 0

  for i in 0..<len(c):
    let
      new_x = cos(p[0]) * (float(c[i].a) - centerX) * 0.055 - sin(p[0]) * (float(c[i].b) - centerY) * 0.055
      new_y = sin(p[0]) * (float(c[i].a) - centerX) * 0.055 + cos(p[0]) * (float(c[i].b) - centerY) * 0.055
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
  # TODO: add grad to `FuncProto` in some way
  # if addr(grad) != nil:
  #   # normally we'd calculate the gradient for the current parameters, but we're
  #   # not going to use it. Can also remove this whole if statement
  #   discard
  result = -exc

proc gradExample(p: seq[float], grad: seq[float], func_data: FitGradObj): (float, seq[float]) =
  var nGrad = newSeq[float](p.len)
  nGrad[0] = 0.0
  nGrad[1] = 0.5 / sqrt(p[1])
  # calculate the actual value of the function
  result = (sqrt(p[1]), nGrad)

proc gradExampleConstraint(p: seq[float], grad: seq[float], func_data: FitGradObj): (float, seq[float]) =
  let
    a = func_data.a
    b = func_data.b
  var nGrad = newSeq[float](grad.len)
  nGrad[0] = 3.0 * a * (a*p[0] + b) * (a*p[0] + b)
  nGrad[1] = -1.0
  # calculate the actual value of the function
  let val = (a*p[0] + b) * (a*p[0] + b) * (a*p[0] + b) - p[1]

  # now return the value of the function as well as the new gradients
  result = (val, nGrad)

template time_block(actions: untyped) =
  # just a benchmark template
  let t0 = epochTime()
  for _ in 0 ..< 1_000:
    actions
  echo "Block took $# to execute" % $(epochTime() - t0)

when isMainModule:

  block:
    # a simple example of optimizing the excentricity via the rotation angle
    # of a (trivial) 2D "cluster"
    let opt_name = "LN_COBYLA"
    # create new NloptOpt object, choosing an algorithm and already
    # setting upper and lower bounds
    var opt: NloptOpt = newNloptOpt(opt_name, 1, @[(-3.0, 3.0)])

    # check whether setting an algorithm works
    let opt_name_tab = getNloptAlgorithmTable()
    check: nlopt_get_algorithm(opt.optimizer) == opt_name_tab[opt_name]

    # check if bound setting works
    # NOTE: this is purely for simple verification and unnecessary
    block:
      var
        x_l: cdouble = 0
        x_u: cdouble = 0
        status: nlopt_result = NLOPT_SUCCESS
        
      status = nlopt_get_lower_bounds(opt.optimizer, addr(x_l))
      status = nlopt_get_upper_bounds(opt.optimizer, addr(x_u))
      if status == NLOPT_SUCCESS:
        check: x_l == opt.lBounds[0]
        check: x_u == opt.uBounds[0]

    # given two sequences from 0..99, one for x, the other for y,
    # defines a line from (0, 0) to (99, 99), i.e. a line with
    # slope 1 and thus, if interpreted as an ellipse, an ellipse
    # which is rotated by 45 degrees and an eccentricity, which
    # approaches Inf
    let
      x = toSeq(0..99)
      y = toSeq(0..99)
      center = (45.0, 45.0)
      xy = zip(x, y)

    var fobj = FitObject(cluster: xy, center: center)
    # now instantiate the generic `VarStruct` object with our
    # optimization function and our data object, we're using in
    # that function

    echo "Equality of func ", excentricity is FuncProto
    # either
    #var vars = VarStruct[FitObject](userFunc: excentricity, data: fobj, kind: FuncKind.NoGrad)
    # or use the `newVarStruct` template
    var vars = newVarStruct(excentricity, fobj)

    opt.setFunction(vars)
    # TODO: include a simple fit of a known distribution to check
    # library is working
    # check: 

    # time the optimization procedure over 10_000 iterations, just for
    # curiosity's sake
    echo "Performing 1_000 iterations of full optimization..."
    var
      params: seq[float]
      minVal: float
    time_block:
      var p: seq[float] = @[0.0]
      # set initial step size
      opt.initial_step = 0.1
      # set some stopping criteria
      opt.ftol_rel = 1e-9
      opt.xtol_rel = 1e-9
      
      (params, min_val) = opt.optimize(p)
      # the following check is only very rough, since we do not want to
      # start comparing float values. Instead we simply round to the next
      # integer and check whether it's 45, as we expect
      check: abs(round(radToDeg(params[0]))) == 45
    echo "\tOptimization resulted in f(p[0]) = $# at p[0] = $#" % [$min_val, $params[0]]
      

    # finally destroy the optimizer
    nlopt_destroy(opt.optimizer)

  block:
    let opt_name = "LD_MMA"
    # create new NloptOpt object, choosing an algorithm and already
    # setting upper and lower bounds
    let bounds = @[(-Inf, Inf), (0.0, Inf)]
    var opt: NloptOpt = newNloptOpt(opt_name, 2, bounds)
    let p = @[1.234, 5.678]

    # check whether setting an algorithm works
    let opt_name_tab = getNloptAlgorithmTable()
    check: nlopt_get_algorithm(opt.optimizer) == opt_name_tab[opt_name]

    var fobjs = newSeq[FitGradObj](2)
    fobjs[0] = FitGradObj(a: 2.0, b: 0.0)
    fobjs[1] = FitGradObj(a: -1.0, b: 1.0)
    var varConstraint1 = newVarStruct(gradExampleConstraint, fobjs[0])
    var varConstraint2 = newVarStruct(gradExampleConstraint, fobjs[1])    
    opt.addInequalityConstraint(varConstraint1)
    opt.addInequalityConstraint(varConstraint2)
    # now instantiate the generic `VarStruct` object with our
    # optimization function and our data object, we're using in
    # that function

    echo "Equality of func ", gradExample is FuncProtoGrad
    # either
    #var vars = VarStruct[FitObject](userFunc: excentricity, data: fobj, kind: FuncKind.NoGrad)
    # or use the `newVarStruct` template
    #let fobj = FitGradObj
    # just set the object to nil
    # need a dummy object for the data field! Just reuse one of the constraint objects for now
    # TODO: allow for empty field!
    var vars = newVarStruct(gradExample, fobjs[0])

    opt.setFunction(vars)

    # stopping criteria / step size
    # set initial step size
    opt.initialStep = 0.1
    # set some stopping criteria
    opt.ftolRel = 1e-9
    opt.xtolRel = 1e-4
    
    # time the optimization procedure over 10_000 iterations, just for
    # curiosity's sake
    echo "Performing 1_000 iterations of full optimization..."
    var
      params: seq[float]
      minVal: float
    
    time_block:
      (params, minVal) = opt.optimize(p)
      # the following check is only very rough, since we do not want to
      # start comparing float values. Instead we simply round to the next
      # integer and check whether it's 45, as we expect
      check: round(params[0], places = 2) == 0.33
      check: round(params[1], places = 2) == 0.30
      check: round(minVal, places = 2) == 0.54

    echo "\tFound minimum at f(",params[0],", ", params[1], ") = ", min_val,"\n"
    # finally destroy the optimizer
    nlopt_destroy(opt.optimizer)
