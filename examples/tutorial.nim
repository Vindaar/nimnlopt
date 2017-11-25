import os
import math
import ../nlopt/nimnlopt

type
  my_constraint_data = object
    a: cdouble
    b: cdouble

  # test_type = object
  #   x: seq[float]
  #   y: seq[float]
  
proc myfunc(n: cuint, x: array[2, cdouble], grad: var array[2, cdouble], my_func_data: var pointer): cdouble {.cdecl.} =
  # let xy = cast[test_type](my_func_data)
  # for i in 0..<len(xy.x):
  #   echo xy.x[i]
  
  if addr(grad) != nil:
    grad[0] = 0.0
    grad[1] = 0.5 / sqrt(x[1])
  result = sqrt(x[1])

proc myconstraint(n: cuint, x: array[2, cdouble], grad: var array[2, cdouble], data: pointer): cdouble {.cdecl.} =
  let
    d: ptr my_constraint_data = cast[ptr my_constraint_data](data)
    a: cdouble = d.a
    b: cdouble = d.b
  if addr(grad) != nil:
    grad[0] = 3.0 * a * (a*x[0] + b) * (a*x[0] + b)
    grad[1] = -1.0
  result = (a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]
  
proc main() =
  var
    # lower bounds
    lb: array[2, cdouble] = [cdouble(-Inf), cdouble(0.0)]
    opt: nlopt_opt
    data: array[2, my_constraint_data]
    minf: cdouble
    # some initial guess
    x: array[2, cdouble] = [1.234, 5.678]

  data[0] = my_constraint_data(a: 2, b: 0)
  data[1] = my_constraint_data(a: -1, b: 1)
  # algorithm and dimensionality
  echo NLOPT_LD_MMA
  #opt = nlopt_create(NLOPT_LN_COBYLA, 2)
  opt = nlopt_create(NLOPT_LD_MMA, 2)
  var status: nlopt_result
  status = nlopt_set_lower_bounds(opt, addr(lb[0]))
  echo status


  # var xy: test_type
  # xy = test_type(x: @[], y: @[])
  # for i in 0..<100:
  #   xy.x.add(float(i))
  # for i in 100..<200:
  #   xy.y.add(float(i))
  
  status = nlopt_set_min_objective(opt, cast[nlopt_func](myfunc), nil)#cast[pointer](addr xy))
  echo status

  status = nlopt_add_inequality_constraint(opt, cast[nlopt_func](myconstraint), addr(data[0]), 1e-8)
  status = nlopt_add_inequality_constraint(opt, cast[nlopt_func](myconstraint), addr(data[1]), 1e-8)

  status = nlopt_set_xtol_rel(opt, 1e-4)
  echo status  
  # the minimum objective value, upon return
  echo "starting minimization"
  let t = nlopt_optimize(opt, addr(x[0]), addr(minf))#addr(x[0]), addr(minf))
  echo "done"
  if cast[int](t) < 0:
    echo t
    echo "nlopt failed!\n"
  else:
    echo t
    echo "found minimum at f(",x[0]," , ", x[1], "), = ", minf,"\n"

  nlopt_destroy(opt)

when isMainModule:
  main()
