import nimnlopt/nlopt_wrapper
export nlopt_wrapper
import tables
import macros
import strutils, sequtils, strformat
import typetraits


# this file provides the (extremely limited) high level functionality of the NLopt
#  nim library, especially containing the type conversion to compatible types and 
# dealing with addresses and pointer

type
  NloptOpt* = object
    optimizer*: nlopt_opt
    optName*: string
    nDims*: int
    lBounds*: seq[float]
    uBounds*: seq[float]
    xtolRel*: float
    xtolAbs*: float
    ftolRel*: float
    ftolAbs*: float
    maxTime*: float
    maxEval*: int
    initialStep*: float
    status*: nlopt_result
    optFunc*: nlopt_func

  # NloptFunc is the user defined function, which takes
  # - an input seq or openArray (to be impl'd)
  #   - one element for each parameter to fit
  # - a mutable gradient for the gradients 
  #   - d func / dx_i, i for each parameter in x
  # func_data: object = a user defined object, which can be used
  #   to hand specific data to the user defined function, if desired
  #   - e.g. if for each parameter, f depends on an array of input data
  #     one can create an object, which stores said data, hand it to
  #     NloptFunc and it will be available in the function
type

  FuncKind* {.pure.} = enum
    NoGrad, Grad
  
  FuncProto*[T] = proc (p: seq[float], func_data: T): float
  FuncProtoGrad*[T] = proc (p: seq[float], grad: seq[float], func_data: T): (float, seq[float])
  VarStruct*[T] = ref object
    # a generic object, which the user has to initialize with a
    # function following the FuncProto signature and any custom 
    # object in `data`
    case kind*: FuncKind:
    of NoGrad:
      userFunc*: FuncProto[T]
    of Grad:
      userFuncGrad*: FuncProtoGrad[T]
    data*: T
# * = proc (n: cuint; x: ptr cdouble; gradient: ptr cdouble; func_data: pointer): cdouble {.cdecl.}

# template newVarStruct*(uFunc: typed, fobj: typed): untyped =    
proc newVarStruct*[T, U](uFunc: T, data: U): VarStruct[U] =
  when T is FuncProto:
    result = VarStruct[U](userFunc: uFunc, data: data, kind: FuncKind.NoGrad)
  elif T is FuncProtoGrad:
    result = VarStruct[U](userFuncGrad: uFunc, data: data, kind: FuncKind.Grad)
  else:
    raise newException(AssertionError, "Unexpected type for first argument of " &
      "`newVarStruct`. Allowed proc types are: `FuncProto` and `FuncProtoGrad`.\n" &
      "Is: " & T.name)
    
template withDebug(actions: untyped) =
  when defined(DEBUG):
    actions

proc getNloptAlgorithmTable*(): Table[string, nlopt_algorithm] =
  # TODO: clean up nlopt_wrapper's definition of the `nlopt_algorithm`
  # enum. And then define a custom pure enum to remove the `NLOPT_` prefixes.
  result = { "GN_DIRECT" : NLOPT_GN_DIRECT,
             "GN_DIRECT_L" : NLOPT_GN_DIRECT_L,
             "GN_DIRECT_L_RAND" : NLOPT_GN_DIRECT_L_RAND,
             "GN_DIRECT_NOSCAL" : NLOPT_GN_DIRECT_NOSCAL,
             "GN_DIRECT_L_NOSCAL" : NLOPT_GN_DIRECT_L_NOSCAL,
             "GN_DIRECT_L_RAND_NOSCAL" : NLOPT_GN_DIRECT_L_RAND_NOSCAL,
             "GN_ORIG_DIRECT" : NLOPT_GN_ORIG_DIRECT,
             "GN_ORIG_DIRECT_L" : NLOPT_GN_ORIG_DIRECT_L,
             "GD_STOGO" : NLOPT_GD_STOGO,
             "GD_STOGO_RAND" : NLOPT_GD_STOGO_RAND,
             "LD_LBFGS_NOCEDAL" : NLOPT_LD_LBFGS_NOCEDAL,
             "LD_LBFGS" : NLOPT_LD_LBFGS,
             "LN_PRAXIS" : NLOPT_LN_PRAXIS,
             "LD_VAR1" : NLOPT_LD_VAR1,
             "LD_VAR2" : NLOPT_LD_VAR2,
             "LD_TNEWTON" : NLOPT_LD_TNEWTON,
             "LD_TNEWTON_RESTART" : NLOPT_LD_TNEWTON_RESTART,
             "LD_TNEWTON_PRECOND" : NLOPT_LD_TNEWTON_PRECOND,
             "LD_TNEWTON_PRECOND_RESTART" : NLOPT_LD_TNEWTON_PRECOND_RESTART,
             "GN_CRS2_LM" : NLOPT_GN_CRS2_LM,
             "GN_MLSL" : NLOPT_GN_MLSL,
             "GD_MLSL" : NLOPT_GD_MLSL,
             "GN_MLSL_LDS" : NLOPT_GN_MLSL_LDS,
             "GD_MLSL_LDS" : NLOPT_GD_MLSL_LDS,
             "LD_MMA" : NLOPT_LD_MMA,
             "LN_COBYLA" : NLOPT_LN_COBYLA,
             "LN_NEWUOA" : NLOPT_LN_NEWUOA,
             "LN_NEWUOA_BOUND" : NLOPT_LN_NEWUOA_BOUND,
             "LN_NELDERMEAD" : NLOPT_LN_NELDERMEAD,
             "LN_SBPLX" : NLOPT_LN_SBPLX,
             "LN_AUGLAG" : NLOPT_LN_AUGLAG,
             "LD_AUGLAG" : NLOPT_LD_AUGLAG,
             "LN_AUGLAG_EQ" : NLOPT_LN_AUGLAG_EQ,
             "LD_AUGLAG_EQ" : NLOPT_LD_AUGLAG_EQ,
             "LN_BOBYQA" : NLOPT_LN_BOBYQA,
             "GN_ISRES" : NLOPT_GN_ISRES,
             # new variants that require local_optimizer to be set,
             # not with older constants for backwards compatibility
             "AUGLAG" : NLOPT_AUGLAG,
             "AUGLAG_EQ" : NLOPT_AUGLAG_EQ,
             "G_MLSL" : NLOPT_G_MLSL,
             "G_MLSL_LDS" : NLOPT_G_MLSL_LDS,
             "LD_SLSQP" : NLOPT_LD_SLSQP,
             "LD_CCSAQ" : NLOPT_LD_CCSAQ,
             "GN_ESCH" : NLOPT_GN_ESCH }.toTable()

proc newNloptOpt*(opt_name: string, nDims: int, bounds: seq[tuple[l, u: float]] = @[]): NloptOpt =
  ## creator of a new NloptOpt object, which takes a string describing the algorithm to be
  ## used as well as (optionally) the lower and upper bounds to be used, as a tuple
  ## `nDims` is the dimensionality of the problem to be optimized. If `bounds` is given, there
  ## needs to be one tuple for each dimension
  ## If no bounds are given, they are set to `-Inf` to `Inf`
  ## TODO: add options to also already add arguments for other fields of NloptOpt
  ## TODO: fix the bounds: currently we only allow global bounds (for all dimensions the same)!
  doAssert bounds.len == 0 or bounds.len == nDims, " need bounds for each dimension!"
  
  var
    opt: nlopt_opt
    status: nlopt_result
    f: nlopt_func
  status = NLOPT_SUCCESS

  let opt_name_table = getNloptAlgorithmTable()
  
  opt = nlopt_create(opt_name_table[opt_name], nDims.cuint)

  # extract and set bounds
  var
    lBounds = bounds.mapIt(it.l.cdouble)
    uBounds = bounds.mapIt(it.u.cdouble)  
  if bounds.len > 0:
    status = nlopt_set_lower_bounds(opt, addr lBounds[0])
    status = nlopt_set_upper_bounds(opt, addr uBounds[0])
    
  result = NloptOpt(optimizer: opt,
                    opt_name: opt_name,
                    nDims: nDims,
                    lBounds: lBounds,
                    uBounds: uBounds,
                    xtol_rel: 0,
                    xtol_abs: 0,
                    ftol_rel: 0,
                    ftol_abs: 0,
                    maxtime: 0,
                    initial_step: 0,
                    status: status,
                    opt_func: f)

template genOptimizeImpl(uType: untyped): untyped =
  ## helper template to generate the wrapper around the user defined procedure
  ## with the correctly inserted type of the user data object
  ## NOTE: in the current implementation the template is not even needed, because
  ## the user may always use `VarStruct`. However, this in principle allows
  ## the user to use a custom type as well. But that type will be bound to the
  ## same field names, which makes it not useful?
  
  proc optimizeImpl(n: cuint, pPtr: ptr cdouble, gradPtr: ptr cdouble, func_data: var pointer): cdouble {.cdecl.} =
    # func_data contains the actual function, which we fit
    var p = newSeq[float](n)
    let pAr = cast[ptr UncheckedArray[float]](pPtr)

    for i in 0 ..< n.int:
      p[i] = pAr[i]
    # using the type given to the template, cast the user function data
    # to that type
    # NOTE: while technically we have no way to assert that `uType` really is the
    # type of `func_data`, thanks to the way we call the `genOptimizeImpl` template
    # we do indeed make sure of that, since we call it in `setFunction`, and use
    # the generic type `T`, which is precisely the type of `func_data` as the input.
    let ff = cast[uType](func_data)
    # get the user data object
    let fobj = ff.data
    # and the user function
    case ff.kind
    of FuncKind.NoGrad:
      let ufunc = ff.userFunc
      # apply the function and return the result
      result = ufunc(p, fobj)
    of FuncKind.Grad:
      # get the gradient data
      var grad = newSeq[float](n)
      var gradAr = cast[ptr UncheckedArray[float]](gradPtr)
      
      let ufunc = ff.userFuncGrad
      # in this case grad is defined, set the values
      for i in 0 ..< n.int:
        grad[i] = gradAr[i]
      let (res, newGrad) = ufunc(p, grad, fobj)
      # set the new values for the gradient
      for i in 0 ..< n.int:
        gradAr[i] = newGrad[i]
      # finally set result of proc
      result = res      
  optimizeImpl

#proc setFunction*(nlopt: var NloptOpt, f: NloptRawFunc, f_obj: var object) =
proc setFunction*[T](nlopt: var NloptOpt, fObj: var T) =
  ## wrap the user defined proc, which is part of `fObj` (a field
  ## of the object with name `userFunc`)
  const genFunc = genOptimizeImpl(T)
  
  nlopt.opt_func = cast[nlopt_func](genFunc)
  nlopt.status = nlopt_set_min_objective(nlopt.optimizer,
                                         nlopt.opt_func,
                                         cast[pointer](addr(fObj)))


proc setMaxEval*(nlopt: var NloptOpt, val: int) = 
  nlopt.status = nlopt_set_maxeval(nlopt.optimizer, val.cint)
  
# define macro to simply create all procs to set optimizer settings. All receive same
# arguments, therefore can be done in one go
macro set_nlopt_floatvals(func_name: static[string]): typed =
  let nim_func_name: string = """
proc $#*(nlopt: var NloptOpt, val: float) = 
  nlopt.status = nlopt_$#(nlopt.optimizer, cdouble(val))""" % [func_name, func_name]
  result = parseStmt(nim_func_name)

set_nlopt_floatvals("setXtolAbs1")
set_nlopt_floatvals("setXtolRel")
set_nlopt_floatvals("setFtolAbs")
set_nlopt_floatvals("setFtolRel")
set_nlopt_floatvals("setMaxTime")
static:
  withDebug:
    echo getAst(set_nlopt_floatvals("set_ftol_rel")).repr

proc setInitialStep(nlopt: var NloptOpt, initial_step: float) =
  # simple wrapper for nlopt_set_initial_step, which takes care of
  # handing a ptr cdouble to the Nlopt function
  var dx: cdouble = cdouble(initial_step)
  nlopt.status = nlopt_set_initial_step(nlopt.optimizer, addr(dx))

template nlopt_write_or_raise(nlopt: var NloptOpt, f: untyped, field: untyped) =
  ## simple template, which checks whether a call to libnlopt should be done and
  ## if so, also checks whether the call was successful. Otherwise raises a
  ## LibraryError with a (hopefully) helpful error message
  if field != 0:
    f(field)
    withDebug:
      echo "Setting value " & $field & " with " & astToStr(f(field))
    if nlopt.status != NLOPT_SUCCESS:
      raise newException(LibraryError, "Call to libnlopt failed with error: $#" % $nlopt.status)

proc optimize*[T](nlopt: var NloptOpt, params: seq[T]): tuple[p: seq[float], f: float] =
  ## function performing the actual optimization
  ## first sets all available tolerances, stopping criteria etc. then
  ## calls nlopt_optimize()
  ## inputs:
  ##    nlopt: var NloptOpt = the object, which contains the Nlopt optimizer
  ##    params: seq[float] = a sequence containing starting values for each
  ##            parameter
  ## outputs:
  ##    tuple[p, f: seq[float]] = a tuple of params / function of params
  ##            values for the resulting parameters, e.g.
  ##            ( @[p[0], p[1], ...], @[f(p[0]), f(p[1]), ...] )
  var
    # start parameters
    p = params
    # function value. Will be assigned to in the C library
    f_p: cdouble

  # use template to set nlopt stopping criteria in libnlopt and also perform
  # checks on successful calls to libnlopt, while keeping bloat away
  nlopt_write_or_raise(nlopt, nlopt.setXtolAbs1, nlopt.xtolAbs)
  nlopt_write_or_raise(nlopt, nlopt.setXtolRel, nlopt.xtolRel)
  nlopt_write_or_raise(nlopt, nlopt.setFtolAbs, nlopt.ftolAbs)
  nlopt_write_or_raise(nlopt, nlopt.setFtolRel, nlopt.ftolRel)
  nlopt_write_or_raise(nlopt, nlopt.setMaxTime, nlopt.maxTime)
  nlopt_write_or_raise(nlopt, nlopt.setMaxEval, nlopt.maxEval)  
  nlopt_write_or_raise(nlopt, nlopt.setInitialStep, nlopt.initialStep)

  # now perform optimization
  nlopt.status = nlopt_optimize(nlopt.optimizer, addr p[0], addr f_p)
  if nlopt.status < NLOPT_SUCCESS:
    echo "Warning: nlopt optimization failed with status $#!" % $nlopt.status
  
  result = (p: p, f: f_p)

