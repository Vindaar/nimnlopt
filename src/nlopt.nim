import nlopt/nlopt_wrapper
export nlopt_wrapper
import tables
import macros
import strutils, sequtils, strformat
import typetraits


# this file provides the (extremely limited) high level functionality of the NLopt
#  nim library, especially containing the type conversion to compatible types and
# dealing with addresses and pointer

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
  FuncProtoGrad*[T] = proc (p: seq[float], func_data: T): (float, seq[float])
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

  NloptOpt*[T] = object
    optimizer*: nlopt_opt
    userData: VarStruct[T] # reference to the user data to avoid it being GC'ed
    optKind*: nlopt_algorithm
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

proc newVarStruct*[T, U](uFunc: T, data: U): VarStruct[U] =
  # NOTE: If `U` itself is a generic, calling this function may run into:
  # https://github.com/nim-lang/Nim/issues/11778
  # In that case create the VarStruct manually.
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

proc newNloptOpt*[T](optName: nloptAlgorithm, nDims: int, bounds: seq[tuple[l, u: float]] = @[]): NloptOpt[T] =
  ## creator of a new NloptOpt object, which takes a string describing the algorithm to be
  ## used as well as (optionally) the lower and upper bounds to be used, as a tuple
  ## `nDims` is the dimensionality of the problem to be optimized. If `bounds` is given, there
  ## needs to be one tuple for each dimension
  ## If no bounds are given, they are set to `-Inf` to `Inf`
  ## TODO: add options to also already add arguments for other fields of NloptOpt
  doAssert bounds.len == 0 or bounds.len == nDims, " need bounds for each dimension!"

  var
    opt: nlopt_opt
    status: nlopt_result
    f: nlopt_func
  status = NLOPT_SUCCESS

  opt = nlopt_create(optName, nDims.cuint)

  # extract and set bounds
  var
    lBounds = bounds.mapIt(if it.l != it.u: it.l.cdouble else: -Inf)
    uBounds = bounds.mapIt(if it.l != it.u: it.u.cdouble else: Inf)
  if bounds.len > 0:
    status = nlopt_set_lower_bounds(opt, addr lBounds[0])
    status = nlopt_set_upper_bounds(opt, addr uBounds[0])

  result = NloptOpt[T](optimizer: opt,
                    optKind: optName,
                    opt_name: $optName,
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
      let ufunc = ff.userFuncGrad
      if not gradPtr.isNil:
        var gradAr = cast[ptr UncheckedArray[float]](gradPtr)
        # in this case grad is defined, set the values
        let (res, newGrad) = ufunc(p, fobj)
        # set the new values for the gradient
        for i in 0 ..< n.int:
          gradAr[i] = newGrad[i]
        # finally set result of proc
        result = res
      else:
        let (res, newGrad) = ufunc(p, fobj)
        result = res

  optimizeImpl

#proc setFunction*[T](nlopt: var NloptOpt, vStruct: T) =
proc setFunction*[T](nlopt: var NloptOpt, vStruct: var VarStruct[T]) =
  ## wrap the user defined proc, which is part of `fObj` (a field
  ## of the object with name `userFunc`)
  const genFunc = genOptimizeImpl(type(vStruct))

  # assign `vStruct` as `userData`, so that we keep a reference to it around
  # so that `addr nlopt.userData` is valid, even if the `vStruct` given goes
  # out of scope in the calling scope of `setFunction`.
  nlopt.userData = vStruct
  nlopt.opt_func = cast[nlopt_func](genFunc)
  nlopt.status = nlopt_set_min_objective(nlopt.optimizer,
                                         nlopt.opt_func,
                                         cast[pointer](addr nlopt.userData))

proc addInequalityConstraint*[T](nlopt: var NloptOpt, vStruct: var VarStruct[T]) =
  ## adds an inequality constraint to the optimizer, i.e. a function, which
  ## is evaluated regarding some constraint on the data
  # TODO: add equivalent forr equality constraints
  # TODO: allow custom setting of the `tol` parameter of the constraints

  # the inequality function also needs to be of the same signature as the optimization
  # function, so either `FuncProto` or `FuncProtoGrad`
  const genFunc = genOptimizeImpl(type(vStruct))
  nlopt.status = nlopt_add_inequality_constraint(nlopt.optimizer,
                                                 cast[nlopt_func](genFunc),
                                                 cast[pointer](addr vStruct),
                                                 1e-8)

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

proc setLocalOptimizer*(nlopt: var NLoptOpt, local: NloptOpt) =
  ## assigns the local optimizer `local` as the local optimizer for `nlopt`.
  ## This is only applicable for certain algorithms. If `nlopt` has the
  ## wrong algorithm, this call will fail!
  case nlopt.optKind
  of G_MLSL_LDS, G_MLSL, AUGLAG, AUGLAG_EQ:
    # create the local optimizer
    nlopt.status = nlopt.optimizer.nlopt_set_local_optimizer(local.optimizer)
    if nlopt.status != NLOPT_SUCCESS:
      raise newException(Exception, "Could not set local optimizer of kind " &
        local.optName & "for parent optimizer of kind " & nlopt.optName & "!")
  else:
    raise newException(Exception, "Unsupported kind to set local optimizer: " & $nlopt.optKind)


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
