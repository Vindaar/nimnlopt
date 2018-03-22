import nimnlopt/nlopt_wrapper
export nlopt_wrapper
import tables
import macros
import strutils

# this file provides the (extremely limited) high level functionality of the NLopt
#  nim library, especially containing the type conversion to compatible types and 
# dealing with addresses and pointer
  
type
  NloptOpt* = object
    optimizer*: nlopt_opt
    opt_name*: string
    l_bound*: float
    u_bound*: float
    xtol_rel*: float
    xtol_abs*: float
    ftol_rel*: float
    ftol_abs*: float
    maxtime*: float
    initial_step*: float
    status*: nlopt_result
    opt_func*: nlopt_func

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
  #NloptFunc = proc[T](x: seq[T], gradient: var seq[T], func_data: object): float
  # raw func pointer, which is the definition we need to hand to NLopt. For now demand
  # a raw func to be defined by the user, switch to macro creating NloptRawFunc from NloptFunc
  # at a later time
  NloptRawFunc[N: static[int]] = proc(n: cuint, p: array[N, cdouble], grad: var array[N, cdouble], func_data: var pointer): cdouble {.cdecl.}
  
# * = proc (n: cuint; x: ptr cdouble; gradient: ptr cdouble; func_data: pointer): cdouble {.cdecl.}

template withDebug(actions: untyped) =
  when defined(DEBUG):
    actions

proc getNloptAlgorithmTable*(): Table[string, nlopt_algorithm] =
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

proc newNloptOpt*(opt_name: string, bounds: tuple[l, u: float] = (-Inf, Inf)): NloptOpt =
  ## creator of a new NloptOpt object, which takes a string describing the algorithm to be
  ## used as well as (optionally) the lower and upper bounds to be used, as a tuple
  ## TODO: add options to also already add arguments for other fields of NloptOpt
  var
    opt: nlopt_opt
    status: nlopt_result
    f: nlopt_func
  status = NLOPT_SUCCESS

  let opt_name_table = getNloptAlgorithmTable()
  
  opt = nlopt_create(opt_name_table[opt_name], 1)

  # set bounds
  var (l_bound, u_bound) = bounds
  if l_bound != -Inf and u_bound != Inf:
    status = nlopt_set_lower_bounds(opt, addr(cdouble(l_bound)))
    status = nlopt_set_upper_bounds(opt, addr(cdouble(u_bound)))
    
  result = NloptOpt(optimizer: opt,
                    opt_name: opt_name,
                    l_bound: l_bound,
                    u_bound: u_bound,
                    xtol_rel: 0,
                    xtol_abs: 0,
                    ftol_rel: 0,
                    ftol_abs: 0,
                    maxtime: 0,
                    initial_step: 0,
                    status: status,
                    opt_func: f)

proc setFunction*(nlopt: var NloptOpt, f: NloptRawFunc, f_obj: var object) =
  # create a NLopt internal function object, which we use to
  # pass our high level object down to the NLopt library
  nlopt.opt_func = cast[nlopt_func](f)
  nlopt.status = nlopt_set_min_objective(nlopt.optimizer,
                                         nlopt.opt_func,
                                         cast[pointer](addr(f_obj)))

# define macro to simply create all procs to set optimizer settings. All receive same
# arguments, therefore can be done in one go
macro set_nlopt_floatvals(func_name: static[string]): typed =
  let nim_func_name: string = """
proc $#*(nlopt: var NloptOpt, val: float) = 
  nlopt.status = nlopt_$#(nlopt.optimizer, cdouble(val))""" % [func_name, func_name]
  result = parseStmt(nim_func_name)

set_nlopt_floatvals("set_xtol_abs1")
set_nlopt_floatvals("set_xtol_rel")
set_nlopt_floatvals("set_ftol_abs")
set_nlopt_floatvals("set_ftol_rel")
set_nlopt_floatvals("set_maxtime")
static:
  withDebug:
    echo getAst(set_nlopt_floatvals("set_ftol_rel")).repr

proc set_initial_step(nlopt: var NloptOpt, initial_step: float) =
  # simple wrapper for nlopt_set_initial_step, which takes care of
  # handing a ptr cdouble to the Nlopt function
  var dx: cdouble = cdouble(initial_step)
  nlopt.status = nlopt_set_initial_step(nlopt.optimizer, addr(dx))

template nlopt_write_or_raise(nlopt: var NloptOpt, f: untyped, field: untyped) =
  ## simple template, which checks whether a call to libnlopt should be done and
  ## if so, also checks whether the call was successful. Otherwise raises a
  ## LibraryError with a (hopefully) helpful error message
  if field != 0:
    #actions
    f(field)
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
    p = params
    f_p: cdouble = 0.1

  # use template to set nlopt stopping criteria in libnlopt and also perform
  # checks on successful calls to libnlopt, while keeping bloat away
  nlopt_write_or_raise(nlopt, nlopt.set_xtol_abs1, nlopt.xtol_abs)
  nlopt_write_or_raise(nlopt, nlopt.set_xtol_rel, nlopt.xtol_rel)
  nlopt_write_or_raise(nlopt, nlopt.set_ftol_abs, nlopt.ftol_abs)
  nlopt_write_or_raise(nlopt, nlopt.set_ftol_rel, nlopt.ftol_rel)
  nlopt_write_or_raise(nlopt, nlopt.set_maxtime, nlopt.maxtime)
  nlopt_write_or_raise(nlopt, nlopt.set_initial_step, nlopt.initial_step)

  # now perform optimization
  nlopt.status = nlopt_optimize(nlopt.optimizer, addr(p[0]), addr(f_p))
  if nlopt.status < NLOPT_SUCCESS:
    echo "Warning: nlopt optimization failed with status $#!" % $nlopt.status
  
  result = (p: p, f: f_p)
  
