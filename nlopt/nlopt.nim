import nlopt_wrapper

# this file provides the high level functionality of the NLopt nim library,
# especially containing the type conversion to compatible types and dealing
# with addresses and pointer

type
  NloptOpt = object
    optimizer: nlopt_opt
    opt_name: string
    l_bound: float
    u_bound: float
    xtol_rel: float
    xtol_abs: float
    ftol_rel: float
    ftol_abs: float
    maxtime: float
    initial_step: float
    status: nlopt_result
    func: nlopt_func
          
# * = proc (n: cuint; x: ptr cdouble; gradient: ptr cdouble; func_data: pointer): cdouble {.cdecl.}          

proc newNlopOpt(opt_name: string, bounds: tuple[l, u: float]): NloptOpt =
  opt_name_table = { "GN_DIRECT" : NLOPT_GN_DIRECT,
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

  var opt: nlopt_opt
  opt = nlopt_create(opt_name_table[opt_name], 1)
  let (l_bound, u_bound) = bounds
  var status: nlopt_result
  result = NloptOpt(opt, opt_name, l_bound, u_bound, 0, 0, 0, 0, 0, 0, status, nil)

proc setFunction(nlopt: NloptOpt,   
