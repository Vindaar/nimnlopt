##  Copyright (c) 2007-2014 Massachusetts Institute of Technology
## 
##  Permission is hereby granted, free of charge, to any person obtaining
##  a copy of this software and associated documentation files (the
##  "Software"), to deal in the Software without restriction, including
##  without limitation the rights to use, copy, modify, merge, publish,
##  distribute, sublicense, and/or sell copies of the Software, and to
##  permit persons to whom the Software is furnished to do so, subject to
##  the following conditions:
##  
##  The above copyright notice and this permission notice shall be
##  included in all copies or substantial portions of the Software.
##  
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
##  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
##  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
##  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
##  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
## 

{.deadCodeElim: on.}
const
  libname* = "libnlopt.so"

##  for now comment out NLOPT macros, since they are only needed for windows
##  Change 0 to 1 to use stdcall convention under Win32
##  #if 0 && (defined(_WIN32) || defined(__WIN32__))
##  #  if defined(__GNUC__)
##  #    define NLOPT_STDCALL __attribute__((stdcall))
##  #  elif defined(_MSC_VER) || defined(_ICC) || defined(_STDCALL_SUPPORTED)
##  #    define NLOPT_STDCALL __stdcall
##  #  else
##  #    define NLOPT_STDCALL
##  #  endif
##  #else
##  #  define NLOPT_STDCALL
##  #endif
##  /\* for Windows compilers, you should add a line
##             #define NLOPT_DLL
##     when using NLopt from a DLL, in order to do the proper
##     Windows importing nonsense. *\/
##  #if defined(NLOPT_DLL) && (defined(_WIN32) || defined(__WIN32__)) && !defined(__LCC__)
##  /\* annoying Windows syntax for calling functions in a DLL *\/
##  #  if defined(NLOPT_DLL_EXPORT)
##  #    define NLOPT_EXTERN(T) extern __declspec(dllexport) T NLOPT_STDCALL
##  #  else
##  #    define NLOPT_EXTERN(T) extern __declspec(dllimport) T NLOPT_STDCALL
##  #  endif
##  #else
##  #  define NLOPT_EXTERN(T) extern T NLOPT_STDCALL
##  #endif

type
  nlopt_func* = proc (n: cuint; x: ptr cdouble; gradient: ptr cdouble; func_data: pointer): cdouble {.cdecl.}                 ##  NULL if not needed
  nlopt_mfunc* = proc (m: cuint; result: ptr cdouble; n: cuint; x: ptr cdouble;
                    gradient: ptr cdouble; func_data: pointer) {.cdecl.} ##  NULL if not needed

##  A preconditioner, which preconditions v at x to return vpre. 
##    (The meaning of "preconditioning" is algorithm-dependent.)

type
  ##  Naming conventions:
  ## 
  ##         NLOPT_{G/L}{D/N}_* 
  ## 	             = global/local derivative/no-derivative optimization, 
  ##               respectively 
  ##  
  ## _RAND algorithms involve some randomization.
  ## 
  ## _NOSCAL algorithms are *not* scaled to a unit hypercube
  ## 	         (i.e. they are sensitive to the units of x)
  ## 
  nlopt_precond* = proc (n: cuint; x: ptr cdouble; v: ptr cdouble; vpre: ptr cdouble; data: pointer) {.cdecl.}
  nlopt_algorithm* {.size: sizeof(cint).} = enum
    NLOPT_GN_DIRECT = 0, NLOPT_GN_DIRECT_L, NLOPT_GN_DIRECT_L_RAND,
    NLOPT_GN_DIRECT_NOSCAL, NLOPT_GN_DIRECT_L_NOSCAL,
    NLOPT_GN_DIRECT_L_RAND_NOSCAL, NLOPT_GN_ORIG_DIRECT, NLOPT_GN_ORIG_DIRECT_L,
    NLOPT_GD_STOGO, NLOPT_GD_STOGO_RAND, NLOPT_LD_LBFGS_NOCEDAL, NLOPT_LD_LBFGS,
    NLOPT_LN_PRAXIS, NLOPT_LD_VAR1, NLOPT_LD_VAR2, NLOPT_LD_TNEWTON,
    NLOPT_LD_TNEWTON_RESTART, NLOPT_LD_TNEWTON_PRECOND,
    NLOPT_LD_TNEWTON_PRECOND_RESTART, NLOPT_GN_CRS2_LM, NLOPT_GN_MLSL,
    NLOPT_GD_MLSL, NLOPT_GN_MLSL_LDS, NLOPT_GD_MLSL_LDS, NLOPT_LD_MMA,
    NLOPT_LN_COBYLA, NLOPT_LN_NEWUOA, NLOPT_LN_NEWUOA_BOUND, NLOPT_LN_NELDERMEAD,
    NLOPT_LN_SBPLX, NLOPT_LN_AUGLAG, NLOPT_LD_AUGLAG, NLOPT_LN_AUGLAG_EQ,
    NLOPT_LD_AUGLAG_EQ, NLOPT_LN_BOBYQA, NLOPT_GN_ISRES, ##  new variants that require local_optimizer to be set,
                                                      ## 	not with older constants for backwards compatibility
    NLOPT_AUGLAG, NLOPT_AUGLAG_EQ, NLOPT_G_MLSL, NLOPT_G_MLSL_LDS, NLOPT_LD_SLSQP,
    NLOPT_LD_CCSAQ, NLOPT_GN_ESCH, NLOPT_NUM_ALGORITHMS ##  not an algorithm, just the number of them


proc nlopt_algorithm_name*(a: nlopt_algorithm): cstring {.cdecl,
    importc: "nlopt_algorithm_name", dynlib: libname.}
type
  nlopt_result* {.size: sizeof(cint).} = enum
    NLOPT_FORCED_STOP = - 5, NLOPT_ROUNDOFF_LIMITED = - 4, NLOPT_OUT_OF_MEMORY = - 3,
    NLOPT_INVALID_ARGS = - 2, NLOPT_FAILURE = - 1, ##  generic failure code
    NLOPT_SUCCESS = 1,          ##  generic success code
    NLOPT_STOPVAL_REACHED = 2, NLOPT_FTOL_REACHED = 3, NLOPT_XTOL_REACHED = 4,
    NLOPT_MAXEVAL_REACHED = 5, NLOPT_MAXTIME_REACHED = 6


const
  NLOPT_MINF_MAX_REACHED* = NLOPT_STOPVAL_REACHED

proc nlopt_srand*(seed: culong) {.cdecl, importc: "nlopt_srand", dynlib: libname.}
proc nlopt_srand_time*() {.cdecl, importc: "nlopt_srand_time", dynlib: libname.}
proc nlopt_version*(major: ptr cint; minor: ptr cint; bugfix: ptr cint) {.cdecl,
    importc: "nlopt_version", dynlib: libname.}
## ************************** OBJECT-ORIENTED API *************************
##  The style here is that we create an nlopt_opt "object" (an opaque pointer),
##    then set various optimization parameters, and then execute the
##    algorithm.  In this way, we can add more and more optimization parameters
##    (including algorithm-specific ones) without breaking backwards
##    compatibility, having functions with zillions of parameters, or
##    relying non-reentrantly on global variables.

type
  nlopt_opt_s* = object
  

##  opaque structure, defined internally

type
  nlopt_opt* = ptr nlopt_opt_s

##  the only immutable parameters of an optimization are the algorithm and
##    the dimension n of the problem, since changing either of these could
##    have side-effects on lots of other parameters

proc nlopt_create*(algorithm: nlopt_algorithm; n: cuint): nlopt_opt {.cdecl,
    importc: "nlopt_create", dynlib: libname.}
proc nlopt_destroy*(opt: nlopt_opt) {.cdecl, importc: "nlopt_destroy", dynlib: libname.}
proc nlopt_copy*(opt: nlopt_opt): nlopt_opt {.cdecl, importc: "nlopt_copy",
    dynlib: libname.}
proc nlopt_optimize*(opt: nlopt_opt; x: ptr cdouble; opt_f: ptr cdouble): nlopt_result {.
    cdecl, importc: "nlopt_optimize", dynlib: libname.}
proc nlopt_set_min_objective*(opt: nlopt_opt; f: nlopt_func; f_data: pointer): nlopt_result {.
    cdecl, importc: "nlopt_set_min_objective", dynlib: libname.}
proc nlopt_set_max_objective*(opt: nlopt_opt; f: nlopt_func; f_data: pointer): nlopt_result {.
    cdecl, importc: "nlopt_set_max_objective", dynlib: libname.}
proc nlopt_set_precond_min_objective*(opt: nlopt_opt; f: nlopt_func;
                                     pre: nlopt_precond; f_data: pointer): nlopt_result {.
    cdecl, importc: "nlopt_set_precond_min_objective", dynlib: libname.}
proc nlopt_set_precond_max_objective*(opt: nlopt_opt; f: nlopt_func;
                                     pre: nlopt_precond; f_data: pointer): nlopt_result {.
    cdecl, importc: "nlopt_set_precond_max_objective", dynlib: libname.}
proc nlopt_get_algorithm*(opt: nlopt_opt): nlopt_algorithm {.cdecl,
    importc: "nlopt_get_algorithm", dynlib: libname.}
proc nlopt_get_dimension*(opt: nlopt_opt): cuint {.cdecl,
    importc: "nlopt_get_dimension", dynlib: libname.}
proc nlopt_get_errmsg*(opt: nlopt_opt): cstring {.cdecl, importc: "nlopt_get_errmsg",
    dynlib: libname.}
##  constraints:

proc nlopt_set_lower_bounds*(opt: nlopt_opt; lb: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_lower_bounds", dynlib: libname.}
proc nlopt_set_lower_bounds1*(opt: nlopt_opt; lb: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_lower_bounds1", dynlib: libname.}
proc nlopt_get_lower_bounds*(opt: nlopt_opt; lb: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_get_lower_bounds", dynlib: libname.}
proc nlopt_set_upper_bounds*(opt: nlopt_opt; ub: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_upper_bounds", dynlib: libname.}
proc nlopt_set_upper_bounds1*(opt: nlopt_opt; ub: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_upper_bounds1", dynlib: libname.}
proc nlopt_get_upper_bounds*(opt: nlopt_opt; ub: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_get_upper_bounds", dynlib: libname.}
proc nlopt_remove_inequality_constraints*(opt: nlopt_opt): nlopt_result {.cdecl,
    importc: "nlopt_remove_inequality_constraints", dynlib: libname.}
proc nlopt_add_inequality_constraint*(opt: nlopt_opt; fc: nlopt_func;
                                     fc_data: pointer; tol: cdouble): nlopt_result {.
    cdecl, importc: "nlopt_add_inequality_constraint", dynlib: libname.}
proc nlopt_add_precond_inequality_constraint*(opt: nlopt_opt; fc: nlopt_func;
    pre: nlopt_precond; fc_data: pointer; tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_add_precond_inequality_constraint", dynlib: libname.}
proc nlopt_add_inequality_mconstraint*(opt: nlopt_opt; m: cuint; fc: nlopt_mfunc;
                                      fc_data: pointer; tol: ptr cdouble): nlopt_result {.
    cdecl, importc: "nlopt_add_inequality_mconstraint", dynlib: libname.}
proc nlopt_remove_equality_constraints*(opt: nlopt_opt): nlopt_result {.cdecl,
    importc: "nlopt_remove_equality_constraints", dynlib: libname.}
proc nlopt_add_equality_constraint*(opt: nlopt_opt; h: nlopt_func; h_data: pointer;
                                   tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_add_equality_constraint", dynlib: libname.}
proc nlopt_add_precond_equality_constraint*(opt: nlopt_opt; h: nlopt_func;
    pre: nlopt_precond; h_data: pointer; tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_add_precond_equality_constraint", dynlib: libname.}
proc nlopt_add_equality_mconstraint*(opt: nlopt_opt; m: cuint; h: nlopt_mfunc;
                                    h_data: pointer; tol: ptr cdouble): nlopt_result {.
    cdecl, importc: "nlopt_add_equality_mconstraint", dynlib: libname.}
##  stopping criteria:

proc nlopt_set_stopval*(opt: nlopt_opt; stopval: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_stopval", dynlib: libname.}
proc nlopt_get_stopval*(opt: nlopt_opt): cdouble {.cdecl,
    importc: "nlopt_get_stopval", dynlib: libname.}
proc nlopt_set_ftol_rel*(opt: nlopt_opt; tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_ftol_rel", dynlib: libname.}
proc nlopt_get_ftol_rel*(opt: nlopt_opt): cdouble {.cdecl,
    importc: "nlopt_get_ftol_rel", dynlib: libname.}
proc nlopt_set_ftol_abs*(opt: nlopt_opt; tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_ftol_abs", dynlib: libname.}
proc nlopt_get_ftol_abs*(opt: nlopt_opt): cdouble {.cdecl,
    importc: "nlopt_get_ftol_abs", dynlib: libname.}
proc nlopt_set_xtol_rel*(opt: nlopt_opt; tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_xtol_rel", dynlib: libname.}
proc nlopt_get_xtol_rel*(opt: nlopt_opt): cdouble {.cdecl,
    importc: "nlopt_get_xtol_rel", dynlib: libname.}
proc nlopt_set_xtol_abs1*(opt: nlopt_opt; tol: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_xtol_abs1", dynlib: libname.}
proc nlopt_set_xtol_abs*(opt: nlopt_opt; tol: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_xtol_abs", dynlib: libname.}
proc nlopt_get_xtol_abs*(opt: nlopt_opt; tol: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_get_xtol_abs", dynlib: libname.}
proc nlopt_set_maxeval*(opt: nlopt_opt; maxeval: cint): nlopt_result {.cdecl,
    importc: "nlopt_set_maxeval", dynlib: libname.}
proc nlopt_get_maxeval*(opt: nlopt_opt): cint {.cdecl, importc: "nlopt_get_maxeval",
    dynlib: libname.}
proc nlopt_get_numevals*(opt: nlopt_opt): cint {.cdecl,
    importc: "nlopt_get_numevals", dynlib: libname.}
proc nlopt_set_maxtime*(opt: nlopt_opt; maxtime: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_maxtime", dynlib: libname.}
proc nlopt_get_maxtime*(opt: nlopt_opt): cdouble {.cdecl,
    importc: "nlopt_get_maxtime", dynlib: libname.}
proc nlopt_force_stop*(opt: nlopt_opt): nlopt_result {.cdecl,
    importc: "nlopt_force_stop", dynlib: libname.}
proc nlopt_set_force_stop*(opt: nlopt_opt; val: cint): nlopt_result {.cdecl,
    importc: "nlopt_set_force_stop", dynlib: libname.}
proc nlopt_get_force_stop*(opt: nlopt_opt): cint {.cdecl,
    importc: "nlopt_get_force_stop", dynlib: libname.}
##  more algorithm-specific parameters

proc nlopt_set_local_optimizer*(opt: nlopt_opt; local_opt: nlopt_opt): nlopt_result {.
    cdecl, importc: "nlopt_set_local_optimizer", dynlib: libname.}
proc nlopt_set_population*(opt: nlopt_opt; pop: cuint): nlopt_result {.cdecl,
    importc: "nlopt_set_population", dynlib: libname.}
proc nlopt_get_population*(opt: nlopt_opt): cuint {.cdecl,
    importc: "nlopt_get_population", dynlib: libname.}
proc nlopt_set_vector_storage*(opt: nlopt_opt; dim: cuint): nlopt_result {.cdecl,
    importc: "nlopt_set_vector_storage", dynlib: libname.}
proc nlopt_get_vector_storage*(opt: nlopt_opt): cuint {.cdecl,
    importc: "nlopt_get_vector_storage", dynlib: libname.}
proc nlopt_set_default_initial_step*(opt: nlopt_opt; x: ptr cdouble): nlopt_result {.
    cdecl, importc: "nlopt_set_default_initial_step", dynlib: libname.}
proc nlopt_set_initial_step*(opt: nlopt_opt; dx: ptr cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_initial_step", dynlib: libname.}
proc nlopt_set_initial_step1*(opt: nlopt_opt; dx: cdouble): nlopt_result {.cdecl,
    importc: "nlopt_set_initial_step1", dynlib: libname.}
proc nlopt_get_initial_step*(opt: nlopt_opt; x: ptr cdouble; dx: ptr cdouble): nlopt_result {.
    cdecl, importc: "nlopt_get_initial_step", dynlib: libname.}
##  the following are functions mainly designed to be used internally
##    by the Fortran and SWIG wrappers, allow us to tel nlopt_destroy and
##    nlopt_copy to do something to the f_data pointers (e.g. free or
##    duplicate them, respectively)

type
  nlopt_munge* = proc (p: pointer): pointer {.cdecl.}

proc nlopt_set_munge*(opt: nlopt_opt; munge_on_destroy: nlopt_munge;
                     munge_on_copy: nlopt_munge) {.cdecl,
    importc: "nlopt_set_munge", dynlib: libname.}
type
  nlopt_munge2* = proc (p: pointer; data: pointer): pointer {.cdecl.}

proc nlopt_munge_data*(opt: nlopt_opt; munge: nlopt_munge2; data: pointer) {.cdecl,
    importc: "nlopt_munge_data", dynlib: libname.}
##  The deprecated API is not supported. No need for it.
## ************************** DEPRECATED API *************************
##  The new "object-oriented" API is preferred, since it allows us to
##    gracefully add new features and algorithm-specific options in a
##    re-entrant way, and we can automatically assume reasonable defaults
##    for unspecified parameters.
##  Where possible (e.g. for gcc >= 3.1), enable a compiler warning
##    for code that uses a deprecated function
##  #if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__==3 && __GNUC_MINOR__ > 0))
##  #  define NLOPT_DEPRECATED __attribute__((deprecated))
##  #else
##  #  define NLOPT_DEPRECATED
##  #endif
##  typedef double (*nlopt_func_old)(int n, const double *x,
##  				 double *gradient, /\* NULL if not needed *\/
##  				 void *func_data);
##  nlopt_result nlopt_minimize(
##       nlopt_algorithm algorithm,
##       int n, nlopt_func_old f, void *f_data,
##       const double *lb, const double *ub, /\* bounds *\/
##       double *x, /\* in: initial guess, out: minimizer *\/
##       double *minf, /\* out: minimum *\/
##       double minf_max, double ftol_rel, double ftol_abs,
##       double xtol_rel, const double *xtol_abs,
##       int maxeval, double maxtime) NLOPT_DEPRECATED;
##  nlopt_result nlopt_minimize_constrained(
##       nlopt_algorithm algorithm,
##       int n, nlopt_func_old f, void *f_data,
##       int m, nlopt_func_old fc, void *fc_data, ptrdiff_t fc_datum_size,
##       const double *lb, const double *ub, /\* bounds *\/
##       double *x, /\* in: initial guess, out: minimizer *\/
##       double *minf, /\* out: minimum *\/
##       double minf_max, double ftol_rel, double ftol_abs,
##       double xtol_rel, const double *xtol_abs,
##       int maxeval, double maxtime) NLOPT_DEPRECATED;
##  nlopt_result nlopt_minimize_econstrained(
##       nlopt_algorithm algorithm,
##       int n, nlopt_func_old f, void *f_data,
##       int m, nlopt_func_old fc, void *fc_data, ptrdiff_t fc_datum_size,
##       int p, nlopt_func_old h, void *h_data, ptrdiff_t h_datum_size,
##       const double *lb, const double *ub, /\* bounds *\/
##       double *x, /\* in: initial guess, out: minimizer *\/
##       double *minf, /\* out: minimum *\/
##       double minf_max, double ftol_rel, double ftol_abs,
##       double xtol_rel, const double *xtol_abs,
##       double htol_rel, double htol_abs,
##       int maxeval, double maxtime) NLOPT_DEPRECATED;
##  void nlopt_get_local_search_algorithm(nlopt_algorithm *deriv,
##  					     nlopt_algorithm *nonderiv,
##  					     int *maxeval) NLOPT_DEPRECATED;
##  void nlopt_set_local_search_algorithm(nlopt_algorithm deriv,
##  					     nlopt_algorithm nonderiv,
##  					     int maxeval) NLOPT_DEPRECATED;
##  int nlopt_get_stochastic_population(void) NLOPT_DEPRECATED;
##  void nlopt_set_stochastic_population(int pop) NLOPT_DEPRECATED;
## *******************************************************************
