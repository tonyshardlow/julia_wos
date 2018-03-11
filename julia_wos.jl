# Main driver routine for WOS fractional Laplacian solver
#
# Type
#
# julia> include("julia_wos.jl"); run_field(0.2,1,true)
#
# to solve with alpha=0.2, with parameters from Example 3.11
# from T. Shardlow "A walk outside spheres for the fractional Laplacian: fields and
# first eigenvalue"
# 
# Take advantage of extra cores by initialising with add_procs(n).
#
# You need these packages on your LOAD_PATH (KOS refers to the original
# authors of the algorithm: Kyprianou, Osojnik, Shardlow)
using KOSHELPER
@everywhere using KOS
using KOS_FIELD_SOLVE
#
# other Julia packages
@everywhere using Interpolations
using PyPlot
#
ion()
##########################################
##########################################
# begin: end user routines
function run_field(alpha,egnum,plot_flag)
  # Field solve for built-in examples (egnum)
  # If plot_flag=true, fig_err*.pdf and fig_soln*.pdf files are generated
  # for solution and error relative to exact soln (if available).
  #
  if 1==egnum # Example 3.11 (from the paper)
    p,exact=KOS.example1(alpha) # constant rhs, zero exterior
    myext="_1";l0=5
    R=1.; tol=1e-2
  elseif 2==egnum # Example 3.12
    p,exact=KOS.example2(alpha) # quadratic rhs, zero exterior
    myext="_2"; l0=5
    R=1.; tol=1e-2
  elseif 3==egnum
    p,exact=KOS.example3(alpha) # Green's fn exterior on unit ball, zero rhs
    myext="_3"
    R=1.2; l0=7 
    tol=1e-3
  elseif 4==egnum
    p,exact=KOS.example4(alpha) # Green's fn exterior on square, zero rhs
    myext="_3"
    R=1.2; l0=7 
    tol=1e-2
  elseif 5==egnum
    p,exact=KOS.example5(alpha) # Example 3.13 (no exact available)
    myext="_5"
    R=1.2; l0=7 
    tol=1e-2
  end
  dict=Dict("max_samples"=>Int32(1e6),
  "tol"=>tol,
  "no_samps"=>4000,
  "no_levels"=>7,"l0"=>l0,
  "report"=>true, 
  "solve_method"=>"ml")
  G=KOSHELPER.get_tri_mesh([-R,R,-R,R],
                            dict["no_levels"],p.d)
  #
  out,elapsed_time=get_solution(p, dict, G)  
  G.get_ext(out,p.fn_ext)
  #
  Z=G.move_to_TriMesh(out)
  #
  exact_Z=G.evalfn(exact, p.fn_ext)
  # two ways of computing L2 error
  err=sqrt(G.sq_int(out-exact_Z,dict["no_levels"]))
  err2=sqrt(G.sq_int2(out,exact,dict["no_levels"]))
  # L2 norm of exact solution
  norm_exact=sqrt(G.sq_int(exact_Z,dict["no_levels"]))
  println("Error norm=",err, " err2=",err2," elapsed time=", elapsed_time,".")
  #
  Ze=G.move_to_TriMesh(exact_Z)
  if plot_flag
    do_plot(G,Z,"",string("fig_soln",myext,".pdf") )
    do_plot(G,abs.(Z-Ze),"",string("fig_err",myext,".pdf") )
  end
  println("Elapsed time=", elapsed_time) 
  return elapsed_time, err2, (err2/norm_exact)
end
#################
function run_stats(alpha)
  # Analyse MLMC parameter
  # Find computation times for different initial level
  # from l0 upto no_levels
  #
  #p,exact=KOS.example3(alpha)
  p,exact=KOS.example2(alpha)
  dict=Dict("max_samples"=>Int32(1e5),
  "tol"=>1e-2,
  "no_samps"=>4000,
  "no_levels"=>9,"l0"=>6,
  "report"=>true, 
  "solve_method"=>"ml_analyse")
  G=KOSHELPER.get_tri_mesh([-1.,1.,-1.,1.],
                            dict["no_levels"],p.d)
  #
  out,elapsed_time=get_solution(p, dict, G)  
  #
end
####################
function run_couple(vec_alpha)
  # Compute stats on the coupling (variance and time per sample)
  #
  #p,exact=KOS.example3(1.0)
  p,exact=KOS.example5(1.0)
  dict=Dict("max_samples"=>Int32(1e5),
  "tol"=>1e-2,
  "no_levels"=>7,"l0"=>3,
  "report"=>true, 
  "no_samps"=>40000,
  "solve_method"=>"couple_analyse")
  G=KOSHELPER.get_tri_mesh([-1.,1.,-1.,1.],
                            dict["no_levels"],p.d)
  for alpha in vec_alpha
    p.alpha=alpha
    p,exact=KOS.example5(alpha)
    out,elapsed_time=get_solution(p, dict, G)  
  end
end
##################
function run_ev(alpha, lam_tol, no_its,relax_flag,no_restarts,p_restart)
  # Main routine for computing eigenvalues on the unit ball.
  # Inputs: alpha, parameter for fractional Laplacian in (0,2).
  # lam_tol: tolerance for the eigenvalue.
  # no_its: number of Arnoldi iterations.
  # relax_flag: use variable accuracy if true.
  #
  # experimental (set no_restarts=0 for algorithm in paper)
  # no_restarts: no of iteratively restarted its
  # p_restart:Krylov dimension for restart.
  #
  # Ouputs: lam: computed eigenvalue.
  # el_time: elapsed time in seconds.
  # eig_res: eigenvalue residual.
  println("\n==================\nGet smallest eigenvalue.\n")
  #
  # domain is set here via choice of distance function.
  p,exact=KOS.example0(alpha)
  #
  # set the parameters for the method
  dict=Dict("max_samples"=>Int32(1e5), "report"=>true, 
    #
    "no_levels"=>7, "l0"=>5, # number of levels 
    "solve_method"=>"ml", # don't change this
    #
    "lam_tol"=>lam_tol, # tolerance for eigenvalue solve
    "no_arnoldi_its"=>no_its,
    #
    "relax"=>relax_flag, # for inexact Arnolid
    "B"=>3,#see paper for B (large B smaller WOS tolerance)
    # experimental
    "no_restarts"=>no_restarts, # for implicit restarts
    "p_restart"=>p_restart
   )

  println("Using no_restarts=",dict["no_restarts"],", restart dim=", p_restart, 
          ",  variable accuracy=", dict["relax"], 
          ", no_levels=",dict["no_levels"], ", lam_tol=",dict["lam_tol"])
  #
  # set up the TriMesh
  G=KOSHELPER.get_tri_mesh([-1.,1.,-1.,1.],
                            dict["no_levels"],p.d)
  # first Arnoldi run
  # initial vector for eigenvector
  Y=ones(G.n_int);
  for i=1:G.n_int
    Y[i]=1-(1-p.d(G.get_pt(i))[1])^2
  end 
  arnoldi=Construct_Arnoldi(Y)
  ##########################################
  i=0
  println(">>>>First Arnoldi run")
  lam,evec,delta_lam,arnoldi,el_time,eig_res=wos_ev(p,dict,G,arnoldi,1.)
  dyda_compare(alpha,  lam)
  println("-----------------------------------\n\n")
  #
  delta_lam=1.0
  while (i<dict["no_restarts"])
    i+=1
    KOSHELPER.SmartRestart!(arnoldi, dict["p_restart"])
    println("     >>>>>\n     >>>>Next Arnoldi run number=", i+1)
    #
    lam,evec,delta_lam,arnoldi,el_time,eig_res=wos_ev(p,dict,G,arnoldi,lam)
    #
    dyda_compare(alpha,  lam)
    println("-----------------------------------\n\n")
  end
  
  println("Using no_restarts=",dict["no_restarts"],", restart dim=",p_restart,
          ", variable accuracy=",dict["relax"], 
          ", no_levels=",dict["no_levels"], ", lam_tol=",dict["lam_tol"])
  out=G.reshape(evec,KOS.vec_fn_zero)
  Z=G.move_to_TriMesh(out)
  #
  do_plot(G,Z,"evec", "evec.png")
  return lam, el_time, eig_res
end
#######################################
#######################################
function batch_ev()
  # This script runs the eigenvalue solver for a range of alpha
  #
  lam_tol=1e-2; no_its=5; relax_flag=true; no_restarts=0; p_restart=3;
  valpha=[0.1,0.2,0.5,1.0,1.5,1.8,1.9]
   valpha=[0.5,1.0,1.5]
  n=length(valpha); vl=zeros(n); vel_time=zeros(n); veig_res=zeros(n)
  for i in 1:n
    l,et,er=run_ev(valpha[i],lam_tol,no_its,relax_flag,no_restarts,p_restart)
    vl[i]=l; vel_time[i]=et; veig_res[i]=er;
  end
  println("Vector of computed lambda=", vl)
  println("Vector of elasped times=", vel_time)
  println("Vector of eigenvalue residuals=",veig_res)
end
############################################
#######################################
function batch_ev2()
  # This script runs the eigenvalue solver for a range of alpha
  #
  lam_tol=1e-2; no_its=5; relax_flag=true; no_restarts=0; p_restart=3;
  valpha=[0.1,0.2,0.5,1.0,1.5,1.8,1.9]
   valpha=[0.5,1.0,1.5]
 #  valpha=[0.25, 0.75, 1.25, 1.75]
  n=length(valpha); vl=zeros(n); vel_time=zeros(n); veig_res=zeros(n)
  for i in 1:n
    l,et,er=run_ev(valpha[i],lam_tol,no_its,relax_flag,no_restarts,p_restart)
    vl[i]=l; vel_time[i]=et; veig_res[i]=er;
  end
  println("Vector of computed lambda=", vl)
  println("Vector of elasped times=", vel_time)
  println("Vector of eigenvalue residuals=",veig_res)
  #######################################
  lam_tol=1e-2; no_its=5; relax_flag=false; no_restarts=0; p_restart=3;
  valpha=[0.1,0.2,0.5,1.0,1.5,1.8,1.9]
   valpha=[0.5,1.0,1.5]
  n=length(valpha); vl2=zeros(n); vel_time2=zeros(n); veig_res2=zeros(n)
  for i in 1:n
    l,et,er=run_ev(valpha[i],lam_tol,no_its,relax_flag,no_restarts,p_restart)
    vl2[i]=l; vel_time2[i]=et; veig_res2[i]=er;
  end
  println("alpha=",valpha)
  println("relax_flag=true")
  println("Vector of computed lambda=", vl)
  println("Vector of elasped times=", vel_time)
  println("Vector of eigenvalue residuals=",veig_res)
  println("relax_flag=false")
  println("Vector of computed lambda=", vl2)
  println("Vector of elasped times=", vel_time2)
  println("Vector of eigenvalue residuals=",veig_res2)
end
#######################################
function batch_field()
  # This script runs the field solver for the following set of alpha
  #
  valpha=[0.1,0.2,0.5,1.0,1.5,1.8,1.9]
  #
  n=length(valpha); vrel_err=zeros(n); vel_time=zeros(n); verr=zeros(n)
  for i in 1:n
    et,er,err_rel=run_field(valpha[i],1,false)
    vel_time[i]=et; verr[i]=er; vrel_err[i]=err_rel
  end
  println("Vector of elasped times=", vel_time)
  println("Vector of errors=",verr)
  println("Vector of rel errors=",vrel_err)
end
############################################
############################################
function run_pt(alpha,x0)
  # This script runs WOS for one point inside the domain
  # p,exact=example2(alpha)
  p,exact=KOS.example2(alpha)
  #
  dict=Dict("max_samples"=>Int32(1e5),
  "no_per_batch"=>16,
  "tol"=>1e-3)
  #
  my_step,my_rhs_ball,dist_fn,ext_fn=setup_run_wos(p,dict)
  #
  mean_val, var_sample, err_est, no_samples = fn_run_wos(x0, 
    p, dict, my_step, my_rhs_ball, dist_fn, ext_fn)
  #
  println(mean_val," variance=",  var_sample," error=",
   err_est, " no_samples=",no_samples)
  print("exact=",exact(x0), "; error=",exact(x0)-sum(mean_val))
end
####################################
################ end julia_wos.jl
####################################