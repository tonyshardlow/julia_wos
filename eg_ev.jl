# call add_procs(n) to add extra processors if desired
# load packages
using KOSHELPER
@everywhere using KOS
using KOS_FIELD_SOLVE
#
# other Julia packages
@everywhere using Interpolations
using PyPlot
#
ion()
# distance function
@everywhere function d_ball_0_1( x::Array{Float64,1} )
  # out=d_ball_0_1(x)
  #
  # Distance of points to the domain boundary
  # Domain is a ball of radius 1 centred at (0,0)
  # x: point
  R=1.
  return R-sqrt(sum(x.^2))
end
#
@everywhere function vec_d_ball_0_1( xs )
  # out=d_ball_0_1(x)
  # same as above
  # points are rows of xs
  R=1.
  return (R-sqrt.(sum(xs.^2,length(size(xs)))))[:]
end
# Zero functions
@everywhere function fn_zero( x::Array{Float64,2})
  # 
  return 0.
end
#
@everywhere function fn_zero( x::Array{Float64,1} )
  # Right-hand side designed to expose slow coupling rates.  #
  # x: point(s) as columns for evaluation
  return zeros(size(x)[1])
end
##########################################
function mwe2(alpha)
  # Main routine for computing eigenvalues on unit ball
  # Inputs: alpha, parameter for fractional Laplacian in (0,2)
  # lam_tol, tolerance for the eigenvalue
  lam_tol=5e-2
  # no_its, number of Arnoldi iterations
  no_its=5
  # relax_flag, use variable accuracy if True
  relax_flag=true
  #
  # Ouputs:lam, computed eignvalue
  # el_time, elapsed time in seconds
  # eig_res, eigenvalue residual
  println("\n==================\nGet smallest eigenvalue.\n")
  #
  # domain is set here (fn_zero functions don't do much)
  p=KOS.Params(fn_zero,fn_zero,vec_d_ball_0_1,alpha)
  #
  # set the parameters for the method
  dict=Dict("max_samples"=>Int32(1e5), "report"=>true, 
    #
    "no_levels"=>7, "l0"=>5,  "solve_method"=>"ml",
    "lam_tol"=>lam_tol, # tolerance for eigenvalue solve
    "no_arnoldi_its"=>no_its,
    #
    "relax"=>relax_flag, # for inexact Arnolid
    "B"=>3,#see paper for B (large B smaller WOS tolerance)
   )
  println("Variable accuracy=", dict["relax"], 
          ", no_levels=",dict["no_levels"], ", lam_tol=",dict["lam_tol"])
  # set up the mesh
  G=KOSHELPER.get_tri_mesh([-1.,1.,-1.,1.],
                            dict["no_levels"],p.d)
  # first Arnoldi run
  # initial vector for eigenvector
  Y=ones(G.n_int);
  for i=1:G.n_int
    Y[i]=1-(1-p.d(G.get_pt(i))[1])^2
  end 
  arnoldi=Construct_Arnoldi(Y)
  #########################################
  lam,evec,delta_lam,arnoldi,el_time,eig_res=wos_ev(p,dict,G,arnoldi,1.)
  dyda_compare(alpha,  lam)
  println("-----------------------------------\n\n")
  #
  println("Variable accuracy=",dict["relax"], 
          ", no_levels=",dict["no_levels"], ", lam_tol=",dict["lam_tol"])
  out=G.reshape(evec,KOS.vec_fn_zero)
  Z=G.move_to_TriMesh(out)
  #
  do_plot(G,Z,"evec", "evec.png")
  return lam, el_time, eig_res
end
#######################################
################ end eg_ev.jl
####################################