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
# exterior-data function g (in vector and scalar forms)
@everywhere function fn_ext( x::Array{Float64,1})
   # Exterior values
   tmp=sin((sum(x.^2)))
   return tmp::Float64
end
#
@everywhere function fn_ext( x::Array{Float64,2})
  tmp=sin.((sum(x.^2, 2)))
  return tmp[:]::Array{Float64,1}
end
#
# right-hand side function f (in vector and scalar forms)
@everywhere function fn_rhs( x::Array{Float64,1} )
  # Right-hand side designed to expose slow coupling rates.  #
  # x: point(s) as columns for evaluation
  tmp=2+sum(x.^2)
  return tmp::Float64
end
#
@everywhere function fn_rhs( x::Array{Float64,2} )
  tmp=2+sum(x.^2, 2)
  return tmp[:]::Array{Float64,1}
end
##########################################
##########################################
function mwe(alpha)
# set up example
p=KOS.Params(fn_ext,fn_rhs,vec_d_ball_0_1,alpha)
# set up params  
dict=Dict("max_samples"=>Int32(1e5),
  "tol"=>1e-2,
  "no_samps"=>4000,
  "no_levels"=>7,"l0"=>5,
  "report"=>true, 
  "solve_method"=>"ml")
# set up grid
R=1.2
G=KOSHELPER.get_tri_mesh([-R,R,-R,R],
                            dict["no_levels"],p.d)
# get the solution
 out,elapsed_time=get_solution(p, dict, G)
# make sure boundary conditions are applied to out
G.get_ext(out,p.fn_ext)
# put solution on a data structure suitable for plotting
Z=G.move_to_TriMesh(out)
#
if true # plot_flag
  do_plot(G,Z,"",string("fig_soln",".pdf") )
end
println("Elapsed time=", elapsed_time) 
end
#################
################ end eg_field.jl
####################################