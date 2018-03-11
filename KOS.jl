__precompile__()
#
#@everywhere module KOS
module KOS
#
using Distributions
#
export init, fn_run_wos, fn_wos_core_estimation!, fn_run_wos_many, fn_wos_core_estimation,update_rhs, setup_run_wos, Params, chunk!,vec_fn_wos_single_samp,fn_wos_single_samp,chunkchunk!,chunkchunkchunk!,chunkchunk2!,chunkchunkchunk2!

#
function __init__()
end
#
type Params
  fn_ext::Function  # exterior data
  fn_rhs::Function  # source (right-hand side) function
  d::Function       # signed distance to the boundary
  alpha::Float64    # stability index (Laplace exponent is alpha/2)
end

#####################################################################################
# Begin of WOSRNG functions
######################################################################################
######################################################################################
type WOSRNG
  beta_samples::Array{Float64}  # samples
  angle_samples::Array{Float64,2}  #
  radius_samples::Array{Float64}  #
  cradius_samples::Array{Float64}  #
  # iterator (second index)
  it::Int64  # 
  # 
  rv_length::Int64 # number of samples in first index
  beta_dist # beta distribution
  alpha # alpha
end
#
function wosrng_reset!(w::WOSRNG)
  # set dimension and reset iterator
  w.it=1
end
#
function get_beta!(w::WOSRNG)
  return w.beta_samples[w.it]
end
#
function get_angle!(w::WOSRNG)
  # get uniform sample
  return w.angle_samples[w.it,1:2]
end
#
function get_angle2!(w::WOSRNG)
  # get uniform sample
  return w.angle_samples[w.it,3:4]
end
#
function get_radius!(w::WOSRNG) # first
  c     =w.cradius_samples[w.it]
  sample=w.radius_samples[w.it]
  return sample,c
end
#
#
function wosrng_load(beta_dist,alp)
  # construct WOSRNG object
  #print("wosrng_load with rv_length=",rv_length,"  ")
  rv_length=10
  th=2*pi*rand(rv_length,2);    
  u1=[cos.(th[:,1]) sin.(th[:,1]) cos.(th[:,2]) sin.(th[:,2])]
  u2=rand(rv_length).^(1/alp)
  u3=cdf(beta_dist,1-u2.^2)
  b=sqrt.(rand(beta_dist,rv_length)); 
  #
  return WOSRNG(b,u1,u2,u3,0,
                rv_length,beta_dist,alp)
end
#
function wosrng_extend(w::WOSRNG, grow)
  w.it+=1
  if (w.rv_length<grow+w.it)
    w.rv_length+grow 
    #
    th=2*pi*rand(2*grow,2)
    u1=[cos.(th[:,1]) sin.(th[:,1]) cos.(th[:,2]) sin.(th[:,2])]
    w.angle_samples =vcat(w.angle_samples, u1)
    #
    w.radius_samples=vcat(w.radius_samples, rand(grow).^(1/w.alpha))
    w.cradius_samples=cdf(w.beta_dist, 1-w.radius_samples.^2)
    #
    w.beta_samples  =vcat(w.beta_samples,sqrt.(rand(w.beta_dist,grow)) )
  end
end
# 
function wosrng_reload!(w::WOSRNG)
  # New random random variables
  th=2*pi*rand(w.rv_length,2);    
  u1=[cos.(th[:,1]) sin.(th[:,1]) cos.(th[:,2]) sin.(th[:,2])]
  u2=rand(w.rv_length).^(1/w.alpha)
  u3=cdf(w.beta_dist,1-u2.^2)
  b=sqrt.(rand(w.beta_dist,w.rv_length)); 
  w.angle_samples =u1
  w.radius_samples=u2
  w.cradius_samples=u3
  w.beta_samples  =b
  #
  w.it   =0
end
######################################
#####################################################################################
# End of WOSRNG functions
######################################################################################
######################################################################################
#  Begin distance/example PDE function
#
function d_ball_0_1( x::Array{Float64,1} )
  # out=d_ball_0_1(x)
  #
  # Distance of points to the domain boundary
  # Domain is a ball of radius 1 centred at (0,0)
  # x: point
  R=1.
  return R-sqrt(sum(x.^2))
end
#
function vec_d_ball_0_1( xs )
  # out=d_ball_0_1(x)
  # same as above
  # points are rows of xs
  R=1.
  return (R-sqrt.(sum(xs.^2,length(size(xs)))))[:]
end
#

function d_box( x )
  # out=d_box(x)
  #
  # Distance of points to the domain boundary
  # Domain is a box [a,b]x[a,b]
  # x
  a=-1; b=1.
  return minimum([x[1]-a,b-x[1],x[2]-a,b-x[2]])
end
#
function vec_fn_zero(x::Array{Float64,1})
  # Zero exterior function
  # x: point
  return 0.
end
#
function vec_fn_zero(x::Array{Float64,2})
  # Zero exterior function
  # x: point(s) as rows
  return zeros(size(x)[1])
end
#
function  vec_fn_ext1(x::Array{Float64,1}, alpha)
  # Exterior function/model solution: Free-space
  # Green's function centred at P
  # From D1.7 of Bucur
  # See Section 7, Example 1 of paper
  #
  # x: point(s) as columns for evaluation
  #
  P=[2. 0.] # any point outside ball
  tmp= sum((broadcast(-,x,P).^2)).^(-1+(alpha/2))
  return tmp
end
#
function  vec_fn_ext1(x::Array{Float64,2}, alpha)
  # Exterior function/model solution: Free-space
  # Green's function centred at P
  # From D1.7 of Bucur
  # See Section 7, Example 1 of paper
  #
  # x: point(s) as columns for evaluation
  #
  P=[2. 0.0] # any point outside ball
  tmp=sum((broadcast(-,x,P).^2),2).^(-1+(alpha/2))
  return tmp[:]
end
#
function vec_fn_ext2(x::Array{Float64,2})  #
  # Exterior function: Gaussian with P (vectorised)
  # See Section 7, Example 2 of paper
  # Solution (numerical) for this problem with g=0 is u_hom_check
  #
  # x: point(s) as columns for evaluation
  #
  P=[2 0]; # any point outside ball
  tmp=exp(-sum(broadcast(-,x,P).^2,2))
  return tmp[:]
end
#
function vec_fn_ext2(x::Array{Float64,1})
  P=[2 0]; # any point outside ball
  tmp=exp(-sum(broadcast(-,x,P).^2))
  return tmp
end
#
function fn_rhs1(x::Array{Float64,2}, alpha )
  # Source-term function: Constant example
  # From Dyda, page 549, Table 3, Example 2 (rescaled)
  # Exact solution for this problem with f=0 is fn_exact1
  # x: point(s) as columns for evaluation
  return ones(size(x)[1])::Array{Float64,1}
end
#
function fn_rhs1(x::Array{Float64,1}, alpha )
  # Source-term function: Constant example
  # From Dyda, page 549, Table 3, Example 2 (rescaled)
  # Exact solution for this problem with f=0 is fn_exact1
  # x: point(s) as columns for evaluation
  return 1::Float64
end
#
function fn_rhs2( x::Array{Float64,1}, alpha )
  # Source-term function: Non-constant example
  # From Dyda, page 549, Table 3, Example 2
  # Exact solution for this problem with f=0 is u4
  #
  # x: point(s) as columns for evaluation
  w=alpha/2
  c=(2^alpha)*gamma(2+w)*gamma(1+w)
  tmp=c*(1-(1+w)*sum(x.^2))
  return tmp::Float64
end
#
function fn_rhs2( x::Array{Float64,2}, alpha )
  # Source-term function: Non-constant example
  # From Dyda, page 549, Table 3, Example 2
  # Exact solution for this problem with f=0 is u4
  #
  # x: point(s) as columns for evaluation
  w=alpha/2
  c=(2^alpha)*gamma(2+w)*gamma(1+w)
  tmp=c*(1-(1+w)*sum(x.^2, 2))
  return tmp[:]::Array{Float64,1}
end
#
#
function fn_c_rhs( x::Array{Float64,1} )
  # Right-hand side designed to expose slow coupling rates.  #
  # x: point(s) as columns for evaluation
  tmp=2+sum(x.^2)
  return tmp::Float64
end
#
function fn_c_rhs( x::Array{Float64,2} )
  tmp=2+sum(x.^2, 2)
  return tmp[:]::Array{Float64,1}
end
#
#
#
function fn_c_ext( x::Array{Float64,1})
  # Exterior values
  # Designed to expose slow coupling rates
  tmp=sin((sum(x.^2)))
  return tmp::Float64
end
#
function fn_c_ext( x::Array{Float64,2})
  # 
  tmp=sin.((sum(x.^2, 2)))
  return tmp[:]::Array{Float64,1}
end
#
function fn_exact1(x,alpha)
  # Model solution: Constant source example
  # From Dyda, page 549, Table 3, Example 1
  # x: point(s) as columns for evaluation
  w=alpha/2; z=1-alpha/2;
  c=((2^alpha)*gamma(1+w)*w*beta(w,z)/ gamma(z))
  if d_ball_0_1(x)>0
   return (1.-sum(x.^2)).^(alpha/2)/c
  else
   return 0.
  end
end
#
function fn_exact2(x::Array{Float64,1}, alpha )
  # Model solution: Non-constant source example
  # From Dyda, page 549, Table 3, Example 2
  # x: point(s) as columns for evaluation
  if d_ball_0_1(x)>0
    return (1-sum(x.^2)).^(1+(alpha/2))
  else
    return 0.
  end
end
#
function example0(alpha) # all zero
  p=KOS.Params(vec_fn_zero,vec_fn_zero,vec_d_ball_0_1,alpha)
  exact = x->0. # dummy
  return p, exact
end
#
function example2(alpha) # quadratic rhs
  println("Example 2: quadratic rhs")
  p=KOS.Params(vec_fn_zero,x->fn_rhs2(x,alpha),vec_d_ball_0_1,alpha)
  exact = x->fn_exact2(x,alpha) #yes
  return p, exact
end
#
function example1(alpha) # constant rhs
  println("Example 1: constant rhs")
  p=KOS.Params(vec_fn_zero,x->fn_rhs1(x,alpha), vec_d_ball_0_1, alpha)
  exact = x->fn_exact1(x,alpha) #yes
  return p, exact
end
#
function example3(alpha) # Green's fn
  println("Example 3: Green's fn exterior")
  p=KOS.Params(x->vec_fn_ext1(x,alpha),vec_fn_zero,vec_d_ball_0_1,alpha)
  return p, x->vec_fn_ext1(x,alpha)
end
#
function example4(alpha)
  # Green's fn
  println("Example 4: Green's fn exterior of box")
  p=KOS.Params(fn_ext2,fn_zero,d_box,alpha)
  return p, fn_ext2
end
#
function example5(alpha)
  # Difficult for coupling
  println("Example 5: difficult for coupling")
  p=KOS.Params(fn_c_ext,fn_c_rhs,vec_d_ball_0_1,alpha)
  return p, (x->0.0)
end
##########################################################################################
##########################################################################################
# end PDE functions
##########################################################################################
function setup_run_wos(p::KOS.Params,dict::Dict)
  alp=p.alpha; z=(2-alp)/2; w=alp/2
  # (in_c)= 2 pi* c(alp,2)*B(z,w) on page 1 of ideas.pdf
  in_c = alp^(-1) *2^(-alp+1) *gamma(w)^(-2) *beta(z,w)
  dict["c"]=in_c
  #
  beta_dist=Beta(w,z)
  dict["dist_beta"]=  beta_dist
  # in_val=E[(1-I(R^2,z,w))] for R=X^(1/alp) and X~u(0,1)
  # = int (1-I(x^(2/alp),z,w)) dx
  function integrand(X)
    return cdf(beta_dist,1-X.^(2/alp))
  end
  in_val, error=quadgk(integrand,0,1)
  dict["val"]=in_val
  function my_step(d::Array{Float64,1})
    return vec_wos_step(d,beta_dist)
  end
  if haskey(dict,"no_samp_MC")
    in_no       =dict["no_samp_MC"]
  else
    in_no=1; dict["no_samp_MC"]=1
  end
  my_rhs_ball =update_rhs(p,dict)
  dist_fn=y::Array{Float64,1}->p.d(y)[1]
  ext_fn=y::Array{Float64,1}->p.fn_ext(y)[1]
  #
  return my_step,my_rhs_ball,dist_fn,ext_fn
end
#
function update_rhs(p::Params,dict::Dict)
  in_val     = dict["val"]
  no_samp    = dict["no_samp_MC"]
  in_c       = dict["c"]
  beta_dist  = dict["dist_beta"]
  # 
  my_rhs_ball=(rhon::Array{Float64,2},rn::Array{Float64,1}) ->vec_eval_MC0(rhon, rn, in_c, in_val, beta_dist,p.alpha,p.fn_rhs)::Array{Float64,1}
  #
   return my_rhs_ball
end

##########################################################################################
function eval_MC0(rhon::Array{Float64,1}, rn::Float64, 
    in_c::Float64,in_value::Float64, beta_dist::Distributions.Beta{Float64}, alp::Float64, fn_rhs::Function)
# rhon=current position, rn=current radius,
# in_c, in_value useful constant
# beta_dist useful Beta distribution
# alp=alpha
# fn_rhs=function
  r   = rand()^(1/alp)
  tmp = polar_quad0(r*rn, rhon, fn_rhs) * cdf(beta_dist,1-r^2)
  return (in_c*(rn^alp)*(tmp+in_value*fn_rhs(rhon)))::Float64
end
##########################################################################################
function vec_eval_MC0(rhon::Array{Float64,2}, rn::Array{Float64,1}, 
    in_c::Float64,in_value::Float64, beta_dist::Distributions.Beta{Float64}, alp::Float64, fn_rhs::Function)
# rhon=current position, rn=current radius,
# in_c, in_value useful constant
# beta_dist useful Beta distribution
# alp=alpha
# fn_rhs=function
  r   = rand()^(1/alp)
  tmp = vec_polar_quad0(r*rn, rhon, fn_rhs) * cdf(beta_dist,1-r^2)
  return (in_c*(rn.^alp).*(tmp+in_value*fn_rhs(rhon)))::Array{Float64,1}
end
###########################################################################################
function eval_MC_coupled(rhon::Array{Float64,1}, rn::Float64, 
    in_c::Float64,in_value::Float64, beta_dist::Distributions.Beta{Float64}, 
    alp::Float64, fn_rhs::Function, w::WOSRNG)
# rhon=current position, rn=current radius,
# in_c, in_value useful constant
# beta_dist useful Beta distribution
# alp=alpha
# fn_rhs=function
  r,c  = get_radius!(w)#^(1/alp)
  tmp= polar_quad_coupled(r*rn, rhon, fn_rhs, w) *c
  # cdf(beta_dist,1-r^2)
  return (in_c*(rn^alp)*(tmp+in_value*fn_rhs(rhon)))::Float64
end
#########################################################################################
function eval_MC_coupled_alt(rhon::Array{Float64,1}, rn::Float64,  
    in_c::Float64,in_value::Float64, beta_dist::Distributions.Beta{Float64}, 
    alp::Float64, fn_rhs::Function, w::WOSRNG)
# rhon=current position, rn=current radius,
# in_c, in_value useful constant
# beta_dist useful Beta distribution
# alp=alpha
# fn_rhs=function
  r,c = get_radius!(w)#^(1/alp)
  tmp=polar_quad_coupled_alt(r*rn, rhon, fn_rhs, w) * c
  #cdf(beta_dist,1-r^2)
  return (in_c*(rn^alp)*(tmp+in_value*fn_rhs(rhon)))::Float64
end
##########################################################################################
function  polar_quad0(r::Float64, rhon::Array{Float64,1}, 
  fn_rhs::Function)
  # Function fn_rhs(rhon+r*x) in terms of polar coordinates of x
  theta = 2*pi*rand() 
  X=r*[cos(theta),sin(theta)]
  return (fn_rhs(rhon+X)-fn_rhs(rhon))::Float64
end
##########################################################################################
function  vec_polar_quad0(r::Array{Float64,1}, rhon::Array{Float64,2}, 
  fn_rhs::Function)
  # Function fn_rhs(rhon+r*x) in terms of polar coordinates of x
  theta = 2*pi*rand() 
  X=r*[cos(theta),sin(theta)]'
  return (fn_rhs(rhon+X)-fn_rhs(rhon))::Array{Float64,1}
end
########################################################################################
function  polar_quad_coupled_alt(r::Float64, rhon::Array{Float64,1}, 
  fn_rhs::Function, w::WOSRNG)
  # 
  # Function fn_rhs(rhon+r*x) in terms of polar coordinates of x
  # 
  X = get_angle!(w)
  X=r*X;  Y=r*[-X[2],X[1]]
  mysum=0.25*(fn_rhs(rhon+X)+fn_rhs(rhon-X)+
              fn_rhs(rhon+Y)+fn_rhs(rhon-Y))
  return (mysum-fn_rhs(rhon))::Float64
end
########################################################################################
function  polar_quad_coupled(r::Float64, rhon::Array{Float64,1}, 
  fn_rhs::Function, w::WOSRNG)
  # 
  # Function fn_rhs(rhon+r*x) in terms of polar coordinates of x
  # 
  X = r*get_angle!(w)
  return (fn_rhs(rhon+X)-fn_rhs(rhon))::Float64
end
#######################################################################################
function wos_step(d::Float64, dist::Distributions.Beta{Float64})
  r     = d/sqrt(rand(dist))
  theta = rand()*2*pi
  return r*[cos(theta); sin(theta)]::Array{Float64,1}
end
#######################################################################################
function vec_wos_step(d::Array{Float64,1}, dist::Distributions.Beta{Float64})
  r     = d/sqrt(rand(dist))
  theta = rand()*2*pi
  return (r.*[cos(theta), sin(theta)]')::Array{Float64,2}
end
########################################################################################
function wos_step_coupled(d::Float64, w::WOSRNG)
  r     = d/get_beta!(w)
  X = get_angle2!(w)
  return r*X::Array{Float64,1}
end
#########################################################################################
#########################################################################################
# end key WOS functions
#########################################################################################
function  fn_wos_core_estimation!(val::Array{Float64,2},
    x0::Array{Float64,1},my_step::Function, 
    my_rhs_ball::Function, dist_fn::Function,
    ext_fn::Function,  no_its::Int64,  method::Bool)
  # Executes WoS random walks for initial data x0
  # Results returned in val (first col homogeneou part, second col inhom part)
  # Produce no_its number of setup_run_wos
  # method=TRUE for non-zero rhs (false otherwise)
  # must be "coupled"=false
  y=deepcopy(x0); d0=dist_fn(x0);
  #
  if (method) # Use when rhs non-zero
    for i=1:no_its
        d=d0;   y=x0  # initial position of WoS
        val[i,2]=0.   # inhom
        while (d>0.)  # walk
          # inhomog part for current sphere (y,d)
          val[i,2] +=my_rhs_ball(reshape(y,(1,2)),[d])[1]
          y+=my_step([d])[1] #  next sphere
          d=dist_fn(y)
        end # Exit domain
        # Calculate homogeneous part at the exit point
        val[i,1]=ext_fn(y)# homogenous
      end # End i loop
  else # Use when rhs is zero and method set to False
    for i=1:no_its
        d=d0; y=x0
        while (d>0.)
          y+=my_step(d)#  next sphere
          d=dist_fn(y)
        end # Exit domain
        # Calculate the homogeneous part at the exit point
        val[i,1]=ext_fn(y)
    end # End i loop
  end
end
########################################################################################################
function  fn_wos_core_estimation(x0::Array{Float64,1},
 my_step::Function, my_rhs_ball::Function,dist_fn::Function,ext_fn::Function, 
  no_its::Int64,  method::Bool)
  # Executes WoS random walks for initial data x0
  # Results [mean,variance] returned as return
  # Produce no_its number of setup_run_wos
  # method=TRUE for non-zero rhs (false otherwise)
  # must be "coupled"=false
  y=deepcopy(x0); d0=dist_fn(x0); mom1=0.; mom2=0.
  #
  if (method) # Use when rhs non-zero
    for i=1:no_its
        tmp=0.; d=d0;   y=x0  # initial position of WoS
        while (d>0)
          tmp+=my_rhs_ball(y,d); 
          y+=my_step(d) #  next sphere
          d=dist_fn(y)
        end # Exit domain
        # Calculate homogeneous part at the exit point
        tmp+=ext_fn(y)# homogenous
        mom1+=tmp; mom2+=tmp^2
      end # End i loop
  else # Use when rhs is zero and method set to False
    for i=1:no_its
        d=d0; y=x0
        while (d>0.)
          y+=my_step(d)#  next sphere
          d=dist_fn(y)
        end # Exit domain
        # Calculate the homogeneous part at the exit point
        tmp=ext_fn(y); mom1+=tmp; mom2+=tmp^2
    end # End i loop
  end
  mom1/=no_its; mom2/=no_its
  return mom1, (mom2-mom1^2)
end
##################################################################################
########################################################################################################
function  fn_wos_single_samp(x0s::Array{Float64,2},
 my_step::Function, my_rhs_ball::Function,dist_fn::Function,ext_fn::Function, index_fn::Function,
    method::Bool)
  # Executes WoS random walks for initial data x0
  # Results [mean,variance] returned as return
  # Produce no_its number of setup_run_wos
  # method=TRUE for non-zero rhs (false otherwise)
  no_pts=size(x0s)[1]; out=zeros(no_pts)
  for i=1:no_pts
    index_fn()#
    x0=x0s[i,:]; d0=dist_fn(x0)
    #
    if (method) # Use when rhs non-zero
      tmp=0.; d=d0;   y=x0  # initial position of WoS
      while (d>0)
        tmp+=my_rhs_ball(y,d); 
        y+=my_step(d) #  next sphere
        d=dist_fn(y)
      end # Exit domain
      # Calculate homogeneous part at the exit point
      tmp+=ext_fn(y)# homogenous
    else # Use when rhs is zero and method set to False
      d=d0; y=x0
      while (d>0.)
        y+=my_step(d)#  next sphere
        d=dist_fn(y)
      end # Exit domain
      # Calculate the homogeneous part at the exit point
      tmp=ext_fn(y)
    end
    out[i]=tmp
  end
  return out
end
# 
function  vec_fn_wos_single_samp(x0s::Array{Float64,2},
 my_step::Function, my_rhs_ball::Function, dist_fn::Function, ext_fn::Function)
  # Executes WoS random walks for vector initial data x0 (points in rows)
  # Gives a  single vector of WoS samples (for each IC)
  # method=TRUE for non-zero rhs (false otherwise)
  no_pts=size(x0s)[1]
  tmp=zeros(no_pts) 
  init_dist=dist_fn(x0s)
  #
  alive_index=find(z->(z>0),init_dist)
  d_alive=init_dist[alive_index]
  y=deepcopy(x0s)  # initial positions of WoS
  while (length(alive_index)>0)
    tmp[alive_index]+=my_rhs_ball(y[alive_index,:],d_alive) 
    y[alive_index,:]+=my_step(d_alive) #  next sphere
    d_alive    =dist_fn(y[alive_index,:])
    sub_alive  =find(z->(z>0),d_alive)
    alive_index=alive_index[sub_alive]
    d_alive    =d_alive[sub_alive]
  end # all points exit domain
  # Calculate homogeneous part at the exit point
  tmp+=ext_fn(y)# homogenous
  return tmp
end
##################################################################################
####################################
function chunkchunk!(delta_sum::SharedArray,
                    no_samps,my_step,
                    my_rhs_ball,dist_fn,ext_fn,
                    finest_level_int,correction_level,pa,pts)
  idx=max(1,myid()-1) # processor number
  tmp=zeros(correction_level[end])
  m=0
  while (m<no_samps )
    # run wos 
    tmp[correction_level]=vec_fn_wos_single_samp(pts, 
                   my_step, my_rhs_ball, dist_fn, ext_fn)
    m+=1
    # update as corrections
    for i=finest_level_int # update on finest interior
      delta  = tmp[i] - mean(tmp[pa[i,:]])     
      delta_sum[i,idx]+= delta  
    end    
  end # loop over samples
end
####################################
function chunkchunk2!(delta_sum::SharedArray,
                    delta_sq_sum::SharedArray,
                    no_samps,my_step,
                    my_rhs_ball,dist_fn,ext_fn,
                    finest_level_int,correction_level,pa,pts)
  idx=max(1,myid()-1)
  tmp=zeros(correction_level[end])
  #
  m=0
  while (m<no_samps )
     # run wos 
     tmp[correction_level]=vec_fn_wos_single_samp(pts, 
                    my_step, my_rhs_ball, dist_fn, ext_fn)
     m+=1
     # update as corrections
     for i=finest_level_int # update on finest interior
        delta               = tmp[i] - mean(tmp[pa[i,:]])     
        delta_sum[i,idx]   += delta 
        delta_sq_sum[i,idx]+= delta^2  
     end    
   end # loop over samples
end
####################################
function chunkchunkchunk!(delta_sum::SharedArray, no_samps,
                          my_step, my_rhs_ball,dist_fn,ext_fn,
                          coarse_level, pts)
  #
  idx=max(1,myid()-1) # processor number
  m=0
  while (m<no_samps )
     # run wos 
     delta_sum[coarse_level,idx]+=vec_fn_wos_single_samp(pts, 
                   my_step, my_rhs_ball, dist_fn, ext_fn)
     m+=1    
   end # loop over samples
end
####################################
function chunkchunkchunk2!(delta_sum::SharedArray, delta_sq_sum::SharedArray,
                    no_samps,
                    my_step, my_rhs_ball, dist_fn, ext_fn,
                    coarse_level, pts)
  idx=max(1,myid()-1) # processor number
  m=0
  while (m<no_samps )
     # run wos 
     tmp=vec_fn_wos_single_samp(pts, 
                   my_step, my_rhs_ball, dist_fn, ext_fn)
     delta_sum[coarse_level,idx]+=tmp
     delta_sq_sum[coarse_level,idx]+=tmp.^2
     m+=1    
  end # loop over samples
end
######################
############################
function fn_run_wos_many(x0::Array{Float64,2}, 
  p::Params, dict::Dict, my_step::Function, 
  my_rhs_ball::Function, dist_fn::Function,
  ext_fn::Function)
  #
  # Private driver routine for Walk-on-spheres algorithm
  #
  # x0: initial position
  # p: params for PDE
  # dict: dict of method parameters*
  # return mean_val, var_sample, err_est, no_samples
  #
  # initialise
  no_per_batch=dict["no_per_batch"]
  max_samples =dict["max_samples"]/dict["no_per_batch"]
  # two error estimates (for hom, inhom parts)
  err_est=dict["tol"]+ones(2); num=zeros(2); val=zeros(no_per_batch,2)
  # initialise
  no_particles=size(x0)[end]
  val=zeros(no_per_batch,2)
  mean_val=zeros(no_particles)
  #
  if dict["coupled"]==true
    wosrng_reload!(dict["WOSRNG"])  
  end
  #
  assert(dict["coupled"]==false)
  index_fn=dict["index_fn"]
  #
  for i=1:(size(x0)[end])
    if dist_fn(x0[:,i])>0
      fn_wos_core_estimation!(val, x0[:,i], my_step, my_rhs_ball, dist_fn, ext_fn, index_fn, no_per_batch, true)
      mean_val[i]   +=2*mean(val)
    end
  end
  #
  return mean_val
end
##########################################################################################
#
function fn_run_wos(x0::Array{Float64,1}, p::Params, dict::Dict,
 my_step::Function, my_rhs_ball::Function, dist_fn::Function,
 ext_fn::Function)
  #
  # Private driver routine for Walk-on-spheres algorithm
  #
  # x0: initial position
  # p: params for PDE
  # dict: dict of method parameters*
  # return mean_val, var_sample, err_est, no_samples
  #
  # initialise
  no_per_batch=dict["no_per_batch"]
  max_samples =dict["max_samples"]/dict["no_per_batch"]
  # two error estimates (for hom, inhom parts)
  err_est=dict["tol"]+ones(2); num=zeros(2); val=zeros(no_per_batch,2)
  # initialise
  sum_val    =zeros(2); sum_sq_val=zeros(2);  mean_val=zeros(2)
  mean_sq_val=zeros(2); var_sample=zeros(2)
  #index_fn=dict["index_fn"]
  #
  while ( err_est[2]>(dict["tol"]/2) && num[1]<max_samples)
    # stopping criteria on inhomogeneous part
    num[1] += 1
    fn_wos_core_estimation!(val, x0, my_step, my_rhs_ball, dist_fn, ext_fn,  no_per_batch, true)
    sum_val   +=mean(val,1)[:]
    sum_sq_val+=mean(val.^2,1)[:]
    mean_val   =sum_val/num[1]
    mean_sq_val=sum_sq_val/num[1]
    var_sample =mean_sq_val-mean_val.^2
    err_est=sqrt.(var_sample/num[1])
  end
  num[2]=num[1]
  #
  while err_est[1]>(dict["tol"]/2) && num[1]<max_samples
    # homogeneous part only
    num[1]+=1
    fn_wos_core_estimation!(val,x0, my_step, my_rhs_ball,dist_fn,ext_fn,  no_per_batch,false)
    sum_val[1]   =sum_val[1]   +mean(val[:,1])
    sum_sq_val[1]=sum_sq_val[1]+mean(val[:,1].^2)
    mean_val[1]   =sum_val[1]/num[1];
    mean_sq_val[1]=sum_sq_val[1]/num[1];
    var_sample[1] =mean_sq_val[1]-mean_val[1]^2
    err_est[1]=sqrt(var_sample[1]/num[1]);
  end
  no_samples=no_per_batch * num
  #
  return mean_val, var_sample, err_est, no_samples
end
#
######################################################
end# end of module KOS