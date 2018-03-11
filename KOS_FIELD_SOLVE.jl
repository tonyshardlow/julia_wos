__precompile__()
# WOS Fractional Laplacian
# The algorithms are implemented here.
########################
module KOS_FIELD_SOLVE
#######################
using Distributions
using Interpolations
#
using KOS
using KOSHELPER
################################
export init, get_solution, wos_ev
################################
function __init__()
end
##########################
function ann_ml_params(p, dict, G, no_samps, report_flag)
  # Analyse working of ML
  # compute variance rates, cputimes, total runtimes
  # and fastest start level
  no_levels=dict["no_levels"];  l0=dict["l0"]
  #
  if report_flag
    println("====\nAnalyse multilevel params with lmax=", no_levels)
  end
  #############################################################
  # get variances and costs
  ##############################################################
  cvariance=zeros(no_levels);  ccost=ones(no_levels)
  variance=zeros(no_levels);  cost=ones(no_levels)
  #
  # coarse level
  for ell=l0:no_levels  
    coarse_level=intersect(G.int,1:G.get_level_ind(ell)[end])
    no_samps=Int64(ceil(dict["no_samps"]))
    # initialise
    tic()
    v,no_samps=uncoupled_factor_par2(no_samps, G, p, dict,coarse_level)
    cvariance[ell]=G.sq_int(sqrt.(v),l0)
    ccost[ell]    =toq()/(no_samps)
  end
  # coupled levels
  for ell=(l0+1):(no_levels)
    finest_level_int=intersect(G.int,G.get_level_ind(ell))
    correction_level=intersect(G.int,1:G.get_level_ind(ell)[end])
    no_samps=Int64(ceil(dict["no_samps"]))
    tic()
    v,no_samps=coupled_factor_par2(no_samps, G, p, dict,
       correction_level, finest_level_int)
    variance[ell]=G.sq_int(sqrt.(v),ell)
    cost[ell]=toq()/no_samps
  end # loop over levels
  #
  if report_flag
    println("===Completed ml params.")
  end
  ##################################################################
  ## get ratios for variance
  beta_ratio=log.(variance[l0+1:end-1]./variance[l0+2:end])/log(2)
  if report_flag
    println("beta ML param (base 2) ",beta_ratio)
    coord_print(1:length(variance),variance)
    coord_print(2*2.^(-1.0*collect(l0+1:no_levels) ), variance[l0+1:no_levels])
    println("time per coarse uncoupled sample=",ccost)
    println("                  coupled sample=",cost)
    println("variance for uncoupled =",cvariance)
    println("              coupled  =",variance)
  end
  #################################################################### compute cpu times
  level_samps=zeros(Int64,no_levels);  total_time=zeros(no_levels)
  #
  cputimes=zeros(no_levels-l0+1,2); tols=zeros(no_levels-l0+1) 
  for my_no_levels=l0:no_levels # outer loop (max levels)
    my_k=1+my_no_levels-l0
    hfine=2.0^(-my_no_levels); my_tol=hfine^2;
    println("max_levels=", my_no_levels,", hfine=", hfine,"; tolerance=",my_tol);
    for ell=my_no_levels:-1:l0 # inner loop coarse level
      r=ell+1:my_no_levels
      level_samps[1:ell]*=0
      if ell<my_no_levels
        myconst=(sqrt(cvariance[ell]*ccost[ell])
                +sum(sqrt.(variance[r].*cost[r]))
                )*(2/my_tol^2)
        level_samps[r]  =Int64.(ceil.(sqrt.(variance[r]./cost[r]) *myconst))
      else # vanilla
        myconst=sqrt(cvariance[ell]*ccost[ell])*(2/my_tol^2)
      end
      level_samps[ell]=Int64(ceil(sqrt(cvariance[ell]/ccost[ell])*myconst) )
      #
      mycost=deepcopy(cost); mycost[ell]=ccost[ell]
      myvars=deepcopy(variance); myvars[ell]=cvariance[ell]
      total_time[ell]=sum(mycost.*level_samps)
      # 
      if report_flag
        println("ell=",ell,"   level_samps=",level_samps[ell:my_no_levels]) 
        println("             variances=",myvars[ell:my_no_levels])     
        println("          time per level=",(mycost.*level_samps)[ell:my_no_levels],
              " total time=",total_time[ell])
      end # end if
    end # end inner loop
    # find quickest option
    mint,min_ell=findmin(total_time[l0:my_no_levels]); 
    ell=min_ell+l0-1;   r=(ell+1):my_no_levels
    if ell<my_no_levels
        myconst=(sqrt(cvariance[ell]*ccost[ell])
                +sum(sqrt.(variance[r].*cost[r]))
                )*(2/my_tol^2)
      else
        myconst=sqrt(cvariance[ell]*ccost[ell])*(2/my_tol^2)
      end
    level_samps[1:ell]*=0
    level_samps[ell]=Int64(ceil(sqrt(cvariance[ell]/ccost[ell])*myconst) )
    level_samps[r]=Int64.(ceil.(sqrt.(variance[r]./cost[r]) *myconst))
    if report_flag
      println(total_time )
      println("choose ell0=",ell," level_samps",
              level_samps[ell:my_no_levels])
      cputimes[my_k,1]=total_time[ell];
      tols[my_k]=(2.0^(-my_no_levels))^2;
      cputimes[my_k,2]=total_time[my_no_levels]
      println("% ML")
      coord_print(tols,cputimes[:,1])
      println("% Vanilla")
      coord_print(tols,cputimes[:,2])
    end
  end
  #
end
##############################################################
function coord_print(x,y)
  # convenient output of co-ordinates for including in LaTeX
  n=length(x)
  println("coordinates {")
  for i=1:n
    print("(",x[i],",",y[i],") ")
  end
  println("\n};")
end
#####################################################
function ann_ml_coupling(p, dict, G,  report_flag)
  # Multilevel field solve
  no_levels=dict["no_levels"];  l0=2
  #
  if report_flag
    println("====\nAnalyse multilevel params with lmax=", no_levels,
            " with tolerance=",dict["tol"]," and ",dict["no_samps"]," samples")
    println("No workers",nworkers())
    no_samps=dict["no_samps"]
  else
    no_samps=3
  end
  variance=zeros(no_levels);  cost=ones(no_levels)
  #######################################
  # coupled levels
  for ell=(l0+1):(no_levels-1)
    finest_level_int=intersect(G.int,G.get_level_ind(ell))
    correction_level=intersect(G.int,1:G.get_level_ind(ell)[end])
    tic()
    v,no_samps=coupled_factor_par2(no_samps, G, p, dict,
       correction_level, finest_level_int)
    variance[ell]=G.sq_int(sqrt.(v),ell)
    cost[ell]=toq()/no_samps
    println(ell," variance ", variance[ell], ", cost ", cost[ell])
  end # loop over levels
  #
  beta_ratio=log.(variance[l0+1:end-1]./variance[l0+2:end])/log(2)
  hv=2*2.^(-1.0*collect(l0+1:(no_levels-1)) ); vv=variance[l0+1:(no_levels-1)];
  #
  if report_flag
    println("beta ML param (base 2) ",beta_ratio)
    #coord_print(1:length(variance),variance)
    coord_print(hv, vv,p.alpha,true)
    println("time per coupled sample=",cost)
    println(" coupled variance=",variance)
  end
  ell=0
  #
  level_samps=zeros(Int64,no_levels);  total_time=zeros(no_levels)
  return ell, level_samps
end
#######################################
function coord_print(x,y,alpha,fit_flag)
  # convenient output of co-ordinates for including in LaTeX
  # include linear fit
  n=length(x)
  #
  println("coordinates {")
  for i=1:n
    print("(",x[i],",",y[i],") ")
  end
  println("\n}; % alpha=", alpha)
  #
  if (fit_flag)
    lft=linreg(log.(x),log.(y))
    yy=exp(lft[1])*x.^lft[2];
    println(lft)
    println("coordinates {")
    for i=1:n
      print("(",x[i],",",yy[i],") ")
    end
    println("\n};% slope ",lft[2], " alp/(alp+1)=", alpha/(1+alpha))
  end
end
#####################################################
function get_ml_params(p, dict, G, l0, no_samps, report_flag)
  # Multilevel field solve
  # find optimum parametrs (l0,L)
  no_levels=dict["no_levels"];  
  #
  if report_flag
    println("====\nGet multilevel params with l0=",l0,
            " and lmax=", no_levels,
            " with tolerance=",dict["tol"])
  end
  # scale constant for stopping criteria (K_C)
  cvariance=zeros(no_levels);  ccost=ones(no_levels)
  variance=zeros(no_levels);  cost=ones(no_levels)
  ########################################
  # coarse level
  ell=l0
  coarse_level=intersect(G.int,1:G.get_level_ind(ell)[end])
  # initialise
  tic()
  v,no_samps=uncoupled_factor_par2(no_samps, G, p, dict,
       coarse_level)
  cvariance=G.sq_int(sqrt.(v),l0)
  ccost    =toq()/(no_samps)
  #
  #######################################
  # coupled levels
  for ell=(l0+1):(no_levels)
    finest_level_int=intersect(G.int,G.get_level_ind(ell))
    correction_level=intersect(G.int,1:G.get_level_ind(ell)[end])
    tic()
    v,no_samps=coupled_factor_par2(no_samps, G, p, dict,
       correction_level, finest_level_int)
    variance[ell]=G.sq_int(sqrt.(v),ell)
    cost[ell]=toq()/no_samps
  end # loop over correction levels
  #
  if report_flag
    println("===Completed ml params.")
  end
  #
  beta_ratio=log.(variance[l0+1:end-1]./variance[l0+2:end])/log(2)
  if report_flag
    println("beta ML param (base 2) ",beta_ratio)
    println("time per coarse uncoupled sample=",ccost)
    println("time per coupled sample=",cost)
    println(" coarse uncoupled variance=",cvariance)
    println(" coupled variance=",variance)
  end
  #
  level_samps=zeros(Int64,no_levels);  total_time=zeros(no_levels)
  #
  ell=l0
  r=ell+1:no_levels
  level_samps[1:ell]*=0
  if ell<no_levels
      myconst=(sqrt(cvariance*ccost)
                +sum(sqrt.(variance[r].*cost[r]))
                )*(2/dict["tol"]^2)
      level_samps[r]  =Int64.(ceil.(sqrt.(variance[r]./cost[r]) *myconst))
      level_samps[r]=min.(level_samps[r],dict["max_samples"])
  else
      myconst=sqrt(cvariance*ccost)*(2/dict["tol"]^2)
  end
  level_samps[ell]=Int64(ceil(sqrt(cvariance/ccost)*myconst) )
  level_samps[ell]=min.(level_samps[ell],dict["max_samples"])
    #
  mycost=deepcopy(cost); mycost[ell]=ccost
  total_time[ell]=sum(mycost.*level_samps)
  #
  if report_flag
     println("ell=",ell,"   level_samps=",level_samps)     
     println("          time per level=",mycost.*level_samps,
              " total time=",total_time[ell])
  end
  #
  return level_samps
end
#####################################################
######################################################
function wos_field_ml(p, dict, G)
  # Multilevel field solve
  no_levels=dict["no_levels"]; l0=dict["l0"]
  # get key WoS functions
  level_samps=get_ml_params(p,dict,G,l0,600,false)
  #
  if haskey(dict,"report")
    println("====\nMultilevel field solve with l0=",l0,
            " and lmax=", no_levels,
            " with tolerance=",dict["tol"]  )
  end
  ########################################
  # coarse level
  coarse_level=intersect(G.int,1:G.get_level_ind(l0)[end])
  if haskey(dict,"report")
      println("level=",l0,
              ", points on level=",length(coarse_level), 
              ", with samples=", level_samps[l0],
              "." )
  end
  tic()
  out,no_samps=uncoupled_factor_par(level_samps[l0], G, p, dict,
        coarse_level)
  elapsed=toq(); println("elapsed_time=",elapsed)
  #######################################
  # coupled levels
  for ell=(l0+1):no_levels
    # get indices
    finest_level_int=intersect(G.int,G.get_level_ind(ell))
    correction_level=intersect(G.int,1:G.get_level_ind(ell)[end])
    #
    if haskey(dict,"report")
      println("level=",ell, 
              ", points on level=",length(finest_level_int),
              ", with samples=", level_samps[ell],".")
    end 
    #
    tic()
    delta_sum,no_samps=coupled_factor_par(level_samps[ell], 
                     G, p, dict, 
                    correction_level, finest_level_int)
    elapsed=toq(); println("elapsed_time=",elapsed)
    #
    for i=finest_level_int # update out with correction
         pa=G.get_parents(i)
         out[i]=mean(out[pa])+delta_sum[i]
    end
    #
    if haskey(dict,"report")
      println("level=",ell, 
              ", H=",G.scale(ell)^2, # length scale for this lev,
              ", points on level=",length(finest_level_int))
    end 
  end # loop over correction levels
  #
  #
  if haskey(dict,"report")
    println("===Completed multilevel solve.")
  end
  return out
end
###########################################
###########################################
function coupled_factor_par(no_samps, G, p, dict,
       correction_level, finest_level_int)
  # compute contribution form a ML correction levels
  pts=G.get_pt_all(correction_level)
  N=G.n_int+G.n_ext; pa=zeros(Int64,N,2)
  #
  for i=finest_level_int # get parents
      pa[i,:] = G.get_parents(i)
  end 
  #
  np=nworkers() # no processors 
  delta_sum=SharedArray{Float64}((N,np) )
  my_rhs_ball=Array{Any}(np)
  my_step=Array{Any}(np)
  #
  for i=1:np
    my_step[i],my_rhs_ball[i],dist_fn,ext_fn=setup_run_wos(p,dict)
  end
  dist_fn=p.d; ext_fn=p.fn_ext
  #
  no_samps=ceil(no_samps/np)
  @sync begin
    for k in 1:np
      j=workers()[k]
      @async  remotecall_wait(chunkchunk!,j,
                    delta_sum, no_samps,  my_step[k],
                    my_rhs_ball[k], dist_fn, ext_fn,
                    finest_level_int,
                    correction_level, pa, pts)
    end # proc loop
  end # @sync
  if haskey(dict,"report")
    print(".")
  end
  return mean(delta_sum,2)/no_samps, no_samps*np
end 
###########################################
function coupled_factor_par2(no_samps, G, p, dict,
       correction_level, finest_level_int)
  # variant of above with variance output
  pts=G.get_pt_all(correction_level)
  N=G.n_int+G.n_ext; pa=zeros(Int64,N,2)
  #
  for i=finest_level_int # get parents
      pa[i,:] = G.get_parents(i)
  end 
  #
  np=nworkers()# no processors for parallel
  delta_sum=SharedArray{Float64}((N,np) )
  delta_sq_sum=SharedArray{Float64}((N,np) )
  #tmp=SharedArray{Float64}((N,np) )
  my_rhs_ball=Array{Any}(np)
  my_step=Array{Any}(np)
  #
  for i=1:np
    my_step[i],my_rhs_ball[i],dist_fn,ext_fn=setup_run_wos(p,dict)
  end
  dist_fn=p.d; ext_fn=p.fn_ext
  #
  #  if haskey(dict,"report")
  #   print("Start parallel comp on ", workers())
  # end
  #
  no_samps=ceil(no_samps/np)
  println("no_samps=", no_samps,", length correction_level=",length(correction_level))
  @sync begin
    for k in 1:nworkers()
      j=workers()[k]
      @async  remotecall_wait(chunkchunk2!,j,
                    delta_sum, delta_sq_sum, #tmp,
                    no_samps,
                    my_step[k],
                    my_rhs_ball[k], dist_fn, ext_fn,
                    finest_level_int,
                    correction_level, pa, pts)
    end # proc loop
  end # @sync
  if haskey(dict,"report")
    print(".")
  end
  m=mean(delta_sum,2)/no_samps
  m2=mean(delta_sq_sum,2)/no_samps
  return m2-m.^2, no_samps*np
end 
###########################################
###########################################
function uncoupled_factor_par(no_samps, G, p, dict,
        coarse_level)
  # compute uncoupled contribution to ML
  pts=G.get_pt_all(coarse_level)
  N=G.n_int+G.n_ext; 
  #
  np=nworkers()# no processors for parallel
  delta_sum=SharedArray{Float64}((N,np) )
  my_rhs_ball=Array{Any}(np)
  my_step=Array{Any}(np)
  #
  for i=1:np
    my_step[i],my_rhs_ball[i],dist_fn,ext_fn=setup_run_wos(p,dict)
  end
  dist_fn=p.d; ext_fn=p.fn_ext
  #
  no_samps=ceil(no_samps/np)
  @sync begin
    for k in 1:nworkers()
      j=workers()[k]
      @async  remotecall_wait(chunkchunkchunk!, j,
                    delta_sum,  no_samps,
                    my_step[k], my_rhs_ball[k], dist_fn, ext_fn,
                    coarse_level, pts)
    end # proc loop
  end # @sync
  if haskey(dict,"report")
    print(".")
  end
  return mean(delta_sum,2)/no_samps,no_samps*np
end
###########################################
###########################################
function uncoupled_factor_par2(no_samps, G, p, dict,
        coarse_level)
  # variant of above with variance output
  pts=G.get_pt_all(coarse_level)
  N=G.n_int+G.n_ext; 
  #
  np=nworkers()# no processors for parallel
  delta_sum=SharedArray{Float64}((N,np) )
  delta_sq_sum=SharedArray{Float64}((N,np) )
  my_rhs_ball=Array{Any}(np)
  my_step=Array{Any}(np)
  #
  for i=1:np
    my_step[i],my_rhs_ball[i],dist_fn,ext_fn=setup_run_wos(p,dict)
  end
  dist_fn=p.d; ext_fn=p.fn_ext
  #
  no_samps=ceil(no_samps/np)
  @sync begin
    for k in 1:nworkers()
      j=workers()[k]
      @async  remotecall_wait(chunkchunkchunk2!,j,
                    delta_sum, delta_sq_sum,  no_samps,
                    my_step[k], my_rhs_ball[k], dist_fn, ext_fn,
                    coarse_level, pts)
    end # proc loop
  end # @sync
  m=mean(delta_sum,2)/no_samps
  m2=mean(delta_sq_sum,2)/no_samps
  if haskey(dict,"report")
    print(".")
  end
  return abs.(m2-m.^2), no_samps*np
end 
###########################
###########################
function get_solution(p, dict, G)  
  # driver routine for access to WOS field solve
  tic()
  if  dict["solve_method"]=="simple"  
    l0=dict["l0"]
    dict["l0"]=dict["no_levels"]
    out=wos_field_ml(p, dict, G)  
    dict["l0"]=l0  
  elseif dict["solve_method"]=="ml_analyse"
    # find bests ML paramters
    #ann_ml_params(p, dict, G,3,true)
    ann_ml_params(p, dict, G,dict["no_samps"],true)
    out=zeros(G.n_int) 
  elseif dict["solve_method"]=="couple_analyse"
    ann_ml_coupling(p, dict, G,true)
    #
    out=zeros(G.n_int) 
  else
    out=wos_field_ml(p, dict, G)  
  end
  elapsed_time=toq()
  
  return out[:],elapsed_time
end
######################
######################
function  wos_ev(p::KOS.Params, dict, G, arnoldi, lam_old)
  # drive routine for access to Arnoldi-method based for finding eigenvalues
  #
  if haskey(dict,"report")
    println("====\nStarting Arnoldi iteration",
    " lam_tol=",dict["lam_tol"], " and ",dict["no_arnoldi_its"]," iterations.")
  end
  tic()
  #
  my_step,my_rhs_ball,dist_fn,ext_fn=setup_run_wos(p,dict)
  #
  its=arnoldi.k; eig_res=1.; delta_lam=1.; sep=1.
  while (its<dict["no_arnoldi_its"] 
    && eig_res>0
    #    && delta_lam>dict["lam_tol"]
    )
    
    its+=1
    # update rhs with interpolant
    smooth_interpolate!(G,dict,p,arnoldi.rhs,1.)
    my_rhs_ball=update_rhs(p,dict)
    # set tolerance
    if dict["relax"]==true && its>1
      relax_factor=max(1,sep/eig_res) 
    else
      relax_factor=1.
    end
    dict["tol"]=relax_factor*dict["lam_tol"]/(dict["no_arnoldi_its"]-1)/dict["B"]
    # WoS solve (inner iteration)
    out,=get_solution(p, dict, G)  
    # Arnoldi
    lambda,eig_res,sep=Add_Arnoldi!(arnoldi,out[G.int]) # sep=0.2 artificial
    lam_inv=1./lambda
    delta_lam=abs(lam_inv-lam_old); lam_old=lam_inv
    #
    if eig_res>0 && haskey(dict,"report")
      println("\nArnoldi it=",its, 
              ", lam_inv=",lam_inv, 
              ", eig_res=",eig_res,
              ", delta_lam=",delta_lam,", sep=",sep,".")
    end
  end
  lam,evec,all_lam,sep=Final_Arnoldi(arnoldi)
  println("All eigenvalues=",1./all_lam)
  el_time=toq()
  println("Elapsed time for wos_ev() is ", el_time)
  return (1./real(lam)),evec,delta_lam,arnoldi,el_time,eig_res
end
#
end# end of module kos_field_solve