__precompile__()
# Set of helper routines for WOS fractional Laplacian
#
#
module KOSHELPER
#
using PyPlot
using Interpolations
using Dierckx
#
export init, MyTriMesh,get_tri_mesh,Final_Arnoldi, Add_Arnoldi!,Construct_Arnoldi,
Arnoldi,Matern,mat_cov,prior,new_data!,L2norm,L2L2norm,TheData,MyNormal,norm_gradient, dyda_compare, do_plot, Compare, smooth_interpolate!, smooth_interpolate2!
function __init__()
end
###################### begin: TriMesh 
type MyTriMesh
  x::Array{Float64,1}
  y::Array{Float64,1}
  mg_x::Array{Float64,2}
  mg_y::Array{Float64,2}
  hx::Float64
  hy::Float64
  nx::Int64
  ny::Int64
  n_int::Int64
  n_ext::Int64
  get_pt::Function
  get_pt_ext::Function
  get_pt_all::Function
  get_ext::Function
  reshape::Function
  evalfn::Function # evaluate (f,g) on TriMesh, f interior, g exterior.
  int::Array{Int64,1}
  ext::Array{Int64,1}
  get_levels::Function
  parents::Array{Int64,2}
  get_parents::Function
  finest_parents::Array{Int64,1}
  get_level_ind::Function
  move_to_TriMesh::Function 
  sq_int::Function
  sq_int2::Function
  scale::Function
end
###
function form_triangular_TriMesh2(pts_in,line0,tri0,no_levels)
  # pts_in, initial pts, e.g., 5 corners for square (inc centre)
  # tri0, initial triang as [n,3] matrix of point indices
  # assume triangles are well-ordered so [1,2,3]
  # means lines 1,2,3 are order [a,b][b,c][c,a],
  # where a,b,c are point numbers
  #
  level=zeros(Int64,no_levels); 
  level[1]=size(pts_in)[1];
  #
  ndim=8*4^(no_levels)
  pts=zeros(ndim,2);   pts[1:level[1],:]=pts_in
  parents=zeros(Int64,ndim,2)
  #
  cur_tri_no=size(tri0)[1]
  cur_lin_no=size(line0)[1]
  #
  for l=1:(no_levels-2)
    k=0;    line_no=0
    line=zeros(Int64,4*cur_lin_no,2)
    tri =zeros(Int64,4*cur_tri_no,3)
    tri_no=0
    midpoint=zeros(Int64,cur_lin_no)
    halflines=zeros(Int64,4*cur_lin_no,2)
    # divide lines to get new lines and points 
    for i=1:cur_lin_no # split lines
      pt_no=level[l]+i
      n1=line0[i,1]; n2=line0[i,2] # point numbers
      pts[pt_no,:]=(pts[n1,:]+pts[n2,:])/2
      # add 2 lines
      line[line_no+1,:]=[n1,   pt_no]; #  order is important
      line[line_no+2,:]=[pt_no,n2] # preserve clockwise ordering
      line_no+=2
      # record parents
      parents[pt_no,:]=[n1,n2]
      # need map from lines to midpts
      midpoint[i]=pt_no # line0 num to pt_num
      halflines[i,:]=[line_no-1, line_no] # map from cur_lin_no to two half line_no
    end
    new_pts=(1+level[l]):(level[l]+cur_lin_no)
    # loop over current triangles
    for k=1:cur_tri_no# 
      tri_line_no=tri0[k,:] # line0 numbers for this tri
      hl=halflines[tri_line_no,:]
      # sprintln("\nstart", midpoint[tri_line_no]," ",hl, " ", line[hl[:],:] )
      # for each successive pair of lines
      for i=1:3
        # common point from 
        common_ind=line0[tri_line_no[i],2] # need triangle is well-ordered
        j=mod(i, 3)+1
        assert(line0[tri_line_no[i],2]==line0[tri_line_no[j],1])
        # midpoints
        mid1=midpoint[tri_line_no[i]] # =pt_no for line i
        mid2=midpoint[tri_line_no[j]] # =pt_no for line j
        # add line
        line[line_no+1,:]=[mid2,mid1]
        line_no+=1
        # add trinagle
        # [mid2,mid1] [pt_no i,n2 i] [ n1 j, pt_no j]
        assert(line[line_no,2]==line[hl[i,2],1])
        assert(line[line_no,1]==line[hl[j,1],2])
        tri[tri_no+1,:]=[line_no, hl[i,2], hl[j,1]]
        tri_no+=1
      end
      # add the interior triangle
      tri[tri_no+1,:]=[line_no-2,line_no,line_no-1]
      tri_no+=1
    end
    # store number of lines
    level[l+1]=level[l]+cur_lin_no
    cur_lin_no=line_no
    line0=line
    tri0=tri
    cur_tri_no=tri_no
  end
  return pts,level,parents,line0,cur_lin_no
end
###
function get_tri_mesh(xy,no_levels::Int64,d)
  print("Creating triangular mesh...")
  # for plotting TriMesh
  nx=1+2^(no_levels-1); ny=nx
  #
  xTriMesh=linspace(xy[1],xy[2],nx)
  yTriMesh=linspace(xy[3],xy[4],ny)
  #
  hx=(xy[2]-xy[1])/(nx-1) 
  hy=(xy[4]-xy[3])/(ny-1)
  #
  mg_x = repmat(xTriMesh[:]',ny,1) # used for plotting
  mg_y = repmat(yTriMesh[:] ,1,nx)
  #
  pts0=zeros(5,2)
  pts0[1,:]=[xy[1],xy[3]]; pts0[2,:]=[xy[1],xy[4]];
  pts0[3,:]=[xy[2],xy[3]]; pts0[4,:]=[xy[2],xy[4]];
  pts0[5,:]=(pts0[1,:]+pts0[4,:])/2
  line0=[1 2; 1 3;  4 2; 4 3;        5 1; 2 5; 3 5;5 4 ]
  tri0=[1 6 5; 2 7 5; 7 8 4; 3 6 8] # 4x3
  #
  pts,level,parents,line,cur_lin_no=form_triangular_TriMesh2(pts0,
                                      line0, tri0,
                                      no_levels)
  qpts,qlevel,qparents,=form_triangular_TriMesh2(pts0,
                                      line0, tri0,
                                      no_levels+2)
  # add points on finest level to make a rectangular TriMesh
  pt_no=level[no_levels-1]
  for i=1:cur_lin_no
    n1=line[i,1]; n2=line[i,2]
    if (pts[n1,1]==pts[n2,1] || pts[n1,2]==pts[n2,2])
      pt_no+=1 
      pts[pt_no,:]=(pts[n1,:]+pts[n2,:])/2
      parents[pt_no,:]=[n1,n2]
    end
  end
  level[no_levels]=pt_no
  ndim=level[end]
  #
  ds=zeros(ndim); xTriMesh=deepcopy(ds); yTriMesh=deepcopy(ds)
  for i=1:level[no_levels]
    ds[i]=d(pts[i,:])[1]
    xTriMesh[i]=pts[i,1]; yTriMesh[i]=pts[i,2]
  end
  #
  int=find(x-> x>0, ds)[:];  n_int=length(int)
  ext=find(x-> x<=0,ds)[:];  n_ext=length(ext)
  #
  function get_pt_all(i) # return ith pt
    return [xTriMesh[i] yTriMesh[i]]
  end
  #
  function get_pt_ext(i) # return ith exterior pt
    return get_pt_all(ext[i])[:]
  end
  #
  function get_pt_int(i) # return ith interior point
    return get_pt_all(int[i])[:]
  end
  #
  function get_parents(i)
    if i>level[1] && i<=level[no_levels]
      return parents[i,:]
    else
      return [0,0]
    end
  end
  ############################
  # get_finest_parents
    r=(1+level[no_levels-1]):level[no_levels]
    pa=zeros(Int64,length(r),2)
    for i in r # update on finest interior
      j=i-level[no_levels-1]
      pa[j,:]     = parents[i,:]
    end 
    fpa=union(unique(pa[:]),r)
  ##############################
  #######################
  function get_ext(out,fn)# interior vector, ext fn
    # for i=1:n_int
    #      ri=int[i]
    #      #out[ri]=inm[i]
    # end
    for i=1:n_ext
       ri=ext[i]  
       out[ri]=fn(get_pt_ext(i))
    end
    return out
  end
  #######################
    function get_reshape(inm,fn)# interior vector, ext fn
    out=zeros(level[no_levels])
    for i=1:n_int
         ri=int[i]
         out[ri]=inm[i]
    end
    for i=1:n_ext
       ri=ext[i]  
       out[ri]=fn(get_pt_ext(i))
    end
    return out
  end
  ##############
  function get_fn(fin,fout)
    out=zeros(level[no_levels])
    for i=1:n_int
      ri=int[i]
      out[ri]=fin(get_pt_int(i))
    end
    for i=1:n_ext
      ri=ext[i]
      out[ri]= fout(get_pt_ext(i))
    end
    return out
  end
  ############################
  function get_levels(l)
    if (l<1 || l>no_levels)
      error("level number out of range")
    end
    r=1:level[l]
    out=zeros(2,level[l])
    out[1,:]=xTriMesh[r]
    out[2,:]=yTriMesh[r]
    return out# colums are pints
  end 
  #
  function only_level_ind(l)
    if (l>1)
      r=(1+level[l-1]):level[l]
    else
      r=1:level[1]
    end
    return r
  end
  # ###########################
  # function move_to_TriMesh(Z)
  #   s=Spline2D(xTriMesh,yTriMesh,Z; kx=1, ky=1, s=0.2)
  #   # on a TriMesh evaluate spline
  #   return evalTriMesh(s,mg_x[1,:],mg_y[:,1])
  # end
  ###########################
  function to_TriMesh(in)
    Z=zeros(nx,ny)
    for i=1:level[no_levels]
      #println( (pts[i,1]-xy[1])/hx ," ",(pts[i,2]-xy[3])/hy)
      j=1+Int64(round((pts[i,1]-xy[1])/hx ))
      k=1+Int64(round((pts[i,2]-xy[3])/hy ))
      #passed test!
      #println(pts[i,:], " ", mg_x[1,j], " ",mg_y[k,1])
      Z[j,k]=in[i]
    end 
    return Z
  end
  ################################
  function sq_int(Z,ell)    
    if ell<no_levels
      sum=quad_sq_tri(Z,ell)
    else
      H1=(xy[2]-xy[1])*(xy[4]-xy[3])/4
      H=H1*4.0^(2-ell)#area of square quarters at each level
      sum=quad_sq_TriMesh(to_TriMesh(Z),H)
    end
    return sum
  end
  ######################################
  function quad_sq_tri(Z, ell)
    # Let f be the pw linear interpolant
    # on triangles on level ell
    # compute int f^2 dx exact
    # note level ell=1 integrates the 5-point initial 4-triangle TriMesh
    # by looping over all level-2 points and working out averages of parents
    # (which are level-1 points)
    H1=(xy[2]-xy[1])*(xy[4]-xy[3])/4
    H=H1*(4.0^(1-ell))#triangle's areas quarters at each level
    sum=0.
    t = x-> (x[1]>xy[1] && x[1]<xy[2] && x[2]>xy[3] && x[2]<xy[4]) # test interior
    for i=(1+qlevel[ell]):qlevel[ell+1]# all points on level ell
      # average all parents
      pa=qparents[i,:]
      tv1=t(pts[pa[1],:]); tv2=t(pts[pa[2],:])
      if tv1 || tv2  
        sum+=mean(Z[pa]).^2*2 
      else
        sum+=Z[pa[1]]^2
      end 
    end
    return  sum*H*(1./3)
  end
  ##################
  function quad_sq_TriMesh(Z, H)
    # Let f be the  pw linear interpolant of
    # a TriMesh function Z (defined by  pyramid function on each square 
    # of the TriMesh using midpoint and average of each corner)
    # compute int f^2 \,dx exactly
    # H=size of square
    mysum=0.
    nx,ny=size(Z); v=zeros(4)
    for i=1:nx-1
      for j=1:ny-1
        x=i*hx+xy[1]; y=j+hy*xy[3];
        v[1]=Z[i,j];   v[2]=Z[i+1,j]
        v[3]=Z[i+1,j+1]; v[4]=Z[i,j+1]
        tmp=(v+circshift(v,[1]))/2
        c=sum(v.^2)+4*sum(tmp.^2)+16*mean(v)^2
        mysum+=c
      end
    end
    return mysum*H*(1/36)
  end
  ################
  function quad_sq_TriMesh2(Z_, f,ell)
    # Let f be the  pw linear interpolant of
    # a TriMesh function Z (defined by  pyramid function on each square 
    # of the TriMesh using midpoint and average of each corner)
    # compute int f^2 \,dx exactly
    # H=size of square
    H1=(xy[2]-xy[1])*(xy[4]-xy[3])/4
    H=H1*4.0^(2-ell)#area of square quarters at each level
    Z=to_TriMesh(Z_)
    mysum=0.
    nx,ny=size(Z); v=zeros(4)
    for i=1:nx-1
      for j=1:ny-1
        x=(i-1)*hx+xy[1]; y=(j-1)*hy+xy[3];
        apts=[[x, y], [x+hx, y], [x+hx, y+hy], [x, y+hy]]; vf=f.(apts)
        hpts=(apts+circshift(apts,[1]))/2; hf=f.(hpts)
        mpts=mean(apts); mf=f(mpts)
        v[1]=Z[i,j];   v[2]=Z[i+1,j]
        v[3]=Z[i+1,j+1]; v[4]=Z[i,j+1]
        tmp=(v+circshift(v,[1]))/2
        c=sum((v-vf).^2)+4*sum((tmp-hf).^2)+16*(mean(v)-mf)^2
        mysum+=c
      end
    end
    return mysum*H*(1/36)
  end
  #################################
  function scale1(ell) #scaling for norm
    return sqrt((xy[2]-xy[1])*(xy[4]-xy[3]))*2.0^(-ell)/2
  end
  println("......finished.")
  ##################################
  return MyTriMesh(xTriMesh, yTriMesh, mg_x, mg_y, hx, hy, nx,ny, n_int,n_ext,
   get_pt_int,get_pt_ext,get_pt_all,get_ext,get_reshape,get_fn,int,ext,
   get_levels,parents,get_parents, fpa,only_level_ind,
    to_TriMesh, sq_int,quad_sq_TriMesh2,scale1)
end
###################################################
###################################################
###################################################
function norm_gradient(G,itp)
  g=zeros(2)
  out=zeros(G.n_int)
  for i=1:G.n_int
    pt=G.get_pt(i)
    gradient!(g,itp,pt[1],pt[2])
    out[i]=norm(g)*sqrt(G.hx*G.hy)
  end
  return out
end
################# end: TriMesh
#########################################################
#########################################################
#########################################################

# function L2L2norm(dist::MyNormal, TriMesh::MyTriMesh)
#   mysum=trace(dist.C)*TriMesh.hx*TriMesh.hy
#   return sqrt(mysum)
# end
# #
# function L2norm(dist::MyNormal, TriMesh::MyTriMesh)
#   mysum=sum(dist.mu.^2)*TriMesh.hx*TriMesh.hy
#   return sqrt(mysum)
# end
###################### begin: Arnoldi
###
type Arnoldi # method
  KSS::Set # Krylov sub-space
  H::Array # H matrix
  k::Int32 # dimension of H matrix (k+1 -dim Krylov space)
  N::Int32 # dimension of state space
  rhs::Array{Float64,1} # for next solve
  ascale::Float64
end
#
function Construct_Arnoldi(x::Array{Float64,1})
  ascale=sqrt(length(x)*1.0)
  xl=norm(x)
  tmp=x/xl
  S=Set()
  push!(S,[1,tmp])
  return Arnoldi(S,Array{Float64}(1,1),0,length(x),tmp*ascale,ascale)
end

######################
function SmartRestart!(arnoldi_data::Arnoldi,p)
  # Reduce the Arnoldi matrix using implicitly shifted Arnoldi
  # n denotes the dimension of Arnoldi H-matrix (n+1 dim Krylov basis)
  # reduced to dimension  p=n-length(mu) dim H-matrix
  #
  k=arnoldi_data.k # dimension of H-matrix
  if (k>p)
    println("Implicit restart: reducing dimension ", k," to ", p,".")
    rhs=arnoldi_data.rhs*arnoldi_data.H[k+1,k]/arnoldi_data.ascale
    # get Ritz values of H (for shifts)
    H=arnoldi_data.H[1:k,1:k]
    d=eigvals(H) #
    d=sort(d,by=abs)#ordered with increasing magnitue
    # number of shifts required is k-p
    # so throw  away p smallest evalues
    # mu is empty if p \geq k.%
    mu=d[1:(k-p)] # 
    println("Ritz values=", d, " and using shifts=", mu)
    # initialise and do implicit shifts
    # load matrix V with first-k Krylov vectors
    V=zeros(arnoldi_data.N, k)
    for vs in arnoldi_data.KSS
      if (vs[1]<=k)
        V[:,vs[1]]=vs[2]
     end
    end # 
    # initialise values of v
    v=zeros(k); v[end]=1
    # loop over shifts
    for i=1:length(mu)
      Q,R=qr(H-mu[i]*eye(k))
      H=Q'*H*Q # ' is already Hermitian transpose
      V=V*Q;      v=Q'*v
    end
    # define p-dimensional H matrix
    arnoldi_data.H=zeros(p+1,p+1) # yes it has dim p+1
    arnoldi_data.H[1:p,1:p]=H[1:p,1:p] # copy from above
    arnoldi_data.k=p # new dimension
    empty!(arnoldi_data.KSS) # Krylov space
    for i=1:p
      push!(arnoldi_data.KSS,[i,V[:,i]])
    end # 
    # last Krylov vector and new rhs
    z=V[:,p+1]*H[p+1,p]+rhs*v[p]; norm_z=norm(z); z=z/norm_z
    push!(arnoldi_data.KSS,[p+1,z])
    arnoldi_data.H[p+1,p]=norm_z # next rhs
    arnoldi_data.rhs=z*arnoldi_data.ascale
  end 
end
#######################
function Add_Arnoldi!(arnoldi_data::Arnoldi,zin::Array{Float64,1})
  # arnoldi_data (Krylov Subspace as array of [k,matrix], each matrix of unit F-norm, size of psace, 
  # current H matrix)
  # initialise Arnoldi  with (v,[0],0) where v is unit rhs and data is solve of Ax=rhs 
  k=arnoldi_data.k+1
  H=deepcopy(arnoldi_data.H)
  z=deepcopy(zin)/arnoldi_data.ascale
  # H should be k x k matrix already
  for vs in arnoldi_data.KSS,
    H[vs[1],k]=vecdot(z,vs[2])
    z=z-H[vs[1],k]*vs[2]
  end # end Gram-Schmidt
  d,v=eig(arnoldi_data.H[1:k,1:k])
  perm=sortperm(d,by=abs);  d=d[perm] #ordered with increasing magnitue
  ev_max=d[end]  # largest eigenvalue
  if k>1
    sep=abs(ev_max-d[end-1]) # spectral gap
  else
    sep=1.0
  end
  #
  value=0.
  # increase size of H
  arnoldi_data.H=zeros(Float64,k+1,k+1)
  arnoldi_data.H[1:k,1:k]=H
  arnoldi_data.H[k+1,k]=norm(z)
  #
  z=z/arnoldi_data.H[k+1,k]
  push!(arnoldi_data.KSS,[k+1,z])
  arnoldi_data.k=k
  #
  ev_max, evec,all_lam,sep=Final_Arnoldi(arnoldi_data)
  value=abs(arnoldi_data.H[k+1,k])*abs(evec[k])
  arnoldi_data.rhs=z*arnoldi_data.ascale
  # eigenvalue approx, new RHS
  return ev_max,  value*arnoldi_data.ascale, sep
end
#
function Final_Arnoldi(arnoldi_data::Arnoldi)
  # get eigenfunction aswell and return
  k=arnoldi_data.k
  d,v=eig(arnoldi_data.H[1:k,1:k])
  perm=sortperm(d,by=abs)#ordered with increasing magnitue
  d=d[perm]; lam=d[end]; ev=v[:,perm[end]]; #largest eigenvalue 
  if k>1
    sep=abs(lam-d[end-1]) # spectral gap
  else
    sep=1.
  end
  evec=zeros(arnoldi_data.N)
  for vs in arnoldi_data.KSS
    if (vs[1]<=length(ev))
      evec=evec+ev[vs[1]]*vs[2]
    end 
  end # end Gram-Schmidt
  return lam,evec,d,sep
end
# end: Arnoldi
########################
function dyda_lam(alp,d=2)
  # Dyda lower bound for smallest eigenvalue 
  lam=2^alp*gamma(alp/2+1)*gamma((alp+d)/2)*(alp+2)*(alp+d)*(6-alp)/gamma(d/2)/(12*d+(16-2*d)*alp)
  return lam
end
########################
function dyda_lam_up(alp)
  data=[1.05096, 1.10993, 1.34374, 2.00612, 3.27594, 4.56719, 5.13213]
  valp=([0.1, 0.2, 0.5, 1, 1.5, 1.8, 1.9],)
  itp = interpolate(valp,data, Gridded(Linear()))
  return itp[alp] 
end
##
function dyda_compare(alpha, lam)
  lam=real(lam)
  if (lam>dyda_lam(alpha) && lam<dyda_lam_up(alpha))
     println("   >>>> eigenvalue in Dyda's interval <<<<")
  else
    println("Outside Dyda's interval=(",dyda_lam(alpha),",",dyda_lam_up(alpha),").")  
  end
  println("Relative error of lam=",lam," is ",
      min(abs(dyda_lam(alpha)-lam), abs(dyda_lam_up(alpha)-lam))/lam)
end


###################### begin: plotting
function do_plot(TriMesh::MyTriMesh,Z,mytitle,filename)

  PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
  PyPlot.matplotlib[:rc]("xtick", labelsize=7) 
  PyPlot.matplotlib[:rc]("axes", labelsize=8)
  PyPlot.matplotlib[:rc]("axes", labelpad=1)
 ticklabel_format(style="sci")
  fig = figure("pyplot_surfaceplot",figsize=(4.5,3.5))
  clf()
  ax = fig[:add_subplot](2,1,1, projection = "3d")
  ax[:plot_surface](TriMesh.mg_x, TriMesh.mg_y, Z, edgecolors="k", 
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  xlabel(L"x");  ylabel(L"y"); title(mytitle)
  fig[:canvas][:draw]
  tmp=gcf()
  locator_params(nbins=5)
  ticklabel_format(style="sci",axis="z",scilimits=(0,0))
  tmp[:set_size_inches](4.7,3.5)
  savefig(filename)#,bbox_inches="tight")
end
#
function do_plot(TriMesh::MyTriMesh,Z1,Z2,mytitle1,mytitle2) 
  fig = figure("pyplot_surfaceplot",figsize=(10,10))
  #
  ax = fig[:add_subplot](2,1,1, projection = "3d")
  ax[:plot_surface](TriMesh.mg_x, TriMesh.mg_y, Z1, edgecolors="k", 
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  xlabel("X")
  ylabel("Y")
  title(mytitle1)
  #
  ax = fig[:add_subplot](2,1,2, projection = "3d")
  ax[:plot_surface](TriMesh.mg_x, TriMesh.mg_y, Z2, edgecolors="k", 
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  xlabel("X")
  ylabel("Y")
  title(mytitle2)
  #
  tmp=fig[:canvas][:draw]
  tmp=gcf()
end
###
function Compare(G, f1, Z, N,mytitle1,mytitle2)
  gx=linspace(G.x[1],G.x[end],N)
  gy=linspace(G.y[1],G.y[end],N)
  mg_x = repmat(gx[:]',N,1)
  mg_y = repmat(gy[:],1,N)
  itp = interpolate((G.x,G.y),Z,Gridded(Linear()))
  f1v=zeros(N,N); f2v=zeros(N,N)
  for i=1:N
    for j=1:N
      if norm([gx[i],gy[j]])<1
        f1v[i,j]=f1([gx[i],gy[j]])
      end 
      f2v[i,j]=itp[gx[i],gy[j]]
    end
  end
  fig = figure("pyplot_surfaceplot",figsize=(10,10))
  #
  ax = fig[:add_subplot](2,1,1, projection = "3d")
  ax[:plot_surface](mg_x, mg_y, f2v, edgecolors="k", 
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  xlabel("X")
  ylabel("Y")
  title(mytitle1)
  #
  ax = fig[:add_subplot](2,1,2, projection = "3d")
  ax[:plot_surface](mg_x, mg_y, abs(f1v-f2v), edgecolors="k", 
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
  xlabel("X")
  ylabel("Y")
  title(mytitle2)
  #
  fig[:canvas][:draw]
  gcf()
end
# end: plotting


# ######################
function smooth_interpolate!(G,dict,p,data_in,sig_sq)
  N=G.n_int
  new_rhs_full=G.reshape(data_in,x->0.);  Z=G.move_to_TriMesh(new_rhs_full)
  if (true) # linear interpolant
    # Interpolations module
      itp=interpolate((G.mg_x[1,:],G.mg_y[:,1] ),
                      Z,
                      Gridded(Linear()))
      p.fn_rhs=x->[itp[x[i,1],x[i,2]] for i=1:size(x)[1]]#*max.(0,sign.(p.d(x)))
  else # spline interpolant
      itp0=interpolate(Z,BSpline(Cubic(Line())), OnTriMesh())
      itp = scale(itp0, G.x, G.y)
      p.fn_rhs=x->itp[x[1],x[2]]
  end
  if haskey(dict,"extra_plot")
    do_plot(G,Z,"rhs")
  end
end
######################
function smooth_interpolate2!(G,dict,p,data_in,sig_sq)
  prior=deepcopy(dict["prior"])
  N=G.n_int;   data=TheData(zeros(N),zeros(N),zeros(N))
  data.mu=data_in; 
  #println(data_in) good
  #println(sig_sq)good
  data.var=ones(data_in)*sig_sq
  data.no=ones(Int32,N)
  if (false)
    new_data!(prior, data)
    #println(prior.mu)#bad
    new_rhs_full=G.reshape(prior.mu,KOS.fn_zero) # full object 
  else
    for i=1:G.n_int
      data_in[i]/=p.d(G.get_pt(i))
    end 
    new_rhs_full=G.reshape(data_in,x->0.) 
  end
  #println(new_rhs_full)bad 
 
  Z=G.move_to_TriMesh(new_rhs_full)
  itp=interpolate((G.mg_x[1,:],G.mg_y[:,1] ),
                      Z,
                      Gridded(Linear()))
  #p.fn_rhs=x->itp[x[1],x[2]]*max(0,p.d(x))
  p.fn_rhs=x->max(0,p.d(x))
  do_plot(G,Z,"rhs")
end
######################
#
end # module KOSHELPER 