# Walk outside spheres  for the fractional Laplacian
# 

This package provides a Julia implementation of the algorithms described
in

> T. Shardlow. A walk outside spheres for the fractional Laplacian: 
> fields and first eigenvalue. 2018

## Installation

Put KOSHELPER.jl, KOS.jl, and KOS_FIELD_SOLVE.jl in a directory
in the LOAD_PATH.

## Getting started

Start additional processors if available, using add_procs(n).

Look at the examples in julia_wos.jl, e.g,.

> include("julia_wos.jl");
> alpha=1.0; eg_num=1; plot_flag=true;
> run_field(alpha,eg_num,plot_flag)

gives the field solve and plot for the first example.

Minimal-working examples, showing how to define f and g and d, etc., for
a field solve is given in eg_field.jl; type

> include("eg_field.jl"); alpha=1.0; mwe(alpha)

and for an eigenvalue solve in eg_ev.jl; type

> include("eg_ev.jl"); alpha=1.0; mwe2(alpha)

The original algorithm of Kyprianou, Osojnik, and Shardlow for finding the
solution at a single point is defined in run_pt()

> include("julia_wos.jl"); alpha=1.0; x0=[0.2,0.3]; run_pt(alpha,x0)