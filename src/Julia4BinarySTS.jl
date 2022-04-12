module Julia4BinarySTS

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using BenchmarkTools
using Profile
using MAT
using SymPy
using Plots; gr(fmt=:png)
#using UnicodePlots
#using PyPlot; pygui(true)
#using Images
using JLD
using DiffEqOperators
#using DiffEqParamEstim
#using DiffEqDevTools
using StaticArrays
using ODE
using ODEInterface
using ODEInterfaceDiffEq
#using MATLABDiffEq
#using LSODA
#using SciPyDiffEq
#using deSolveDiffEq
#using ModelingToolkit
#using SparsityDetection
#using SparseArrays
#using AlgebraicMultigrid
#using Sundials
using Test
#using Distributed
#using ParameterizedFunctions
using PolynomialRoots
using Polynomials
using Roots
#addprocs()
#@everywhere using DifferentialEquations
#using Unitful
#using PhysicalConstants.CODATA2014
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using DiffEqSensitivity, DifferentialEquations
using Zygote
using ForwardDiff
using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)
using Random
using Statistics
using LatinHypercubeSampling
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load

using DiffResults
using DataFrames
using CSV
using Dierckx
using StatsPlots
using LaTeXStrings

Random.seed!(666);

is_restart = false;
n_epoch = 1000000;
n_plot = 50;

opt = ADAMW(0.005, (0.9, 0.999), 1.f-6);
datasize = 40;
batchsize = 32;
n_exp_train = 20;
n_exp_val = 5;
n_exp = n_exp_train + n_exp_val;
noise = 1.f-4;
ns = 3;
nr = 6;

grad_max = 10^(1);
maxiters = 10000;
################################################

# Switch for mixture species: 1 - N2/N; 2 - O2/O
const sw_sp = 1

# Switch for model oscillator: 1 - anharmonic; 2 - harmonic
const sw_o = 1

# Species molar fractions [xA2 xA] (xc = nc/n)
const xc = [1 0]

# Range of integration
const x_w = 100.

# Physical constants
const c     = 2.99e8     # SpeedOfLightInVacuum
const h     = 6.6261e-34 # PlanckConstant
const k     = 1.3807e-23 # BoltzmannConstant
const N_a   = 6.0221e23  # AvogadroConstant
const Runi  = 8.3145     # MolarGasConstant
const h_bar = h/(2*pi)   # PlanckConstantOver2pi

# To read a single variable from a MAT file
# (compressed files are detected and handled automatically)
file  = matopen("data_species.mat")
const OMEGA = read(file, "OMEGA") # Ω
const BE    = read(file, "BE")    # Bₑ
const ED    = read(file, "ED")
const CArr  = read(file, "CArr")
const NArr  = read(file, "NArr")
const QN    = read(file, "QN")
const MU    = read(file, "MU")    # μ
const R0    = read(file, "R0")    #
const EM    = read(file, "EM")
const RE    = read(file, "RE")
close(file)

const om_e   = OMEGA[sw_sp,1]
const om_x_e = OMEGA[sw_sp,2]
const Be     = BE[sw_sp]
const D      = ED[sw_sp]
const CA     = CArr[sw_sp,:]
const nA     = NArr[sw_sp,:]
const l      = Int(QN[sw_sp,sw_o])

include("en_vibr.jl")
const e_i = en_vibr()

include("en_vibr_0.jl")
const e_0 = en_vibr_0()

const mu     = [MU[sw_sp] 0.5*MU[sw_sp]]*1e-3
const m      = mu / N_a
const sigma0 = pi*R0[sw_sp,1]^2
const r0     = [R0[sw_sp,1] 0.5*(R0[sw_sp,1]+R0[sw_sp,2])]
const em     = [EM[sw_sp,1] sqrt(EM[sw_sp,1]*EM[sw_sp,2]*R0[sw_sp,1]^6*R0[sw_sp,2]^6)/r0[2]^6]
const re     = RE[sw_sp]

# ICs
const p0  = 0.8*133.322
const T0  = 300.
const Tv0 = T0
const M0  = 13.4
const n0  = p0/(k*T0)

if xc[1] != 0
  const gamma0 = 1.4
else
  const gamma0 = 5/3
end

const rho0_c = m.*xc*n0
const rho0   = sum(rho0_c)
const mu_mix = sum(rho0_c./mu)/rho0
const R_bar  = Runi*mu_mix
const a0     = sqrt(gamma0*R_bar*T0)
const v0     = M0*a0

include("in_con.jl")
NN = in_con()
n1 = NN[1]
v1 = NN[2]
T1 = NN[3]

const Zvibr_0 = sum(exp.(-e_i./Tv0./k))

Y0_bar      = zeros(Float64, (l+3))
Y0_bar[1:l] = xc[1]*n1/Zvibr_0*exp.(-e_i./Tv0./k)
Y0_bar[l+1] = xc[2]*n1
Y0_bar[l+2] = v1
Y0_bar[l+3] = T1

const Delta = 1/(sqrt(2)*n0*sigma0)
xspan       = Float64[0.0, x_w]./Delta

#xspan = Float32[0.0, datasize * tstep];
#xsteps = range(xspan[1], xspan[2], length=datasize);

#A  = Matrix{Float64}(undef, l+3, l+3)
#AA = Matrix{Float64}(undef, l+3, l+3)

#ni_b = Vector{Float64}(undef, l)

#kd = Array{Float64}(undef, 2, l)
#kr = Array{Float64}(undef, 2, l)

#kvt_down = Array{Float64}(undef, 2, l-1)
#kvt_up   = Array{Float64}(undef, 2, l-1)

#kvv_down = Array{Float64}(undef, l-1, l-1)
#kvv_up   = Array{Float64}(undef, l-1, l-1)

#RD  = Vector{Float64}(undef, l)
#RVT = Vector{Float64}(undef, l)
#RVV = Vector{Float64}(undef, l)

#B = Vector{Float64}(undef, l+3)

# https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/
include("kdis.jl")
include("kvt_ssh.jl")
include("kvv_ssh.jl")
include("rpart.jl")
# https://diffeq.sciml.ai/v1.10/basics/common_solver_opts.html
# https://discourse.julialang.org/t/handling-instability-when-solving-ode-problems/9019/5
# https://discourse.julialang.org/t/differentialequations-what-goes-wrong/30960

#alg  = AutoTsit5(Rosenbrock23(autodiff=false));
alg = radau()
#alg = Tsit5()
#alg = CVODE_BDF(linear_solver=:GMRES)
#alg = alg_hints=[:stiff]
#alg = BS3()

prob = ODEProblem(rpart!, Y0_bar, xspan, 1.)
#display(@benchmark DifferentialEquations.solve(prob, alg, reltol=1e-8, abstol=1e-8, save_everystep=true, progress=true))
sol  = DifferentialEquations.solve(prob, alg, reltol=1e-8, abstol=1e-8, save_everystep=true, progress=true)

#@profile DifferentialEquations.solve(prob, alg, reltol=1e-4, abstol=1e-4, save_everystep=true, progress=true)
#Juno.profiler()
#Profile.clear()

ode_data = Array(sol)

X       = sol.t
x_s     = X*Delta*100
Temp    = sol[l+3,:]*T0
v       = sol[l+2,:]*v0
n_i     = sol[1:l,:]*n0
n_a     = sol[l+1,:]*n0
n_m     = sum(n_i,dims=1)
time_s  = X*Delta/v0
Npoint  = length(X)
Nall    = sum(n_i,dims=1)
Nall    = Nall[1,:]+n_a
#ni_n   = n_i ./ repeat(Nall,1,l) BUG
#nm_n   = sum(ni_n,dims=2)
na_n    = n_a ./ Nall
#rho    = m[1]*n_m + m[2]*n_a
#p      = Nall .* k .* Temp
#e_v    = repeat(e_i+e_0,Npoint,1) .* n_i
#e_v    = repeat(e_i+e_0,1,Npoint) .* n_i
#e_v    = sum(e_v,dims=2)
#e_v0   = n0*xc[1]/Zvibr_0*sum(exp.(-e_i./Tv0/k) .* (e_i+e_0))
#e_f    = 0.5*D*n_a*k
#e_f0   = 0.5*D*xc[2]*n0*k
#e_tr   = 1.5*Nall*k .* Temp
#e_tr0  = 1.5*n0*k .* T0
#e_rot  = n_m*k .* Temp
#e_rot0 = n0*xc[1]*k .* T0
#E      = e_tr+e_rot+e_v+e_f
#E0     = e_tr0+e_rot0+e_v0+e_f0
#H      = (E+p) ./ rho
#H0     = (E0+p0) ./ rho0
#u10    = rho0*v0
#u20    = rho0*v0^2+p0
#u30    = H0+v0^2/2
#u1     = u10-rho .* v
#u2     = u20-rho .* v.^2-p
#u3     = u30-H-v.^2/2
#d1     = max(abs(u1)/u10)
#d2     = max(abs(u2)/u20)
#d3     = max(abs(u3)/u30)
#
#display("Relative error of conservation law of:");
#println("mass = ", d1);
#println("momentum = ", d2);
#println("energy = ", d3);

#display(Plots.plot(sol))

display(Plots.plot(x_s,Temp))
savefig("T.pdf")

display(Plots.plot(x_s,v))
savefig("V.pdf")

#display(Plots.plot!(x_s,n_i))
#savefig("n_i.pdf")

#display(Plots.plot!(x_s,n_a))
#savefig("n_a.pdf")

# Load Matlab solution
#file  = matopen("solution.mat")
#const X_    = read(file, "X");
#const x_s_  = read(file, "x_s");
#const Temp_ = read(file, "Temp");

#include("rpart_post.jl")
#RDm  = zeros(Npoint,l);
#RDa  = zeros(Npoint,l);
#RVTm = zeros(Npoint,l);
#RVTa = zeros(Npoint,l);
#RVV  = zeros(Npoint,l);
#
#for i = 1:Npoint
#  input = Y[i,:]
#  rdm, rda, rvtm, rvta, rvv = rpart_post(input)
#  RDm[i,:]  = rdm
#  RDa[i,:]  = rda
#  RVTm[i,:] = rvtm
#  RVTa[i,:] = rvta
#  RVV[i,:]  = rvv
#end
#
#RD_mol = RDm+RDa
#RVT    = RVTm+RVTa
#RD_at  = -2*sum(RD_mol,2)

end
