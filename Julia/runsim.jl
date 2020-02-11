# Load required packages
using Random
using Statistics
using LinearAlgebra
using DataFrames
using CSV

# Tell Julia where to-be-called functions are
include("corrcov.jl")
include("DGP.jl")
include("estimatelearning.jl")
include("initlearning.jl")
include("estimatelearningcore.jl")

# Set seed
guess=1;
Random.seed!(guess)

# Run DGP
ID,tvec,N,T,S,y,x,Choice,A,b,sig_ans,Delta_ans,DeltaCorr_ans = DGP(10_000);

# Estimate learning model
bstart,sig,Delta,DeltaCorr = estimatelearning(ID,N,T,S,y,x,Choice); 
#ID is a N*T x 1 vector, 
#N is an integer,
#T is an integer,
#S should be set to 1 for now
#y is a N*T x J matrix of outcomes (with NaNs where outcomes are unobserved),
#x is a N*T x K x J covariate tensor
#Choice is a N*T x 1 vector of integers (1, ..., J+1) where J+1 refers to "not in school"

# Compare estimates with true parameter values
println(Delta)
println(Delta_ans)
println(sig)
println(sig_ans)

# Export results
CSV.write("SimulationResults.csv", DataFrame(Delta), writeheader=false)
CSV.write("SimulationResults.csv", DataFrame(DeltaCorr), writeheader=false, append=true)
CSV.write("SimulationResults.csv", DataFrame(sig), writeheader=false, append=true)
CSV.write("SimulationResults.csv", DataFrame(reshape(bstart[:],length(bstart[:]),1)), writeheader=false, append=true)
