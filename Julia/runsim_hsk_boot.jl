# Load required packages
using Random
using Statistics
using LinearAlgebra
using Distributions
using DataFrames
using CSV

# Tell Julia where to-be-called functions are
include("corrcov.jl")
include("DGPhsk.jl")
include("estimatelearning_hsk.jl")
include("estimatelearning_hsk_boot.jl")
include("initlearning_hsk.jl")
include("estimatelearningcore_hsk.jl")
include("draw_boot_sample.jl")
include("compute_boot_SE.jl")

# Set seed
guess=1;
Random.seed!(guess)

# Run DGP
ID,tvec,courselevel,numcourselevels,N,T,S,y,x,Choice,A,b,sig_ans,Delta_ans,DeltaCorr_ans = DGPhsk(7_034);
for j=1:7
    for c=1:4
        println(sum((Choice.==j) .& (courselevel.==c)))
    end
end


nboot = 3
# Estimate learning model
bstart,sig,Delta,Delta_corr,J = estimatelearning_hsk(ID,N,T,S,y,x,Choice,courselevel,numcourselevels); 
# Bootstrap to get SEs
bstart_se,sig_se,Delta_se,Delta_corr_se = estimatelearning_hsk_boot(nboot,ID,N,T,S,y,x,Choice,courselevel,numcourselevels); 
#nboot is an integer
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

Delta_se = reshape(Delta_se,(J,J))
Delta_corr_se = reshape(Delta_corr_se,(J,J))

# Export results
CSV.write("SimulationResultsHskBoot.csv", DataFrame([Delta Delta_se]), writeheader=false)
CSV.write("SimulationResultsHskBoot.csv", DataFrame([Delta_corr Delta_corr_se]), writeheader=false, append=true)
CSV.write("SimulationResultsHskBoot.csv", DataFrame([sig sig_se]), writeheader=false, append=true)
CSV.write("SimulationResultsHskBoot.csv", DataFrame([reshape(bstart[:],length(bstart[:]),1) bstart_se[:]]), writeheader=false, append=true)
