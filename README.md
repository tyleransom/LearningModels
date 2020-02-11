# LearningModels
Code to estimate correlated learning models in Julia

Code in the `/Julia` folder contains simulations for three different specifications:

1. J-dimensional ability vector that is unknown to the individual; idiosyncratic shocks are homoskedastic
2. Same as (1) but idiosyncratic shocks are allowed to be heteroskedastic
3. Run (2) in bootstrap to obtain standard errors

Note that in none of the above cases is there any assumption of unobserved heterogeneity (known to the individual but not the reasearcher).

Files to run the three specifications above are located in the `/Julia` folder at:

1. `runsim.jl`
2. `runsim_hsk.jl`
3. `runsim_hsk_boot.jl`


