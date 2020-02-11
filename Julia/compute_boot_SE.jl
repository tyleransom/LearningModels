function compute_boot_SE(nboot,x)
    return sqrt.(diag((1/(nboot-1))*(x.-repeat(mean(x;dims=2),outer=[1,nboot]))*(x.-repeat(mean(x;dims=2),outer=[1,nboot]))'))
end
