function initlearning_hsk(N,T,J,Choice,y,x,clevel,ncl)

    bstart   = zeros(size(x,2),J)
    resid    = Array{Float64}[]
    residsub = Array{Float64}[]
    abilsub  = Array{Float64}[]
    Resid    = zeros(size(y,1),J*ncl)
    bigN     = zeros(Int64,J*ncl)
    Csumw    = zeros(N*S,J*ncl)
    Csum     = zeros(N*S,J)
    tresid   = zeros(N*S,J)
    isig     = ones(N*T*S,J)
    for j=1:J
        flag                     = y[:,j].!=999
        bstart[:,j]              = x[flag,:,j]\y[flag,j]
        push!(resid  ,y[flag,j].-x[flag,:,j]*bstart[:,j])
        Csum[:,j]                = (sum(reshape(Choice.==j,(T,N*S));dims=1))'
        tresid[:,j]              = (sum(reshape(Resid[:,j],(T,N*S));dims=1))'
        Resid[vec(Choice.==j),j] = resid[j]
        for c=1:ncl
            push!(abilsub,y[vec((flag) .& (clevel.==c)),j].-x[vec(flag .& (clevel.==c)),:,j]*bstart[:,j])
            push!(residsub,y[vec((flag) .& (clevel.==c)),j].-x[vec(flag .& (clevel.==c)),:,j]*bstart[:,j])
            bigN[(j-1)*ncl+c]      = sum((Choice.==j) .& (clevel.==c)) # later sum needs to be weighted by PTypesl
            Csumw[:,(j-1)*ncl+c]   = (sum(reshape((Choice.==j) .& (clevel.==c),(T,N*S));dims=1))'
        end
    end
    
    abil = tresid./(Csum.+eps())
    abil = kron(abil,ones(T,1))

    Psi1 = deepcopy(Csumw)
    
    for j=1:J
        resid[j]               .+= abil[vec(Choice.==j),j]
        Resid[vec(Choice.==j),j] = resid[j]
    end
    
    sig2=zeros(500,J*ncl)
    cov2=zeros(500,convert(Int64,(J+1)*J/2))

    sigtemp = zeros(J*ncl,1)
    covtemp = zeros(J)

    return bigN,bstart,resid,residsub,abilsub,Resid,Csum,Csumw,tresid,isig,Psi1,sig2,cov2,sigtemp,covtemp
end
