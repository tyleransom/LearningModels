function estimatelearning_hsk(ID,N,T,S,y,x,Choice,clevel,ncl)

    # PTypesl   = PTypes(:)
    # PTypeg    = PType(Cl==1)
    # PTypen    = PType(Cl==2)
    # PType4s   = PType(Cl==3)
    # PType4h   = PType(Cl==4)
    # PType2    = PType(Cl==5)

    iteration = 1
    J = convert(Int64,length(unique(Choice))-1)

    Delta = rand(J,J)
    Delta = 0.5.*(Delta+Delta')
    @assert Delta == Delta'

    sig = rand(J*ncl)

    BigN,bstart,resid,residsub,abilsub,Resid,Csum,Csumw,tresid,isig,Psi1,sig2,cov2,sigtemp,covtemp = initlearning_hsk(N,T,J,Choice,y,x,clevel,ncl)
    for j=1:J
        for c=1:ncl
            isig[vec((y[:,j].!=999) .& (clevel.==c)),j] = 1/sig[(j-1)*ncl+c]*ones(sum(vec((y[:,j].!=999) .& (clevel.==c))))
        end
    end

    j=1

    while maximum(maximum(abs.(covtemp.-Delta)))>1e-4
        sigtemp=sig
        covtemp=Delta
        
        bstart,Delta,sig,Resid,resid,residsub,tresid,isig,abilsub = estimatelearningcore_hsk(y,x,Choice,clevel,ncl,bstart,N,T,S,J,BigN,Delta,sig,Resid,resid,residsub,tresid,Psi1,Csum,Csumw,isig,abilsub)
        
        sig2[j,:]=sig'
        cov2[j,:]=LowerTriangular(Delta)[LowerTriangular(Delta).!=0]
        
        println(j)
        println(maximum(maximum(abs.(covtemp.-Delta))))
        j+=1
    end
    Delta_corr = corrcov(Delta)

    return bstart,sig,Delta,Delta_corr,J
end
