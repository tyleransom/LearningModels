function estimatelearning_hsk_boot(nboot,ID,N,T,S,y,x,Choice,clevel,ncl)

    # PTypesl   = PTypes(:)
    # PTypeg    = PType(Cl==1)
    # PTypen    = PType(Cl==2)
    # PType4s   = PType(Cl==3)
    # PType4h   = PType(Cl==4)
    # PType2    = PType(Cl==5)

    J = convert(Int64,length(unique(Choice))-1)

    bstartBootMat    = zeros(size(bstart,1)*size(bstart,2),nboot)
    sigBootMat       = zeros(J*ncl,nboot)
    DeltaBootMat     = zeros(J*J,nboot)
    DeltaCorrBootMat = zeros(size(DeltaBootMat))

    # loop over each bootstrap sample
    for bb=1:nboot

        # now draw the bootstrapped data
        bootidx = draw_boot_sample(ID,N)
        yb      = y[bootidx,:]
        xb      = x[bootidx,:,:]
        Choiceb = Choice[bootidx,:]
        clevelb = clevel[bootidx,:]

        # do the following code for each bootstrap replicate
        iteration = 1

        Delta = rand(J,J)
        Delta = 0.5.*(Delta+Delta')
        @assert Delta == Delta'

        sig = rand(J*ncl)

        BigN,bstart,resid,residsub,abilsub,Resid,Csum,Csumw,tresid,isig,Psi1,sig2,cov2,sigtemp,covtemp = initlearning_hsk(N,T,J,Choiceb,yb,xb,clevelb,ncl)
        for j=1:J
            for c=1:ncl
                isig[vec((yb[:,j].!=999) .& (clevelb.==c)),j] = 1/sig[(j-1)*ncl+c]*ones(sum(vec((yb[:,j].!=999) .& (clevelb.==c))))
            end
        end

        j=1

        while maximum(maximum(abs.(covtemp.-Delta)))>1e-4
            sigtemp=sig
            covtemp=Delta
            
            bstart,Delta,sig,Resid,resid,residsub,tresid,isig,abilsub = estimatelearningcore_hsk(yb,xb,Choiceb,clevelb,ncl,bstart,N,T,S,J,BigN,Delta,sig,Resid,resid,residsub,tresid,Psi1,Csum,Csumw,isig,abilsub)
            
            sig2[j,:]=sig'
            cov2[j,:]=LowerTriangular(Delta)[LowerTriangular(Delta).!=0]
            j+=1
        end
        Delta_corr = corrcov(Delta)

        # collect results
        bstartBootMat[:,bb]    = bstart[:]
        sigBootMat[:,bb]       = sig
        DeltaBootMat[:,bb]     = Delta[:]
        DeltaCorrBootMat[:,bb] = Delta_corr[:]
    end

    # now compute standard errors
    bstart_se     = compute_boot_SE(nboot,bstartBootMat)
    sig_se        = compute_boot_SE(nboot,sigBootMat)
    Delta_se      = compute_boot_SE(nboot,DeltaBootMat)
    Delta_corr_se = compute_boot_SE(nboot,DeltaCorrBootMat)

    return bstart_se,sig_se,Delta_se,Delta_corr_se
end
