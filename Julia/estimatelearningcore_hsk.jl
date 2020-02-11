function estimatelearningcore_hsk_het(y,x,Choice,clevel,ncl,bstart,N,T,S,J,BigN,Delta,sig,Resid,resid,residsub,tresid,Psi1,Csum,Csumw,isig,abilsub)
    abil=zeros(N*S,J)
    
    idelta   = inv(Delta)
    vtemp2   = zeros(J,J)
    vabil    = zeros(N*S,J)
    vabilw   = zeros(N*S,J)
    for j=1:J
        tresid[:,j] = (sum(reshape(Resid[:,j].*isig[:,j],(T,N*S));dims=1))'
    end
    
    for i=1:S*N
        psit=Psi1[i,:]
        Psi = zeros(J,J)
        for j=1:J
            Psi[j,j] = psit[(j-1)*ncl+1:j*ncl]⋅(1 ./sig[(j-1)*ncl+1:j*ncl])
        end
        
        vtemp=inv(idelta.+Psi)
        
        vectres = zeros(J)
        for j=1:J
            vectres[j] = tresid[i,j]
        end
        temp=(vtemp*vectres)'
        abil[i,:]=temp
        
        for j=1:J
            vabil[i,j] = vtemp[j,j] # later need PTypesl[i]*vtemp[j,j]
        end
        
        vtemp2.+=(vtemp.+temp'*temp) # later need PTypesl[i]*(vtemp+temp'*temp)
    end
    
    Delta=vtemp2./N
    
    vabilw    = kron(vabil,ones(1,ncl))
    Abil      = kron(abil,ones(T,1))
    Vabil     = kron(vabil,ones(T,1))
    
    abil1  = Array{Float64}[]
    sigdem = zeros(J*ncl)
    for j=1:J
        push!(abil1,Abil[vec(Choice.==j),j])
        for c=1:ncl
            abilsub[(j-1)*ncl+c] = Abil[vec((Choice.==j) .& (clevel.==c)),j]
            sigdem[(j-1)*ncl+c]  = sum((residsub[(j-1)*ncl+c].-abilsub[(j-1)*ncl+c]).^2) # later need sum(PTypesub[j].*(resid[j].-abilsub[j]).^2) 
        end
    end
    
    sig=((sum(Csumw.*vabilw;dims=1))'.+sigdem)./BigN
    
    for j=1:J
        flag                      = (y[:,j].!=999)
        bstart[:,j]               = x[flag,:,j]\(y[flag,j].-abil1[j]) # later weight by PTypesub[j]
        resid[j]                  = y[flag,j].-x[flag,:,j]*bstart[:,j]
        Resid[vec(Choice.==j),j]  = resid[j]
        for c=1:ncl
            residsub[(j-1)*ncl+c] = y[vec((flag) .& (clevel.==c)),j].-x[vec((flag) .& (clevel.==c)),:,j]*bstart[:,j]
            isig[vec((flag) .& (clevel.==c)),j] = 1/sig[(j-1)*ncl+c]*ones(sum(vec((flag) .& (clevel.==c))))
        end
    end

    return bstart,Delta,sig,Resid,resid,residsub,tresid,isig,abilsub
end
