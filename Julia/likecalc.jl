function likecalc(Choice,X,Z,y,x,cl,tvec,α_est,β_est,Δ_est,σ_est,N::Int64,T::Int64,S::Int64,J::Int64,ncl::Int64)
#@views @inline function likecalc(Choice,X,Z,y,x,cl,α_est,β_est,Δ_est,σ_est,N::Int64,T::Int64,S::Int64,J::Int64,ncl::Int64)
    #--------------------------------------------------------------------------
    # Likelihood contribution of choice model
    #--------------------------------------------------------------------------
    P = pclogit(α_est,Choice,X,Z,J+1)

    Choicewtemp = reshape(Choice,N,T,S)
    @assert isequal(Choicewtemp[:,:,1],Choicewtemp[:,:,end])
    Pw = zeros(N,T,S,J+1)
    Choicew = zeros(N,T,S,J+1)
    for j=1:J+1
        Pw[:,:,:,j] = reshape(P[:,j],N,T,S,1)
        Choicew[:,:,:,j] = (Choicewtemp.==j)
    end

    choice_like = ones(N,S)
    for s=1:S
        choice_like[:,s] = dropdims(dropdims(prod(prod(Pw[:,:,s,:].^Choicew[:,:,s,:];dims=3);dims=2);dims=3);dims=2)
    end

    #--------------------------------------------------------------------------
    # Likelihood contribution of learning model
    #--------------------------------------------------------------------------
    function create_mu(Choice,y,x,β_est,N,T,S,J,cl,ncl)
        # this function needs to create yhat for each of the grades from various fields
        #predy = zeros(N*T*S,J)
        tvectest = zeros(N*T*S,J*ncl)
        Cnt     = zeros(N*T*S,J*ncl)
        flag   = zeros(N*T*S,1)
        predyjc = zeros(N*T*S,J*ncl)
        yjc     = zeros(N*T*S,J*ncl)
        for j=1:J
            for cc=1:ncl
                flag                       = vec((Choice.==j) .& (cl.==cc))
                predyjc[flag,(j-1)*ncl+cc] = x[flag,:,j]*β_est[:,j]
                yjc[flag,(j-1)*ncl+cc]     = y[flag,j]
                Cnt[:,(j-1)*ncl+cc]        = ((j-1)*ncl+cc).*flag
                tvectest[:,(j-1)*ncl+cc]  .= ((j-1)*ncl+cc)
            end
        end
        tvectest[:,1] = tvec
        testy = permutedims(reshape(tvectest,T,N,S,J*ncl),[2 1 3 4])
        testy = reshape(permutedims(testy,[1 2 4 3]),N,T*J*ncl,S)

        # nt is simply the total number of periods that an individual sees a learning outcome
        Cs = permutedims(reshape(Choice,T,N,S),[2 1 3])
        nt = sum((Cs[:,:,1].>0) .& (Cs[:,:,1].<=J); dims=2)

        # Cnt is a dummy identifying the sector-heteroskedastic pair of the individual
        # need to get Cnt from a N*T*S x J*ncl to a N x T*J*ncl x 1 [dropping 2nd and higher dimensions of S since this is data]
        Cnt = permutedims(reshape(Cnt,T,N,S,J*ncl),[2 1 3 4])
        Cnt = Cnt[:,:,1,:]
        Cnt = reshape(permutedims(Cnt,[1 3 2]),N,T*J*ncl)

        # now do the same reshaping with yjc to generate ξ
        yjc = permutedims(reshape(yjc,T,N,S,J*ncl),[2 1 3 4])
        yjc = yjc[:,:,1,:]
        ξ   = reshape(permutedims(yjc,[1 3 2]),N,T*J*ncl) # needs to be N x (T*J*ncl)
        
        # Cntag is the same as Cnt, but it's the sector instead of the sector-heteroskedastic pair
        Cntag = zeros(Int64,size(Cnt))
        for j=1:J
            Cntag[vec((Cnt.>=(j-1)*ncl+1) .& (Cnt.<=j*ncl))] .= j
        end
        
        # now do the same reshaping with predyjc to generate μ, except here we have an S dimension since μ is model-based rather than data
        predyjc = permutedims(reshape(predyjc,T,N,S,J*ncl),[2 1 3 4])
        μ       = reshape(permutedims(predyjc,[1 2 4 3]),N,T*J*ncl,S)
        
        return Cnt,Cntag,nt,μ,ξ,testy
    end

    # update object that encompasses learning parameters and learning data
    Cnt,Cntag,nt,μ,ξ,testy = create_mu(Choice,y,x,β_est,N,T,S,J,cl,ncl)

    learn_like = zeros(N,S)
    for i=1:N
        if nt[i]>0
            nnt = sum(Cnt[i,:].>0)
            Δi = zeros(nnt,nnt)
            jp=1
            for j=1:(J*ncl*T)
                kp=1
                for k=1:j
                    if Cnt[i,j]>0 && Cnt[i,k]>0
                        if jp!=kp
                            Δi[jp,kp] = Δ_est[Cntag[i,j],Cntag[i,k]]
                        else
                            Δi[jp,kp] = Δ_est[Cntag[i,j],Cntag[i,k]] 
                            for jj=1:J
                                for cc=1:ncl
                                    if Cnt[i,j]==((jj-1)*ncl+cc)
                                        Δi[jp,kp] += σ_est[(jj-1)*ncl+cc]
                                    end
                                end
                            end
                        end
                        kp+=1
                    end
                end
                if Cnt[i,j]>0
                    jp+=1
                end
            end
            Δi = Δi .+ Δi' .- Diagonal(diag(Δi))
            @assert Δi ≈ Δi' "Covariance matrix is not symmetric to 1e-3 tolerance"
            #if all(eig(squeeze(Δi))>0)!=1
            #    println(i)
            #    println(Cnt[i,:])
            #    println(μ[i,:,:])
            #    println(ξ[i,:,:])
            #    println(Δi)
            #    println("Covariance matrix is not positive definite in likecalc")
            #    error("Covariance matrix is not positive definite")
            #end
            for s=1:S
                learn_like[i,s] = pdf(MvNormal(μ[i,ξ[i,:].!=0,s],Δi),ξ[i,ξ[i,:].!=0])
            end
        # Final case: Those who don't contribute to the likelihood
        else
            for s=1:S
                learn_like[i,s] = 1
            end
        end
    end

    #--------------------------------------------------------------------------
    # Overall likelihood (equals product of two)
    #--------------------------------------------------------------------------
    full_like = choice_like.*learn_like
    return full_like,testy
end
