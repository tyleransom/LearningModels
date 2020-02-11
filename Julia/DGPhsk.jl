function DGPhsk(N=10_000,T=10,J=7)
    NT=N*T
    ncl= 4 # number of course levels
    # S=2
    X=[ones(N,1) randn(N,1) randn(N,1).>0 randn(N,1).>.1] # all time-invariant regressors right now; last one is type
    # PType=[X(:,end)==0;X(:,end)==1]
    # prior_ans = [mean(X(:,end)==0) mean(X(:,end)==1)]

    A  = zeros(N,J)
    corrseq = round.(9*rand(J-1,1).+1;digits=0)./10
    idioseq = round.(9*rand(J-1,1).+1;digits=0)./10
    A[:,1] = randn(N,1)
    for j=2:J
        A[:,j] = corrseq[j-1]*A[:,j-1] + idioseq[j-1]*randn(N,1)
    end

    Delta_ans = cov(A)
    DeltaCorr_ans = cor(A)

    sig_ans = round.(10*rand(J*ncl,1).+1;digits=0)./10 

    b = zeros(size(X,2),J+1)
    for j=1:J
        b[:,j] = round.(10*rand(size(X,2),1);digits=0)./10 
    end

    U = zeros(N,J)
    for j=1:J
        U[:,j] = X*b[:,j]
    end

    b_ans = b[:,1:J][:]

    dem = 1 .+ sum(exp.(U);dims=2)
    p = zeros(N,J)
    for j=1:J
        p[:,j] = exp.(U[:,j])./dem
    end

    draw=rand(N,T)

    # generate choice variable (NxT)
    # C=j if choice=j, C=(J+1) if choice=home
    C = (J+1)*ones(N,T)
    for i=1:N
        for t=1:T
            for j=J:-1:2
                if ((draw[i,t]<sum(p[i,1:j])) && (draw[i,t]>sum(p[i,1:j-1])))
                    C[i,t] = j
                end
            end
            if draw[i,t]<p[i,1]
                C[i,t] = 1
            end
        end
    end
    Clp=reshape(C',N*T,1)
    #Cl=kron(ones(S,1),Clp)

    clevel = rand(1:ncl,size(Clp,1),size(Clp,2))

    ID=collect(1:N)
    IDlp=kron(ID,ones(T,1))
    #IDl=kron(ones(S,1),IDlp)

    yrc   = cumsum(C.<=J;dims=2).*(C.<=J)
    #yrcl  = [reshape(yrc',N*T,1);reshape(yrc',N*T,1)]
    yrclp = reshape(yrc',N*T,1)

    # reshape from wide to long
    x = zeros(NT,size(X,2),J)
    for j=1:J
        x[:,:,j] = kron(X,ones(T,1))
    end

    # generate outcomes; each heteroskedastic variance is formally a different outcome 
    y = zeros(NT,J)
    for j=1:J
        for c=1:ncl
            y[vec(clevel.==c),j] = x[vec(clevel.==c),:,j]*b[:,j] .+ kron(A[:,j],ones(T,1))[vec(clevel.==c)] .+ sqrt(sig_ans[(j-1)*ncl+c])*randn(sum(clevel.==c),1)
        end
    end

    # stack the data
    # wageg     = kron(ones(S,1),wageg    )       
    # wagen     = kron(ones(S,1),wagen    )
    # grade2    = kron(ones(S,1),grade2   )
    # grade4s   = kron(ones(S,1),grade4s  )
    # grade4h   = kron(ones(S,1),grade4h  )

    # xg  = [xg(:,1:end-1)  zeros(N*T,1);xg(:,1:end-1)  ones(N*T,1)]
    # xn  = [xn(:,1:end-1)  zeros(N*T,1);xn(:,1:end-1)  ones(N*T,1)]
    # x4s = [x4s(:,1:end-1) zeros(N*T,1);x4s(:,1:end-1) ones(N*T,1)]
    # x4h = [x4h(:,1:end-1) zeros(N*T,1);x4h(:,1:end-1) ones(N*T,1)]
    # x2  = [x2(:,1:end-1)  zeros(N*T,1);x2(:,1:end-1)  ones(N*T,1)]

    # Xs=[kron(X(:,1:end-1),ones(T,1))  zeros(N*T,1);kron(X(:,1:end-1),ones(T,1))  ones(N*T,1)]

    # Create data consistent with the actual data 
    # xg    = xg (Cl==1,:)
    # xn    = xn (Cl==2,:)
    # x4s   = x4s(Cl==3,:)
    # x4h   = x4h(Cl==4,:)
    # x2    = x2 (Cl==5,:)

    for i=1:NT
        for j=1:J
            if Clp[i]!=j
                y[i,j] = 999
            end
        end
    end

    time = kron(ones(N,1),collect(1:T))
    
    S = 1;

    return IDlp,time,clevel,ncl,N,T,S,y,x,Clp,A,b,sig_ans,Delta_ans,DeltaCorr_ans
end
