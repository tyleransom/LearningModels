function draw_boot_sample(ID,N)
    
    drawn = sample(unique(ID), N; replace=true, ordered=false)
    
    bootidx = zeros(Int64,0)
    
    for i = 1:N
        append!(bootidx,[j[1] for j in findall(ID .== drawn[i])])
    end
    
    return bootidx
end
