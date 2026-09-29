using Random

const PRUNING_DIMS = (2,3,4,5,8,12,16,32,64,65,96,128)
const PRUNING_SIGNATURES = (:positive,:indefinite,:degenerate)
const PRUNING_REGIMES = (:contracting,:neutral,:amplifying)
const PRUNING_METHODS = (:exact,:threshold,:roulette,:deferred,:replay)
const PRUNING_MAX_BITS = 16_384
const PRUNING_MAX_PATHS = 10_000_000
const PRUNING_TAU = 1 // 16_384
const PRUNING_PERIOD = 16

function pruning_fixture(n, signature, regime, recurrence, horizon)
    k=min(n,4)
    directions=unique(round.(Int,range(1,n;length=k)))
    diagonal=ones(Int,n)
    signature == :indefinite && (diagonal[directions[2:2:end]].=-1)
    signature == :degenerate && (diagonal[last(directions)]=0)
    masks=[sum((UInt128(1)<<(directions[j]-1) for j in 1:k if !iszero(m & (1<<(j-1))));init=UInt128(0)) for m in 0:(1<<k)-1]
    paths=NTuple{4,Int}[]
    for a in 0:(1<<k)-1,b in 0:(1<<k)-1
        factor=1
        for i in 1:k
            !iszero(a & (1<<(i-1))) || continue
            isodd(count_ones(b & ((1<<(i-1))-1))) && (factor=-factor)
            !iszero(b & (1<<(i-1))) && (factor*=diagonal[directions[i]])
        end
        iszero(factor) || push!(paths,(a+1,b+1,xor(a,b)+1,factor))
    end
    rho=regime==:contracting ? 1//2 : regime==:neutral ? 1//1 : 9//4
    initial=Rational{BigInt}[iszero(m) ? 1//8 : (-1)^count_ones(m)//(big(2)^(m+6)) for m in 0:(1<<k)-1]
    # a_t = 3/4 + e_direction/4, cycling through the active directions.
    # Its coefficient l1 norm is one for all three diagonal signatures.
    affine_paths = if regime==:neutral
        # Multiplication by a non-null unit basis vector is a signed permutation;
        # use identity when that active direction is null.
        [[(a,b,o,f) for (a,b,o,f) in paths
          if a==(diagonal[directions[j]]==0 ? 1 : (1<<(j-1))+1)] for j in 1:k]
    else
        [[(a,b,o,f) for (a,b,o,f) in paths if a==1 || a==(1<<(j-1))+1] for j in 1:k]
    end
    expected_paths=horizon*(recurrence==:affine ? maximum(length,affine_paths) : length(paths)+length(initial))
    expected_paths<=PRUNING_MAX_PATHS || error("reference_path_budget")
    return (;n,signature,regime,recurrence,horizon,k,diagonal,masks,paths,affine_paths,
            rho,initial,tau=PRUNING_TAU,period=PRUNING_PERIOD,expected_paths)
end

# Independent exact oracle: explicit lists of ambient basis directions and
# inversions; no prototype path table and no Garamon multiplication/sign helper.
function pruning_reference_product(x,y,fixture)
    indices=[[i for i in 1:fixture.n if !iszero(mask & (UInt128(1)<<(i-1)))] for mask in fixture.masks]
    lookup=Dict(mask=>i for (i,mask) in enumerate(fixture.masks))
    output=fill(big(0)//big(1),length(x))
    for a in eachindex(x),b in eachindex(y)
        (iszero(x[a]) || iszero(y[b])) && continue
        factor=isodd(sum((i>j for i in indices[a] for j in indices[b]);init=0)) ? -1 : 1
        for i in intersect(indices[a],indices[b]); factor*=fixture.diagonal[i];end
        output[lookup[xor(fixture.masks[a],fixture.masks[b])]]+=factor*x[a]*y[b]
    end
    return output
end

function pruning_oracle(fixture)
    x=copy(fixture.initial)
    history=[copy(x)]
    for step in 1:fixture.horizon
        if fixture.recurrence==:affine
            a=zero(x)
            j=mod1(step,fixture.k)
            if fixture.regime==:neutral
                direction=unique(round.(Int,range(1,fixture.n;length=fixture.k)))[j]
                a[fixture.diagonal[direction]==0 ? 1 : (1<<(j-1))+1]=1
            else
                a[1]=3//4;a[(1<<(j-1))+1]=1//4
            end
            x=fixture.rho.*pruning_reference_product(a,x,fixture)
            x[1]+=1//65_536
        else
            x=fixture.rho.*x.+pruning_reference_product(x,x,fixture)./32
        end
        maximum(v->max(ndigits(numerator(v);base=2),ndigits(denominator(v);base=2)),x)<=PRUNING_MAX_BITS || error("oracle_integer_bit_budget")
        push!(history,copy(x))
    end
    return history
end

mutable struct PruningCounts
    candidates::Int
    kept::Int
    omitted::Int
    correction_paths::Int
    replay_paths::Int
    corrections::Int
end
PruningCounts()=PruningCounts(0,0,0,0,0,0)

@inline function pruning_accumulate!(out,residue,index,value,method,tau,rng,counts,
                                     survival_cutoff,survival_scale)
    counts.candidates+=1
    if method==:roulette
        # Independent Bernoulli(p), with Horvitz-Thompson weighting. Draw one
        # uniform variate per candidate so different p values can use common
        # random numbers without changing the candidate traversal.
        if rand(rng)<survival_cutoff
            out[index]+=survival_scale*value;counts.kept+=1
        else;counts.omitted+=1;end
    elseif method!=:exact && abs(value)<tau
        residue[index]+=value
        counts.omitted+=1
    else
        out[index]+=value;counts.kept+=1
    end
end

function pruning_step!(out,residue,x,fixture,step,method,rng,counts,
                       survival_cutoff,survival_scale)
    T=eltype(x);fill!(out,zero(T));fill!(residue,zero(T))
    rho=T(fixture.rho);tau=T(fixture.tau)
    if fixture.recurrence==:affine
        for (a,b,o,f) in fixture.affine_paths[mod1(step,fixture.k)]
            weight=fixture.regime==:neutral ? one(T) : T(a==1 ? 3//4 : 1//4)
            v=rho*weight*f*x[b]
            pruning_accumulate!(out,residue,o,v,method,tau,rng,counts,
                                survival_cutoff,survival_scale)
        end
        out[1]+=T(1//65_536)
    else
        for i in eachindex(x)
            pruning_accumulate!(out,residue,i,rho*x[i],method,tau,rng,counts,
                                survival_cutoff,survival_scale)
        end
        for (a,b,o,f) in fixture.paths
            pruning_accumulate!(out,residue,o,T(1//32)*f*x[a]*x[b],method,tau,rng,counts,
                                survival_cutoff,survival_scale)
        end
    end
end

function pruning_transport!(dest,error,x,defect,fixture,step,counts)
    T=eltype(x);copyto!(dest,defect)
    if fixture.recurrence==:affine
        for (a,b,o,f) in fixture.affine_paths[mod1(step,fixture.k)]
            weight=fixture.regime==:neutral ? one(T) : T(a==1 ? 3//4 : 1//4)
            dest[o]+=T(fixture.rho)*weight*f*error[b]
            counts.correction_paths+=1
        end
    else
        for i in eachindex(error);dest[i]+=T(fixture.rho)*error[i];end
        for (a,b,o,f) in fixture.paths
            # The quadratic error term is required for exact reconstruction.
            dest[o]+=T(1//32)*f*(x[a]*error[b]+error[a]*x[b]+error[a]*error[b])
            counts.correction_paths+=3
        end
    end
end

function pruning_run(fixture,method,::Type{T}=Float64;seed=1,record=false,
                     clock_parts=false,survival_probability=1//2,
                     measure_buffers=true) where T
    0<survival_probability<=1 || throw(ArgumentError("roulette probability must be in (0,1]"))
    survival_cutoff=Float64(survival_probability)
    survival_scale=inv(T(survival_probability))
    x=T.(fixture.initial);next=zero(x);defect=zero(x);error=zero(x);newerror=zero(x)
    checkpoint=copy(x);replaybuf=zero(x);replaydefect=zero(x)
    rng=Random.Xoshiro(seed);counts=PruningCounts()
    history=record ? Vector{Vector{T}}([copy(x)]) : Vector{Vector{T}}()
    prehistory=Vector{Vector{T}}();maxresidue=0.0;restoration_ns=UInt64(0)
    effective=method==:deferred || method==:replay ? :threshold : method
    checkpoint_step=0
    for step in 1:fixture.horizon
        pruning_step!(next,defect,x,fixture,step,effective,rng,counts,
                      survival_cutoff,survival_scale)
        if method==:deferred
            started=clock_parts ? time_ns() : UInt64(0)
            pruning_transport!(newerror,error,x,defect,fixture,step,counts)
            error,newerror=newerror,error
            maxresidue=max(maxresidue,Float64(sum(abs,error)))
            clock_parts && (restoration_ns+=time_ns()-started)
        end
        x,next=next,x
        record && push!(prehistory,copy(x))
        if method in (:deferred,:replay) && (step%fixture.period==0 || step==fixture.horizon)
            started=clock_parts ? time_ns() : UInt64(0)
            if method==:deferred
                x .+= error;fill!(error,zero(T))
            else
                # checkpoint is the exactly replayed state at checkpoint_step.
                replaycounts=PruningCounts()
                for replaystep in checkpoint_step+1:step
                    pruning_step!(replaybuf,replaydefect,checkpoint,fixture,replaystep,:exact,
                                  rng,replaycounts,survival_cutoff,survival_scale)
                    checkpoint,replaybuf=replaybuf,checkpoint
                end
                counts.replay_paths+=replaycounts.candidates
                copyto!(x,checkpoint);checkpoint_step=step
            end
            counts.corrections+=1
            clock_parts && (restoration_ns+=time_ns()-started)
        end
        all(isfinite,x) || Base.error("nonfinite_trajectory")
        record && push!(history,copy(x))
    end
    # Reflective memory accounting is useful for diagnostics, but can dominate
    # a short recurrence. Timed callers can request it separately from execute.
    buffer_bytes=measure_buffers ?
        Base.summarysize((x,next,defect,error,newerror,checkpoint,replaybuf,replaydefect,rng)) :
        nothing
    return (;value=x,history,prehistory,counts,maxresidue,
            restoration_us=restoration_ns/1000,buffer_bytes)
end

function pruning_errors(run,oracle)
    final=maximum(abs.(BigFloat.(run.value).-BigFloat.(last(oracle))))
    maxerr=final
    for step in eachindex(run.prehistory)
        maxerr=max(maxerr,maximum(abs.(BigFloat.(run.prehistory[step]).-BigFloat.(oracle[step+1]))))
    end
    scale=maximum(abs,BigFloat.(last(oracle)))
    return (;final_abs=Float64(final),max_abs=Float64(maxerr),final_rel=Float64(final/max(scale,big"1e-100")))
end
