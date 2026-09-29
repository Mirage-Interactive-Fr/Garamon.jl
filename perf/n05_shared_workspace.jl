# K3 research code, separate from N05's measured historical implementation.
isdefined(@__MODULE__,:n05_pfaffian!) || include("n05_pfaffian.jl")

struct N05SharedPlan
    n::Int
    pool_size::Int
    chains::Vector{Vector{Int}}
    pairs::Vector{Tuple{Int,Int}}
    requests::Vector{Vector{NTuple{3,Int}}}
end

function n05_shared_plan(n,pool_size,chains;max_pairs=1<<16,max_bytes=64<<20)
    1<=n<=129 && 1<=pool_size<=512 || throw(ArgumentError("K3 dimension/pool budget"))
    1<=length(chains)<=64 || throw(ArgumentError("K3 chain-count budget"))
    owned=[Int.(collect(chain)) for chain in chains]
    all(chain->iseven(length(chain))&&length(chain)<=64,owned) ||
        throw(ArgumentError("K3 requires even chains of at most 64 vectors"))
    all(chain->all(i->1<=i<=pool_size,chain),owned) || throw(ArgumentError("pool index out of bounds"))
    pairs=Tuple{Int,Int}[];slots=Dict{Tuple{Int,Int},Int}()
    requests=Vector{NTuple{3,Int}}[]
    for chain in owned
        local_requests=NTuple{3,Int}[]
        for j in 2:length(chain),i in 1:j-1
            pair=minmax(chain[i],chain[j]) # Equal indices require a self-contraction.
            slot=get!(slots,pair) do
                length(pairs)<max_pairs || throw(ArgumentError("K3 pair budget"))
                push!(pairs,pair);length(pairs)
            end
            push!(local_requests,(i,j,slot))
        end
        push!(requests,local_requests)
    end
    plan=N05SharedPlan(n,pool_size,owned,pairs,requests)
    Base.summarysize(plan)<=max_bytes || throw(ArgumentError("K3 retained-plan budget"))
    plan
end

mutable struct N05SharedWorkspace{T}
    plan::N05SharedPlan
    transformed::Matrix{T}
    contractions::Vector{T}
    skew::Vector{Matrix{T}}
    output::Vector{T}
    previous_metric::Matrix{T}
    previous_pool::Matrix{T}
    valid::Bool
    policy::Symbol
    contraction_builds::Int
end

function n05_shared_workspace(plan,G,U;policy=:recompute,max_bytes=64<<20)
    policy in (:recompute,:check_inputs) || throw(ArgumentError("unknown K3 refresh policy"))
    size(G)==(plan.n,plan.n) && size(U)==(plan.n,plan.pool_size) || throw(DimensionMismatch("K3 inputs"))
    T=n05_field(promote_type(eltype(G),eltype(U)))
    ws=N05SharedWorkspace(plan,Matrix{T}(undef,size(U)...),Vector{T}(undef,length(plan.pairs)),
        [zeros(T,length(chain),length(chain)) for chain in plan.chains],zeros(T,length(plan.chains)),
        Matrix{T}(undef,size(G)...),Matrix{T}(undef,size(U)...),false,policy,0)
    Base.summarysize(ws)<=max_bytes || throw(ArgumentError("K3 retained-workspace budget"))
    ws
end

# Identical elimination for every K3 candidate, avoiding the temporary pivot
# vector and skew-validation arrays of the first N05 research prototype.
function n05_shared_eliminate!(A::Matrix{T}) where T
    n=size(A,1);value=one(T)
    for k in 1:2:n-1
        p=k+1
        for j in k+2:n
            abs(A[k,j])>abs(A[k,p]) && (p=j)
        end
        iszero(A[k,p]) && return zero(T)
        if p!=k+1
            for j in 1:n;A[k+1,j],A[p,j]=A[p,j],A[k+1,j];end
            for i in 1:n;A[i,k+1],A[i,p]=A[i,p],A[i,k+1];end
            value=-value
        end
        pivot=A[k,k+1];value*=pivot
        for j in k+3:n,i in k+2:j-1
            A[i,j]+=(A[k+1,i]*A[k,j]-A[k,i]*A[k+1,j])/pivot
            A[j,i]=-A[i,j]
        end
    end
    isfinite(value) || throw(OverflowError("K3 nonfinite Pfaffian"))
    value
end

"""Borrowed values. Caller must copy to retain them across the next call."""
function n05_shared_values!(ws::N05SharedWorkspace{T},G,U) where T
    plan=ws.plan
    size(G)==(plan.n,plan.n)&&size(U)==(plan.n,plan.pool_size) || throw(DimensionMismatch("K3 inputs changed shape"))
    n05_field(promote_type(eltype(G),eltype(U)))==T || throw(ArgumentError("K3 coefficient type changed"))
    issymmetric(G)&&all(isfinite,G)&&all(isfinite,U) || throw(ArgumentError("finite symmetric K3 metric and finite pool required"))
    refresh=ws.policy==:recompute || !ws.valid || G!=ws.previous_metric || U!=ws.previous_pool
    if refresh
        mul!(ws.transformed,G,U)
        for (slot,(i,j)) in enumerate(plan.pairs)
            value=zero(T)
            for r in 1:plan.n;value+=U[r,i]*ws.transformed[r,j];end
            isfinite(value) || throw(OverflowError("K3 nonfinite contraction"))
            ws.contractions[slot]=value
        end
        if ws.policy==:check_inputs
            copyto!(ws.previous_metric,G);copyto!(ws.previous_pool,U)
        end
        ws.valid=true;ws.contraction_builds+=1
    end
    for chain in eachindex(plan.chains)
        A=ws.skew[chain];fill!(A,zero(T))
        for (i,j,slot) in plan.requests[chain]
            value=ws.contractions[slot];A[i,j]=value;A[j,i]=-value
        end
        ws.output[chain]=n05_shared_eliminate!(A)
    end
    ws.output
end

n05_shared_owned!(ws,G,U)=copy(n05_shared_values!(ws,G,U))

function n05_shared_oracle(G,U,chains)
    [get(n05_oracle(G,U[:,chain]),big(0),big(0)//1) for chain in chains]
end

function n05_shared_fixture(n,k,count,horizon,family,sharing,changes)
    2<=n<=129 && k in (2,4,8,16) && count in (1,4,16) && horizon in (1,3,32,1024) || error("K3 fixture domain")
    sharing in (:shared,:disjoint) && changes in (:stable,:all,:one) || error("K3 pattern domain")
    pool_size=sharing==:shared ? k : k*count
    chains=sharing==:shared ? [[1+mod(j+2c-3,pool_size) for j in 1:k] for c in 1:count] :
        [collect((c-1)*k+1:c*k) for c in 1:count]
    base=n05_fixture(n,k,1,family);G=base.G
    directions=unique(round.(Int,range(1,n;length=min(n,4))))
    pools=map(1:3) do phase
        U=zeros(Float64,n,pool_size)
        for col in 1:pool_size,(slot,r) in enumerate(directions)
            active_phase=changes==:stable || (changes==:one && col!=1) ? 1 : phase
            U[r,col]=(-2.,-1.,1.,2.)[1+mod(active_phase+slot*col+div(col,2),4)]
        end
        U
    end
    exact=[n05_shared_oracle(G,U,chains) for U in pools]
    state=(;n,k,count,horizon,family,sharing,changes,G,pools,chains,exact,expected=Float64.(hcat(exact...)))
    Base.summarysize(state)<=64<<20 || error("K3 fixture memory budget")
    state
end

const N05_SHARED_ROUTES=(:independent,:workspace_unshared,:shared_allocated,:shared_workspace,:shared_cached)

function n05_shared_episode(state,strategy;trace=false)
    strategy in N05_SHARED_ROUTES || throw(ArgumentError("unknown K3 route"))
    G=state.G;first_pool=first(state.pools)
    plan=strategy in (:independent,:workspace_unshared) ? nothing : n05_shared_plan(state.n,size(first_pool,2),state.chains)
    workspace=strategy in (:shared_workspace,:shared_cached) ?
        n05_shared_workspace(plan,G,first_pool;policy=strategy==:shared_cached ? :check_inputs : :recompute) : nothing
    locals=strategy==:workspace_unshared ? [begin
        pool=first_pool[:,chain];p=n05_shared_plan(state.n,length(chain),[collect(1:length(chain))])
        (;pool,workspace=n05_shared_workspace(p,G,pool))
    end for chain in state.chains] : nothing
    output=Matrix{Float64}(undef,length(state.chains),state.horizon)
    builds=0
    for t in 1:state.horizon
        U=state.pools[1+mod(t-1,3)]
        if strategy==:independent
            for (j,chain) in enumerate(state.chains)
                output[j,t]=n05_shared_eliminate!(n05_contractions(G,U[:,chain]))
                builds+=1
            end
        elseif strategy==:workspace_unshared
            for (j,chain) in enumerate(state.chains)
                local_state=locals[j]
                for (local_col,pool_col) in enumerate(chain),r in 1:state.n
                    local_state.pool[r,local_col]=U[r,pool_col]
                end
                output[j,t]=only(n05_shared_values!(local_state.workspace,G,local_state.pool))
                builds+=1
            end
        else
            current=strategy==:shared_allocated ? n05_shared_workspace(plan,G,U) : workspace
            output[:,t]=n05_shared_values!(current,G,U)
            strategy==:shared_allocated && (builds+=1)
        end
    end
    workspace===nothing || (builds=workspace.contraction_builds)
    trace ? (;output,builds,pair_count=plan===nothing ? sum(length(c)*(length(c)-1)÷2 for c in state.chains) : length(plan.pairs),
        retained_bytes=Base.summarysize((plan,workspace,locals))) : output
end

function n05_shared_qualify(state,output)
    size(output)==(length(state.chains),state.horizon) || return false
    all(isapprox(output[j,t],state.expected[j,1+mod(t-1,3)];atol=1e-8,rtol=1e-10)
        for t in 1:state.horizon for j in eachindex(state.chains))
end
