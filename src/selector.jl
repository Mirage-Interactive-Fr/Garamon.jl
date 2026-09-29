"""Experimental exact-selector limits; work counts are conservative pair/probe bounds."""
struct TripleSelectorBudget
    max_paths::Int
    max_work::Int
    max_bytes::Int
    max_support::Int
    function TripleSelectorBudget(; max_paths::Integer=1<<16,
            max_work::Integer=1<<20, max_bytes::Integer=64<<20,
            max_support::Integer=1<<12)
        all(x->0<x<=typemax(Int),(max_paths,max_work,max_bytes,max_support)) ||
            throw(ArgumentError("selector budgets must be positive machine integers"))
        new(Int(max_paths),Int(max_work),Int(max_bytes),Int(max_support))
    end
end

struct TripleSelectionRefusal <: Exception
    reason::Symbol
end
Base.showerror(io::IO,error::TripleSelectionRefusal) =
    print(io,"exact triple selector refused: ",error.reason)

const _TRIPLE_CANDIDATES=(:full,:recursive,:join3,:prepared,:workspace)

# With n <= 12, horizon <= 4096, at most 4096 requested masks and a bounded
# algebra/type footprint, every count, byte bound and analytic cost below fits
# in a signed 64-bit Int. Larger cases retain the arbitrary-precision path.
function _selector_machine_arithmetic(n,horizon,request_count,algebra_bytes,::Type{S}) where S
    Sys.WORD_SIZE==64 && n<=12 && horizon<=4096 && request_count<=4096 &&
        algebra_bytes<=1<<30 && sizeof(S)<=1<<16
end

function _triple_features_arithmetic(ga,operands,supports,counts,histograms,
        targets,active,n,::Type{K},::Type{S},diagonal,metric_class,
        horizon,algebra_bytes,::Type{W}) where {K,S,W<:Integer}
    na,nb,nc=W.(counts)
    capacity=W(2)^n
    ab_bound=min(capacity,na*nb*(diagonal ? one(W) : capacity))
    pair1=na*nb; pair2=ab_bound*nc
    small=sort(collect((na,nb,nc)))
    join_work=W(length(targets))*small[1]*small[2]
    recursive_work=pair1+W(length(targets))*min(ab_bound,nc)*(min(na,nb)+1)
    (; dimension=n,mask_type=K,coefficient_type=S,
       storage=map(x->x isa DenseMultiVector ? :dense : :sparse,operands),
       metric_class,diagonal,supports,support_counts=counts,grade_histograms=histograms,
       active_directions=_blade_grade(active),request_masks=targets,
       request_grades=_blade_grade.(targets),request_count=length(targets),
       horizon=Int(horizon),cache_state=:absent,pair1,pair2,ab_bound,join_work,recursive_work,
       algebra_bytes)
end

"""
    triple_strategy_features(a,b,c,outputs; horizon=1)

Extract observable structural features without computing a product or plan.
The experimental domain is finite, fixed-size numeric coefficients. Requested
coefficients must be distinct. Feature extraction is deliberately explicit so
its cost can be included in a selection experiment.
"""
function triple_strategy_features(a::AbstractMultiVector,b::AbstractMultiVector,
                                  c::AbstractMultiVector,outputs;horizon::Integer=1)
    1<=horizon<=typemax(Int) || throw(ArgumentError("horizon must be a positive machine integer"))
    ga=_same_algebra(a,b); _same_algebra(a,c)
    dimension(ga)<=128 || throw(TripleSelectionRefusal(:dimension_domain))
    S=promote_type(eltype(metric(ga)),eltype(a),eltype(b),eltype(c))
    isbitstype(S) && S<:Real || throw(TripleSelectionRefusal(:coefficient_type))
    all(isfinite,metric(ga)) || throw(TripleSelectionRefusal(:nonfinite_metric))
    K=_masktype(ga)
    supports=map((a,b,c)) do value
        masks=K[]
        for (mask,coefficient) in _terms(value)
            _mask_in_bounds(ga,mask) || throw(TripleSelectionRefusal(:invalid_input_mask))
            isfinite(coefficient) || throw(TripleSelectionRefusal(:nonfinite_coefficient))
            push!(masks,mask)
        end
        sort!(masks)
    end
    targets=K[_mask(ga,indices) for indices in outputs]
    !isempty(targets) && length(unique(targets))==length(targets) ||
        throw(ArgumentError("request a nonempty list of distinct coefficients"))
    n=dimension(ga); counts=map(length,supports)
    histograms=map(supports) do masks
        histogram=zeros(Int,n+1)
        for mask in masks; histogram[1+_blade_grade(mask)]+=1; end
        histogram
    end
    active=zero(K)
    for masks in supports, mask in masks; active|=mask; end
    diagonal=isdiag(metric(ga))
    diagonal_values=diag(metric(ga))
    metric_class=!diagonal ? :nonorthogonal : any(iszero,diagonal_values) ? :degenerate :
        any(<(0),diagonal_values) ? :indefinite :
        all(isone,diagonal_values) ? :euclidean : :scaled_diagonal
    algebra_bytes=Base.summarysize(ga)
    operands=(a,b,c)
    if _selector_machine_arithmetic(n,horizon,length(targets),algebra_bytes,S)
        _triple_features_arithmetic(ga,operands,supports,counts,histograms,targets,
            active,n,K,S,diagonal,metric_class,horizon,algebra_bytes,Int)
    else
        _triple_features_arithmetic(ga,operands,supports,counts,histograms,targets,
            active,n,K,S,diagonal,metric_class,horizon,algebra_bytes,BigInt)
    end
end

"""
    triple_strategy_admission(features,budget=TripleSelectorBudget())

Deterministic admission plus an UNFITTED analytic cost proxy (not nanoseconds).
No measured candidate time or oracle result is consulted. The memory estimate
is conservative admission for retained artifacts, not a process/RSS quota.
"""
function _triple_strategy_admission(f,budget::TripleSelectorBudget,::Type{W}) where {W<:Integer}
    h=W(f.horizon); full=f.pair1+f.pair2
    width=max(1,cld(f.dimension,64))
    capacity=W(2)^f.dimension
    outputs_bound=min(capacity,f.ab_bound*W(last(f.support_counts)))
    slots=sum(W.(f.support_counts))+f.ab_bound+outputs_bound+W(f.request_count)
    plan_bytes=W(2)*W(f.algebra_bytes)+W(256)*slots+W(64)*full+W(4096)
    workspace_bytes=plan_bytes+W(256)*slots+W(sizeof(f.coefficient_type))*slots
    # Explicitly bounded fallback for tiny nonorthogonal algebras only. In 2D
    # a blade pair has at most four exterior coefficients; 16 is a deliberately
    # loose bound on the elementary Clifford-word expansion of that pair.
    nonorthogonal_work=W(16)*(f.pair1+capacity*W(last(f.support_counts)))
    Candidate=NamedTuple{(:admitted,:reason,:prep_work,:per_call_work,:estimated_bytes,:cost_proxy),
        Tuple{Bool,Symbol,W,W,W,W}}
    result=Dict{Symbol,Candidate}()
    for strategy in _TRIPLE_CANDIDATES
        prep=strategy==:prepared ? full : strategy==:workspace ? W(2)*f.pair1+f.pair2 : zero(W)
        per=strategy==:join3 ? f.join_work : strategy==:recursive ? f.recursive_work :
            strategy==:workspace ? f.pair1+W(2)*f.pair2 : full
        strategy==:full && !f.diagonal && (per=nonorthogonal_work)
        bytes=strategy==:workspace ? workspace_bytes : strategy==:prepared ? plan_bytes :
            W(f.algebra_bytes)+W(256)*slots+W(4096)
        reason=if !f.diagonal && !(strategy==:full && f.dimension<=2)
            :metric_domain
        elseif strategy in (:prepared,:workspace) && max(f.pair1,f.pair2)>budget.max_paths
            :path_budget
        elseif strategy==:recursive && maximum((f.support_counts...,f.ab_bound))>budget.max_support
            :support_budget
        elseif prep+h*per>budget.max_work
            :work_budget
        elseif bytes>budget.max_bytes
            :memory_budget
        else
            :admitted
        end
        # Fixed operation-count weights, version analytic_v1_unfitted. These
        # weights are hypotheses to measure, not coefficients learned from CSVs.
        cost=strategy==:full ? h*W(6)*W(width)*per :
             strategy==:recursive ? h*W(8)*W(width)*per :
             strategy==:join3 ? h*W(4)*W(width)*per :
             strategy==:prepared ? W(8)*W(width)*prep+h*(W(2)*full+W(2)*outputs_bound) :
             W(8)*W(width)*prep+h*(full+sum(W.(f.support_counts))+outputs_bound+
                 W(f.request_count)+f.pair2)
        result[strategy]=(;admitted=reason==:admitted,reason,prep_work=prep,
                            per_call_work=per,estimated_bytes=bytes,cost_proxy=cost)
    end
    result
end

function triple_strategy_admission(f,budget::TripleSelectorBudget=TripleSelectorBudget())
    _triple_strategy_admission(f,budget,typeof(f.pair1))
end

"""Choose only among admitted candidates. Explicit strategy forcing is for paired baselines."""
function choose_triple_strategy(features,budget::TripleSelectorBudget=TripleSelectorBudget();
                                strategy::Symbol=:auto)
    strategy in (:auto,_TRIPLE_CANDIDATES...) || throw(ArgumentError("unknown triple strategy"))
    candidates=triple_strategy_admission(features,budget)
    admitted=filter(s->candidates[s].admitted,_TRIPLE_CANDIDATES)
    isempty(admitted) && throw(TripleSelectionRefusal(:no_admissible_candidate))
    selected=strategy==:auto ? admitted[argmin([candidates[s].cost_proxy for s in admitted])] : strategy
    candidates[selected].admitted || throw(TripleSelectionRefusal(candidates[selected].reason))
    (;strategy=selected,candidates,model=:analytic_v1_unfitted)
end

"""Mutable episode state; each task needs its own selector and workspace."""
mutable struct ExactTripleSelector
    features::NamedTuple
    decision::NamedTuple
    budget::TripleSelectorBudget
    requested::Vector{Vector{Int}}
    metric_snapshot::AbstractMatrix
    basis_snapshot::Vector{String}
    algebra_kind::Symbol
    artifact::Any
    remaining_calls::Int
    remaining_work::BigInt
    second_plan_rebuilds::Int
    artifact_bytes::Int
end

function _selector_artifact(strategy,a,b,c,budget)
    strategy in (:full,:recursive,:join3) && return nothing
    first=prepare_product(a,b;max_paths=budget.max_paths)
    if strategy==:prepared
        T=eltype(first.diagonal); K=eltype(first.output_masks)
        structural=multivector(a.algebra,Dict{K,T}(mask=>one(T) for mask in first.output_masks);storage=:sparse)
        second=prepare_product(structural,c;max_paths=budget.max_paths)
        return (;first,second)
    end
    first_workspace=ProductWorkspace(first,a,b;max_bytes=budget.max_bytes)
    intermediate=run_product!(first_workspace,a,b)
    second=prepare_product(intermediate,c;max_paths=budget.max_paths)
    second_workspace=ProductWorkspace(second,intermediate,c;max_bytes=budget.max_bytes)
    (;first=first_workspace,second=second_workspace)
end

"""
    prepare_triple_selector(a,b,c,outputs; horizon=1,budget=TripleSelectorBudget(),strategy=:auto)

Prepare an experimental exact episode. All preparation and feature extraction
belong to this call. Input supports must remain fixed during the episode;
changing them requires a new selector. Intermediate cancellation is supported:
prepared products admit structural subsets, and workspaces rebuild the second
plan when its numeric support changes. No approximate fallback is used.
"""
function prepare_triple_selector(a::AbstractMultiVector,b::AbstractMultiVector,
                                 c::AbstractMultiVector,outputs;horizon::Integer=1,
                                 budget::TripleSelectorBudget=TripleSelectorBudget(),strategy::Symbol=:auto)
    requested=[collect(Int,indices) for indices in outputs]
    features=triple_strategy_features(a,b,c,requested;horizon)
    decision=choose_triple_strategy(features,budget;strategy)
    artifact=_selector_artifact(decision.strategy,a,b,c,budget)
    bytes=Base.summarysize(artifact)
    bytes<=budget.max_bytes || throw(TripleSelectionRefusal(:actual_artifact_memory))
    ga=a.algebra
    selector=ExactTripleSelector(features,decision,budget,requested,copy(metric(ga)),copy(basis(ga)),
        kind(ga),artifact,Int(horizon),big(budget.max_work)-decision.candidates[decision.strategy].prep_work,0,bytes)
    Base.summarysize(selector)<=budget.max_bytes || throw(TripleSelectionRefusal(:actual_retained_memory))
    selector
end

function _validate_selector(selector,a,b,c)
    selector.remaining_calls>0 || throw(TripleSelectionRefusal(:episode_exhausted))
    ga=_same_algebra(a,b); _same_algebra(a,c)
    metric(ga)==selector.metric_snapshot && basis(ga)==selector.basis_snapshot &&
        kind(ga)==selector.algebra_kind || throw(TripleSelectionRefusal(:algebra_changed))
    promote_type(eltype(metric(ga)),eltype(a),eltype(b),eltype(c))==selector.features.coefficient_type ||
        throw(TripleSelectionRefusal(:coefficient_type_changed))
    for (value,masks) in zip((a,b,c),selector.features.supports)
        count=0
        for (mask,coefficient) in _terms(value)
            isfinite(coefficient) || throw(TripleSelectionRefusal(:nonfinite_coefficient))
            index=searchsortedfirst(masks,mask)
            index<=length(masks) && masks[index]==mask || throw(TripleSelectionRefusal(:input_support_changed))
            count+=1
        end
        count==length(masks) || throw(TripleSelectionRefusal(:input_support_changed))
    end
    nothing
end

"""
    run_selected_triple!(selector,a,b,c)

Return a fresh owned coefficient vector in the requested order. The selector
is mutated only to account for its episode and maintain its private workspace.
The result never aliases a workspace; previous results survive later calls.
"""
function run_selected_triple!(selector::ExactTripleSelector,a::AbstractMultiVector,
                               b::AbstractMultiVector,c::AbstractMultiVector)
    _validate_selector(selector,a,b,c)
    strategy=selector.decision.strategy
    charge=selector.decision.candidates[strategy].per_call_work
    charge<=selector.remaining_work || throw(TripleSelectionRefusal(:work_budget))
    result=if strategy==:full
        value=(a*b)*c
        [coefficient_mask(value,mask) for mask in selector.features.request_masks]
    elseif strategy==:recursive
        plan=prepare_expression(@ga (a*b)*c)
        evaluate(plan;outputs=selector.requested,strategy=:recursive,
            max_support=selector.budget.max_support,max_pairs=selector.budget.max_work)
    elseif strategy==:join3
        [triple_product_coefficient(a,b,c,indices;max_pairs=selector.budget.max_work)
         for indices in selector.requested]
    elseif strategy==:prepared
        intermediate=run_product(selector.artifact.first,a,b)
        value=run_product(selector.artifact.second,intermediate,c;allow_subsets=true)
        [coefficient_mask(value,mask) for mask in selector.features.request_masks]
    else
        artifact=selector.artifact
        intermediate=run_product!(artifact.first,a,b)
        if !_matches_plan_support(intermediate,artifact.second.plan.left_support)
            plan=prepare_product(intermediate,c;max_paths=selector.budget.max_paths)
            second=ProductWorkspace(plan,intermediate,c;max_bytes=selector.budget.max_bytes)
            replacement=(;first=artifact.first,second)
            bytes=Base.summarysize(replacement)
            bytes<=selector.budget.max_bytes || throw(TripleSelectionRefusal(:actual_artifact_memory))
            selector.artifact=replacement
            if Base.summarysize(selector)>selector.budget.max_bytes
                selector.artifact=artifact
                throw(TripleSelectionRefusal(:actual_retained_memory))
            end
            selector.artifact_bytes=bytes
            selector.second_plan_rebuilds+=1
        end
        value=run_product!(selector.artifact.second,intermediate,c)
        [coefficient_mask(value,mask) for mask in selector.features.request_masks]
    end
    selector.remaining_work-=charge
    selector.remaining_calls-=1
    convert(Vector{selector.features.coefficient_type},result)
end
