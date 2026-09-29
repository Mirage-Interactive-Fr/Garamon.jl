push!(LOAD_PATH,dirname(@__DIR__))
using Garamon, LinearAlgebra, SHA, TOML, Random

const SE_ALGEBRAS=(:euclidean,:one_negative,:balanced_signature,:one_null,:many_null,:scaled_diagonal)
const SE_SUPPORTS=(:local_bivectors,:dispersed_bivectors,:mixed_grades,:random_sparse,:subalgebra,:full_small)

function se_support(family,template)
    dimensions=family==:full_small ? (2,3,4,5) : family==:local_bivectors ? (4,8,12,16) :
        family==:dispersed_bivectors ? (12,32,65,128) : family==:mixed_grades ? (3,5,64,65) :
        family==:random_sparse ? (4,8,64,65) : (3,4,64,128)
    n=dimensions[template]; K=n<=64 ? UInt64 : UInt128
    masks=if family in (:local_bivectors,:dispersed_bivectors)
        active=min(n,template+3)
        directions=family==:local_bivectors ? collect(1:active) : unique(round.(Int,range(1,n;length=active)))
        ntuple(3) do operand
            unique(sort(K[(one(K)<<(directions[i]-1)) | (one(K)<<(directions[1+mod(i+operand-1,active)]-1))
                          for i in 1:active if i!=1+mod(i+operand-1,active)]))
        end
    elseif family==:full_small
        ntuple(_->K.(0:(1<<n)-1),3)
    elseif family==:subalgebra
        active=1+mod(template-1,3)
        ntuple(_->K.(0:(1<<active)-1),3)
    elseif family==:mixed_grades
        ntuple(3) do operand
            bits=K[one(K)<<(i-1) for i in unique((1,min(n,operand+1),n))]
            unique(sort(vcat(K[0],bits,K[foldl(|,bits)])))
        end
    else
        rng=MersenneTwister(20260927158+template)
        active=min(n,template+3); count=min(1<<active,4+template)
        ntuple(_->sort!(K.(randperm(rng,1<<active)[1:count].-1)),3)
    end
    n,masks
end

# Coarse permutation-invariant grouping: collisions merge instances instead of
# leaking near-identical supports into train and test. Unused dimensions vanish.
function se_invariant(supports,n)
    grades=[sort(count_ones.(masks)) for masks in supports]
    incidence=sort([ntuple(op->count(mask->!iszero(mask&(one(mask)<<(i-1))),supports[op]),3)
                    for i in 1:n if any(masks->any(mask->!iszero(mask&(one(mask)<<(i-1))),masks),supports)])
    intersections=[sort([count_ones(a&b) for a in supports[i] for b in supports[j]])
                   for i in 1:3 for j in i:3]
    bytes2hex(sha256(repr((grades,incidence,intersections))))
end

function se_manifest()
    records=NamedTuple[]
    for algebra_family in SE_ALGEBRAS, support_family in SE_SUPPORTS, template in 1:4
        n,supports=se_support(support_family,template)
        push!(records,(;id="$(algebra_family)-$(support_family)-$template",algebra_family,
            support_family,template,n,supports,ancestry="$support_family/$template",
            invariant=se_invariant(supports,n)))
    end
    # Union both ancestry and invariant equivalence, including transitive links.
    parents=collect(eachindex(records))
    root(i)=parents[i]==i ? i : (parents[i]=root(parents[i]))
    for i in eachindex(records),j in 1:i-1
        if records[i].ancestry==records[j].ancestry || records[i].invariant==records[j].invariant
            parents[root(i)]=root(j)
        end
    end
    group=[minimum(findall(j->root(j)==root(i),eachindex(records))) for i in eachindex(records)]
    testgroups=Set(group[i] for i in eachindex(records) if records[i].algebra_family==:many_null && records[i].support_family==:random_sparse)
    valgroups=Set(group[i] for i in eachindex(records) if records[i].algebra_family==:scaled_diagonal && records[i].support_family==:subalgebra)
    setdiff!(valgroups,testgroups)
    [begin
        record=records[i]
        phase=record.algebra_family==:many_null && record.support_family==:random_sparse ? :test :
              record.algebra_family==:scaled_diagonal && record.support_family==:subalgebra && group[i] in valgroups ? :validation :
              !(record.algebra_family in (:many_null,:scaled_diagonal)) &&
              !(record.support_family in (:random_sparse,:subalgebra)) &&
              !(group[i] in union(testgroups,valgroups)) ? :train : :excluded
        (;record...,group="support-$(group[i])",phase)
    end for i in eachindex(records)]
end

function se_diagonal(record)
    n=record.n; d=ones(Float64,n); f=record.algebra_family
    f==:one_negative && (d[end]=-1)
    f==:balanced_signature && (d[2:2:end].=-1)
    f==:one_null && (d[end]=0)
    f==:many_null && (d[2:2:end].=0)
    f==:scaled_diagonal && (d.=[(0.5,1.,2.)[1+mod(i-1,3)] for i in 1:n])
    d
end
se_value(t,operand,slot)=(-2.,-1.,1.,2.)[1+mod(t+operand+slot,4)]

function se_pair_factor(a,b,d)
    inversions=0
    for i in eachindex(d),j in 1:i-1
        !iszero(a&(one(a)<<(i-1))) && !iszero(b&(one(b)<<(j-1))) && (inversions+=1)
    end
    value=isodd(inversions) ? big(-1)//1 : big(1)//1
    for i in eachindex(d)
        !iszero(a&b&(one(a)<<(i-1))) && (value*=Rational{BigInt}(d[i]))
    end
    value
end

function se_oracle(record,t,targets)
    supports=record.supports; d=se_diagonal(record)
    expected=fill(big(0)//1,length(targets)); absolute=big(0)//1; denominator_bound=big(1)
    for (ai,a) in enumerate(supports[1]),(bi,b) in enumerate(supports[2]),(ci,c) in enumerate(supports[3])
        value=Int(se_value(t,1,ai))*Int(se_value(t,2,bi))*Int(se_value(t,3,ci))*
              se_pair_factor(a,b,d)*se_pair_factor(a⊻b,c,d)
        absolute+=abs(value);denominator_bound=lcm(denominator_bound,denominator(value))
        index=findfirst(==(a⊻b⊻c),targets)
        index===nothing || (expected[index]+=value)
    end
    # Dyadic metric, integral inputs: every partial sum stays on the same grid.
    ispow2(denominator_bound) && absolute*denominator_bound<=big(2)^52 ||
        error("oracle exceeds declared exact Float64 accumulation domain")
    values=Float64.(expected)
    Rational{BigInt}.(values)==expected || error("oracle cannot be represented exactly")
    values
end

function se_fixture(record,horizon,outputs)
    ga=algebra(Matrix(Diagonal(se_diagonal(record))))
    requests=outputs==1 ? [Int[]] : [Int[],[1],[record.n],[1,record.n]]
    targets=[foldl((mask,i)->mask|(big(1)<<(i-1)),indices;init=big(0)) for indices in requests]
    operands=[ntuple(3) do op
        multivector(ga,Dict(mask=>se_value(t,op,slot) for (slot,mask) in enumerate(record.supports[op]));storage=:sparse)
    end for t in 1:horizon]
    periodic=[se_oracle(record,t,targets) for t in 1:4]
    expected=hcat([periodic[1+mod(t-1,4)] for t in 1:horizon]...)
    fixture=(;record,horizon,requests,operands,expected)
    Base.summarysize(fixture)<=256<<20 || error("fixture memory budget exceeded")
    fixture
end

function se_episode(fixture,strategy;trace=false)
    a,b,c=first(fixture.operands)
    selector=prepare_triple_selector(a,b,c,fixture.requests;horizon=fixture.horizon,strategy)
    values=Matrix{Float64}(undef,length(fixture.requests),fixture.horizon)
    for (t,operands) in enumerate(fixture.operands)
        values[:,t]=run_selected_triple!(selector,operands...)
    end
    trace ? (;values,selector) : values
end

function se_setup(id,horizon,outputs,strategy,trace_file)
    BLAS.set_num_threads(1)
    record=only(filter(r->r.id==id,se_manifest()))
    fixture=se_fixture(record,horizon,outputs)
    a,b,c=first(fixture.operands)
    feature=@timed triple_strategy_features(a,b,c,fixture.requests;horizon)
    decision=@timed choose_triple_strategy(feature.value;strategy)
    preparation=@timed prepare_triple_selector(a,b,c,fixture.requests;horizon,strategy)
    total=@timed se_episode(fixture,strategy;trace=true)
    total.value.values==fixture.expected || error("selector episode differs from independent oracle")
    if !isfile(trace_file)
        f=feature.value;s=total.value.selector
        details=Dict("id"=>id,"group"=>record.group,"split"=>string(record.phase),
            "horizon"=>horizon,"outputs"=>outputs,"requested_strategy"=>string(strategy),
            "selected_strategy"=>string(s.decision.strategy),"model"=>string(s.decision.model),
            "dimension"=>f.dimension,"mask_type"=>string(f.mask_type),
            "coefficient_type"=>string(f.coefficient_type),"storage"=>collect(string.(f.storage)),
            "metric_class"=>string(f.metric_class),"support_counts"=>collect(f.support_counts),
            "grade_histograms"=>collect(f.grade_histograms),"active_directions"=>f.active_directions,
            "request_grades"=>f.request_grades,"measured_cache_state"=>"warm_code_new_structure",
            "diagnostic_state"=>"first_episode_after_feature_decision_and_preparation_probes",
            "features_ms"=>1000feature.time,"features_compile_ms"=>1000feature.compile_time,
            "decision_ms"=>1000decision.time,"decision_compile_ms"=>1000decision.compile_time,
            "preparation_inclusive_ms"=>1000preparation.time,
            "preparation_compile_ms"=>1000preparation.compile_time,
            "preparation_bytes"=>preparation.bytes,
            "diagnostic_episode_ms"=>1000total.time,"diagnostic_episode_compile_ms"=>1000total.compile_time,
            "diagnostic_episode_bytes"=>total.bytes,"artifact_bytes"=>s.artifact_bytes,
            "second_plan_rebuilds"=>s.second_plan_rebuilds,"exact"=>true,
            "candidates"=>Dict(string(name)=>Dict("admitted"=>v.admitted,"reason"=>string(v.reason),
                "cost_proxy"=>string(v.cost_proxy),"prep_work_bound"=>string(v.prep_work),
                "per_call_work_bound"=>string(v.per_call_work),"estimated_bytes"=>string(v.estimated_bytes))
                for (name,v) in decision.value.candidates))
        open(io->TOML.print(io,details),trace_file,"w")
    end
    (;fixture,strategy)
end
