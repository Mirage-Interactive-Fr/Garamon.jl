include("binary_rank_cases.jl")
include("binary_rank_compact.jl")
using .BinaryRankCompactPrototype

const COMPACT_SCREEN_DIMS=(2,4,6,8,12,16,24,32,48,64,65,96,128)
const COMPACT_METHODS=(:rank_full,:rank_compact,:rank_workspace)
const COMPACT_DECISION_METHODS=(:prepared,COMPACT_METHODS...)
const COMPACT_FULL_QUERIES=((:all,0),(:grade,1),(:coefficient,0))

function compact_fixture(n,family,signature)
    fixture=rank_fixture(n,family,signature)
    # The degenerate direction is active even for low-grade supports at large
    # n; tests therefore cannot accidentally leave every null factor dormant.
    if signature==:degenerate
        fixture.diagonal[1]=0
        first(fixture.inputs)[1].algebra.metric.diag[1]=0.
    end
    fixture
end

function compact_artifact(fixture,method,query)
    a,b=first(fixture.inputs)
    method==:prepared && return prepare_product(a,b)
    method==:rank_full && return prepare_rank(a,b;query)
    plan=prepare_compact_rank(a,b;query)
    method==:rank_compact && return plan
    method==:rank_workspace && return CompactRankWorkspace(plan,Float64)
    error("unknown method")
end

function compact_execute(fixture,method,artifact,query,horizon)
    outputs=Vector{typeof(first(fixture.inputs)[1])}(undef,horizon)
    checksum=0.0
    for i in 1:horizon
        a,b=fixture.inputs[mod1(i,4)]
        out=method==:prepared ? select_query(run_product(artifact,a,b),query) :
            method==:rank_full ? rank_product(artifact,a,b) :
            method==:rank_compact ? compact_rank_product(artifact,a,b) :
            compact_rank_product!(artifact,a,b)
        outputs[i]=out;checksum+=sum(values(out.values);init=0.)
    end
    (;outputs,checksum)
end

function compact_episode(fixture,method,query,horizon)
    artifact=compact_artifact(fixture,method,query)
    compact_execute(fixture,method,artifact,query,horizon)
end

function compact_correct(fixture,method,artifact,query)
    result=compact_execute(fixture,method,artifact,query,4)
    rank_owned_correct(fixture,result,query,4)
end

compact_plan(artifact::CompactRankWorkspace)=artifact.plan
compact_plan(artifact)=artifact

function compact_metadata(plan::ProductPlan)
    masks=vcat(plan.left_masks,plan.right_masks)
    rank=length(binary_basis(masks,dimension(plan.algebra)))
    (;rank,pairs=length(plan.left_masks)*length(plan.right_masks),slots=length(plan.output_masks),span_slots=big(1)<<rank,
      plan_bytes=Base.summarysize(plan),artifact_bytes=Base.summarysize(plan),numerical_buffer_bytes=0)
end

function compact_metadata(artifact)
    plan=compact_plan(artifact)
    pairs=length(plan.left_masks)*length(plan.right_masks)
    slots=length(plan.phi)
    (;rank=length(plan.basis),pairs,slots,span_slots=big(1)<<length(plan.basis),
      plan_bytes=Base.summarysize(plan),artifact_bytes=Base.summarysize(artifact),
      numerical_buffer_bytes=artifact isa CompactRankWorkspace ?
          sizeof(artifact.left_values)+sizeof(artifact.right_values)+sizeof(artifact.output_values) : 0)
end

# Versioned, stable identifiers for the full decision grid. A dimension is
# exactly one shard: 3 families × 3 metrics × 3 horizons × 3 queries × 4 routes.
const COMPACT_SHARD_SIZE=324
compact_shard_id(n)="n"*lpad(string(n),3,'0')
function compact_case_id(c)
    "K1v2-$(compact_shard_id(c.n))-$(c.family)-$(c.signature)-$(c.query[1])$(c.query[2])-h$(c.horizon)-$(c.method)"
end
compact_execution_id(c)=compact_case_id(c)*"-pass$(c.passage)"
function compact_global_index(c)
    fi=findfirst(==(c.family),RANK_FAMILIES)-1
    si=findfirst(==(c.signature),RANK_SIGNATURES)-1
    hi=findfirst(==(c.horizon),RANK_HORIZONS)-1
    qi=findfirst(==(c.query),COMPACT_FULL_QUERIES)-1
    mi=findfirst(==(c.method),COMPACT_DECISION_METHODS)-1
    (((((c.n-2)*3+fi)*3+si)*3+hi)*3+qi)*4+mi+1
end
