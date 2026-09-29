# I15.9: benchmark-only code. Each task/process owns its mutable workspace.
using Garamon, Serialization, LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

const PARALLEL_LIMITS = (max_paths=4096, max_batch_paths=4_194_304,
    max_output_bytes=64<<20, max_state_bytes=64<<20, max_rss_bytes=2<<30,
    max_pool_rss_bytes=8<<30, seconds_per_configuration=600)

function parallel_masks(n, family)
    active = family == :sparse8 ? min(3, n) : min(6, n)
    directions = unique(round.(Int, range(1, n; length=active)))
    return [sum((UInt128(1) << (directions[i]-1) for i in 1:active
                 if !iszero(bits & (1 << (i-1)))); init=UInt128(0))
            for bits in 0:(1<<active)-1]
end

# Exact, independent blade multiplication: count explicit basis-index inversions.
function parallel_reference(left, right)
    indices(mask) = [i for i in 1:128 if !iszero(mask & (UInt128(1) << (i-1)))]
    result = Dict{UInt128,Int64}()
    for (a,x) in left, (b,y) in right
        inversions = sum((i > j for i in indices(a) for j in indices(b)); init=0)
        mask = xor(a,b)
        result[mask] = get(result,mask,0) + (isodd(inversions) ? -x*y : x*y)
    end
    return result
end

function parallel_fixture(n, family)
    family in (:sparse8, :subalgebra64) || error("unknown family")
    masks = parallel_masks(n,family)
    references = ntuple(4) do variant
        ntuple(2) do operand
            Dict(mask => Int64(1 + mod(2i + 2variant + operand,3))
                 for (i,mask) in enumerate(masks))
        end
    end
    ga = algebra(n,:ega)
    inputs = map(pair -> map(d -> multivector(ga,Dict(k=>Float64(v) for (k,v) in d);
                                              storage=:sparse),pair),references)
    plan = prepare_product(inputs[1]...; max_paths=PARALLEL_LIMITS.max_paths)
    truth = map(pair -> parallel_reference(pair...),references)
    expected = hcat(([Float64(get(d,UInt128(mask),0)) for mask in plan.output_masks] for d in truth)...)
    # Integer inputs, signs ±1, and at most 4096 paths: every Float64 sum is exact.
    9length(plan.paths) < 2^53 || error("float_exactness_budget")
    return (;n,family,inputs,plan,expected)
end

parallel_workspace(fixture) = ProductWorkspace(fixture.plan,fixture.inputs[1]...;
                                               max_bytes=PARALLEL_LIMITS.max_state_bytes)

# Function barrier keeps the inner loop specialized on the concrete workspace.
@noinline function parallel_chunk(workspace, inputs, first_job, count)
    output = Matrix{Float64}(undef,length(workspace.plan.output_masks),count)
    for j in 1:count
        pair = inputs[mod1(first_job+j-1,4)]
        values = run_product_values!(workspace,pair...)
        copyto!(view(output,:,j),values)
    end
    return output
end

function parallel_ranges(batch, lanes)
    cuts = [fld(batch*i,lanes) for i in 0:lanes]
    return [(cuts[i]+1,cuts[i+1]-cuts[i]) for i in 1:lanes]
end

function parallel_thread_batch(states, inputs, batch)
    ranges = parallel_ranges(batch,length(states))
    tasks = map(eachindex(states)) do i
        # Capture the workspace per task, never index scratch buffers by threadid().
        Threads.@spawn parallel_chunk(states[i],inputs,ranges[i]...)
    end
    return reduce(hcat,fetch.(tasks))
end

function parallel_oracle(output, fixture, batch)
    size(output) == (size(fixture.expected,1),batch) || return false
    return all(output[i,j] == fixture.expected[i,mod1(j,4)]
               for j in axes(output,2) for i in axes(output,1))
end

function parallel_admission(fixture,batch)
    length(fixture.plan.paths)*batch <= PARALLEL_LIMITS.max_batch_paths || return "batch_paths"
    8length(fixture.plan.output_masks)*batch <= PARALLEL_LIMITS.max_output_bytes || return "output_bytes"
    return "admitted"
end

const PARALLEL_REMOTE_STATE = Ref{Any}(nothing)
function parallel_remote_setup(n,family)
    GC.gc()
    prepared = @timed begin
        fixture = parallel_fixture(n,family)
        workspace = parallel_workspace(fixture)
        (;fixture,workspace)
    end
    PARALLEL_REMOTE_STATE[] = prepared.value
    first = @timed parallel_remote_chunk(1,4,nothing)
    parallel_oracle(first.value,prepared.value.fixture,4) || error("remote oracle")
    return (;setup_seconds=prepared.time,setup_bytes=prepared.bytes,
        first_seconds=first.time,first_compile_seconds=first.compile_time,
        first_bytes=first.bytes,state_bytes=Base.summarysize(prepared.value),
        maxrss=Sys.maxrss())
end

function parallel_remote_chunk(first_job,count,payload)
    state = PARALLEL_REMOTE_STATE[]
    inputs = isnothing(payload) ? state.fixture.inputs : payload
    return parallel_chunk(state.workspace,inputs,first_job,count)
end

parallel_remote_rss() = Sys.maxrss()

function parallel_remote_allocation_probe(first_job,count,payload)
    observed = @timed parallel_remote_chunk(first_job,count,payload)
    return (;bytes=observed.bytes,seconds=observed.time,gc_seconds=observed.gctime)
end

function parallel_serialization_probe(value)
    warm = IOBuffer(); serialize(warm,value)
    bytes = take!(warm); deserialize(IOBuffer(bytes))
    encoded = @timed begin
        io = IOBuffer(); serialize(io,value); take!(io)
    end
    decoded = @timed deserialize(IOBuffer(encoded.value))
    return (;bytes=length(encoded.value),encode_seconds=encoded.time,
        decode_seconds=decoded.time,encode_allocations=encoded.bytes,
        decode_allocations=decoded.bytes)
end
