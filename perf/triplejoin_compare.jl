push!(LOAD_PATH, dirname(@__DIR__))
using Garamon
using LinearAlgebra
using Statistics
using SHA

# PerfChecker is needed by the controller, whereas its workers use the lean
# runner environment and include this file only for fixtures and the oracle.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using PerfChecker
end

# Compare whole-product materialization, on-demand scalar joins and a reusable
# exact path graph on the same outputs and repeated coefficient changes.
const TJ_DIMENSIONS = (2, 3, 4, 5, 8, 12, 32, 65, 128, 129)
const TJ_STRATEGIES = (:direct, :join3, :tripleplan)

function tj_source_hash()
    source = joinpath(@__DIR__, "..", "src")
    files = sort!(filter(path -> endswith(path, ".jl"), readdir(source; join=true)))
    bytes2hex(sha256(join((basename(path) * "\n" * read(path, String)
                          for path in files), "\n")))
end

function tj_masktype(n)
    n <= 64 && return UInt64
    n <= 128 && return UInt128
    BigInt
end

function tj_masks(n)
    K = tj_masktype(n)
    directions = unique((1, cld(n, 3), cld(2n, 3), n))
    bits = K[one(K) << (direction - 1) for direction in directions]
    left = unique(K[0, bits[1], bits[min(3, length(bits))],
                    bits[1] | bits[min(3, length(bits))]])
    middle = unique(K[0, bits[min(2, length(bits))], bits[end],
                      bits[min(2, length(bits))] | bits[end]])
    candidates = K[xor(a, b) for a in left for b in middle]
    right = unique(K[0, candidates[2], candidates[cld(length(candidates), 2)],
                     candidates[end]])
    targets = K[0, bits[1], bits[end], bits[1] | bits[end]]
    return (sort!(left), sort!(middle), sort!(right)),
           (Int[], [directions[1]], [directions[end]],
            sort!([directions[1], directions[end]])), targets
end

tj_value(t, operand, slot) = Int64(1 + mod(t + 2operand + slot, 3))

# Independent Euclidean Clifford-word oracle: count inversions explicitly.
function tj_indices(mask::Integer)
    result = Int[]
    index = 1
    while !iszero(mask)
        isodd(mask) && push!(result, index)
        mask >>= 1
        index += 1
    end
    result
end

function tj_reference_product(left::Dict{K,Int64},
                              right::Dict{K,Int64}) where K
    result = Dict{K,Int64}()
    for (a, av) in left, (b, bv) in right
        parity = sum((i > j for i in tj_indices(a) for j in tj_indices(b)); init=0)
        mask = xor(a, b)
        contribution = isodd(parity) ? -av*bv : av*bv
        result[mask] = get(result, mask, Int64(0)) + contribution
    end
    filter!(pair -> !iszero(last(pair)), result)
    result
end

function tj_fixture(n, output_count, horizon)
    n in TJ_DIMENSIONS || error("dimension outside protocol")
    output_count in (1, 4) || error("output count outside protocol")
    horizon in (1, 32, 1024, 10000) || error("horizon outside protocol")
    supports, indices, targets = tj_masks(n)
    ga = algebra(n, :ega)
    operands = ntuple(3) do operand
        multivector(ga, Dict(mask => Float64(tj_value(0, operand, slot))
            for (slot, mask) in enumerate(supports[operand])); storage=:sparse)
    end
    active_indices = indices[1:output_count]
    active_targets = targets[1:output_count]
    periodic = ntuple(3) do phase
        refs = ntuple(3) do operand
            Dict(mask => tj_value(phase, operand, slot)
                 for (slot, mask) in enumerate(supports[operand]))
        end
        exact = tj_reference_product(tj_reference_product(refs[1], refs[2]), refs[3])
        Float64[get(exact, mask, Int64(0)) for mask in active_targets]
    end
    expected = zeros(Float64, output_count)
    for t in 1:horizon
        phase = 1 + mod(t - 1, 3)
        for j in eachindex(expected)
            expected[j] += periodic[phase][j]
        end
    end
    return (; n, output_count, horizon, supports, indices=active_indices,
            operands, expected)
end

function tj_update!(state, t)
    for operand in 1:3
        values = state.operands[operand].values
        for (slot, mask) in enumerate(state.supports[operand])
            values[mask] = Float64(tj_value(t, operand, slot))
        end
    end
    nothing
end

function tj_trace(state, strategy)
    strategy in TJ_STRATEGIES || error("unknown strategy")
    a, b, c = state.operands
    artifact = if strategy == :join3
        prepare_expression(@ga (a*b)*c)
    elseif strategy == :tripleplan
        plan = prepare_triple_join(a, b, c, state.indices;
                                   max_probes=1 << 20, max_paths=1 << 16)
        TripleJoinWorkspace(plan, a, b, c; max_bytes=1 << 20)
    else
        nothing
    end
    totals = zeros(Float64, state.output_count)
    for t in 1:state.horizon
        tj_update!(state, t)
        if strategy == :direct
            result = (a*b)*c
            for (j, indices) in enumerate(state.indices)
                totals[j] += coefficient(result, indices)
            end
        elseif strategy == :join3
            for (j, indices) in enumerate(state.indices)
                totals[j] += evaluate(artifact; output=indices, strategy=:join3,
                                      max_support=4096, max_pairs=1 << 16)
            end
        else
            result = run_triple_join_values!(artifact, a, b, c)
            for j in eachindex(totals)
                totals[j] += result[j]
            end
        end
    end
    return (; totals, artifact)
end

function tj_oracle(state, strategy)
    tj_trace(state, strategy).totals == state.expected
end

function tj_setup(n, output_count, horizon, strategy, cold_path)
    BLAS.set_num_threads(1)
    state = tj_fixture(n, output_count, horizon)
    observed = @timed Base.invokelatest(tj_trace, state, strategy)
    observed.value.totals == state.expected || error("cold triple-join oracle failed")
    if !isfile(cold_path)
        open(cold_path, "w") do io
            println(io, "dimension,outputs,horizon,strategy,cold_ms,compile_ms,allocated_bytes,source_sha256")
            println(io, join((n, output_count, horizon, strategy, 1000observed.time,
                1000observed.compile_time, observed.bytes, tj_source_hash()), ','))
        end
    end
    state
end

function tj_campaign(output_directory; smoke=false)
    mkpath(output_directory)
    dimensions = smoke ? (2, 65) : TJ_DIMENSIONS
    horizons = smoke ? (1, 32) : (1, 32, 1024)
    cases = [(n, count, horizon, strategy)
             for n in dimensions for count in (1, 4)
             for horizon in horizons for strategy in TJ_STRATEGIES]
    if !smoke
        append!(cases, [(n, count, 10000, strategy) for n in (65, 128)
                        for count in (1, 4) for strategy in TJ_STRATEGIES])
    end
    fingerprint = tj_source_hash()
    mktempdir() do temporary
        features = FeatureSpec[]
        cold_paths = String[]
        for (n, count, horizon, strategy) in cases
            name = Symbol(:triplejoin_, n, :_, count, :_, horizon, :_, strategy)
            entrypoint = joinpath(temporary, "$name.jl")
            cold_path = joinpath(temporary, "$name.csv")
            push!(cold_paths, cold_path)
            open(entrypoint, "w") do io
                println(io, "include(", repr(abspath(@__FILE__)), ")")
                println(io, "perf_setup = () -> tj_setup(", n, ", ", count, ", ",
                        horizon, ", :", strategy, ", ", repr(cold_path), ")")
                println(io, "perf_workload = state -> tj_trace(state, :", strategy, ")")
                println(io, "perf_oracle = state -> tj_oracle(state, :", strategy, ")")
            end
            push!(features, FeatureSpec(name;
                description="Exact $n D triple product, $count outputs, $horizon calls; $strategy",
                backend=:benchmark, entrypoint,
                comparison_key="garamon/triplejoin/$(n)d/$count/$horizon/v1",
                oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:tags => [:garamon, :triplejoin, strategy],
                             :samples => 7, :evals => 1, :seconds => 0.05)))
        end
        package = PackageSuite("Garamon";
            worker_environment=joinpath(@__DIR__, "runner"), source=dirname(@__DIR__),
            versions=VersionNumber[], dev_sources=String[], features)
        suite = SoftwareSuite(:garamon_triplejoin, [package];
            description="Prepared exact triple joins under real coefficient updates")
        result = run_suite(suite; profile=:quick, strict=false)
        open(joinpath(output_directory, "warm.csv"), "w") do io
            println(io, "dimension,outputs,horizon,strategy,status,median_ms,p95_ms,allocated_bytes,allocs,samples")
            for (run, (n, count, horizon, strategy)) in zip(result.runs, cases)
                if run.status == :pass && run.result isa PerfChecker.CheckerResult
                    table = only(run.result.tables)
                    println(io, join((n, count, horizon, strategy, run.status,
                        median(table.times)/1e6, quantile(table.times, .95)/1e6,
                        Int(round(median(table.memory))),
                        Int(round(median(table.allocs))), length(table.times)), ','))
                else
                    println(io, join((n, count, horizon, strategy, run.status,
                                      "", "", "", "", ""), ','))
                end
            end
        end
        first_file = true
        open(joinpath(output_directory, "cold.csv"), "w") do io
            for path in cold_paths
                isfile(path) || continue
                lines = readlines(path)
                foreach(line -> println(io, line), first_file ? lines : lines[2:end])
                first_file = false
            end
        end
        open(joinpath(output_directory, "environment.txt"), "w") do io
            println(io, "Julia: ", VERSION, "\nPerfChecker: ", pkgversion(PerfChecker),
                    "\nCPU: ", Sys.cpu_info()[1].model,
                    "\nSource SHA256: ", fingerprint,
                    "\nScript SHA256: ", bytes2hex(sha256(read(@__FILE__))),
                    "\nVerdict: ", suite_verdict(result),
                    "\nCases: ", length(result.runs))
        end
        fingerprint == tj_source_hash() || error("source changed during campaign")
        suite_passed(result) || error("triple-join campaign has failed cases")
        println("Triple-join PerfChecker: ", suite_verdict(result),
                "; cases=", length(result.runs))
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) in (1, 2) || error("usage: triplejoin_compare.jl OUTPUT_DIRECTORY [--smoke]")
    tj_campaign(abspath(first(ARGS)); smoke=length(ARGS) == 2 && ARGS[2] == "--smoke")
end
