using Garamon
using LinearAlgebra
using Random
using Statistics
using SHA

# A separate Julia process per family isolates its train/test JIT history.
# No generated program is executed before its recorded first call.
const FAMILIES = [
    (name="vectors6", n=6, count=3, shape=:vector, operation=:geometric, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="bivectors12", n=12, count=4, shape=:bivector, operation=:geometric, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="mixed8", n=8, count=8, shape=:mixed, operation=:geometric, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="subalgebras12", n=12, count=8, shape=:subalgebra, operation=:geometric, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="dense4", n=4, count=8, shape=:mixed, operation=:geometric, scalar=Float64, storage=:dense, metric=:euclidean),
    (name="ceiling256", n=8, count=16, shape=:mixed, operation=:geometric, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="degenerate12", n=12, count=8, shape=:mixed, operation=:geometric, scalar=Float64, storage=:sparse, metric=:degenerate),
    (name="wedge12", n=12, count=8, shape=:bivector, operation=:wedge, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="contraction12", n=12, count=4, shape=:vector_bivector, operation=:left, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="ambient66", n=66, count=4, shape=:bivector, operation=:geometric, scalar=Float64, storage=:sparse, metric=:euclidean),
    (name="float32_12", n=12, count=4, shape=:bivector, operation=:geometric, scalar=Float32, storage=:sparse, metric=:weighted),
    (name="rational12", n=12, count=4, shape=:bivector, operation=:geometric, scalar=Rational{Int}, storage=:sparse, metric=:weighted),
]

function source_fingerprint()
    source = joinpath(@__DIR__, "..", "src")
    files = sort!(filter(p -> endswith(p, ".jl"), readdir(source; join=true)))
    return bytes2hex(sha256(join((basename(p) * "\n" * read(p, String) for p in files), "\n")))
end

function rss_bytes()
    isfile("/proc/self/status") || return -1
    found = match(r"(?m)^VmRSS:\s+(\d+)\s+kB", read("/proc/self/status", String))
    return found === nothing ? -1 : 1024parse(Int, found.captures[1])
end

function random_support(rng, n, count, shape)
    K = n <= 64 ? UInt64 : BigInt
    if shape == :subalgebra
        directions = sort!(randperm(rng, n)[1:3])
        return K[sum((one(K) << (directions[j] - 1)) for j in 1:3
                     if !iszero(mask & (1 << (j - 1))); init=zero(K)) for mask in 0:7]
    end
    support = Set{K}()
    while length(support) < count
        grade = shape == :vector ? 1 : shape == :bivector ? 2 : rand(rng, 0:min(n, 5))
        mask = zero(K)
        for i in randperm(rng, n)[1:grade]
            mask |= one(K) << (i - 1)
        end
        push!(support, mask)
    end
    return sort!(collect(support))
end

function make_case(rng, family, used)
    T, n = family.scalar, family.n
    diagonal = family.metric == :degenerate ? T[iszero(i % 3) ? 0 : isodd(i) ? 1 : -1 for i in 1:n] :
               family.metric == :weighted ? T[isodd(i) ? i % 3 + 1 : -(i % 3 + 1) for i in 1:n] : ones(T, n)
    ga = algebra(Diagonal(diagonal))
    while true
        ashape = family.shape == :vector_bivector ? :vector : family.shape
        bshape = family.shape == :vector_bivector ? :bivector : family.shape
        amasks = random_support(rng, n, family.count, ashape)
        bmasks = family.shape == :subalgebra ? copy(amasks) : random_support(rng, n, family.count, bshape)
        signature = (Tuple(amasks), Tuple(bmasks))
        signature in used && continue
        push!(used, signature)
        # Small integer-valued coefficients keep the correctness oracle exact
        # even for floating point families, including differing summation order.
        a = multivector(ga, Dict(m => T(rand(rng, 1:5)) for m in amasks); storage=family.storage)
        b = multivector(ga, Dict(m => T(rand(rng, 1:5)) for m in bmasks); storage=family.storage)
        return (; a, b, signature)
    end
end

function observe(f, args...)
    @nospecialize f args
    # invokelatest prevents caller inference from compiling the target before
    # the timed region. The same dispatch boundary is used for warm samples.
    return @timed Base.invokelatest(f, args...)
end

function hot_measure(f, args...; samples=31)
    @nospecialize f args
    for _ in 1:3
        observe(f, args...)
    end
    samples_taken = [observe(f, args...) for _ in 1:samples]
    return (us=1e6median(x.time for x in samples_taken),
            p95_us=1e6quantile([x.time for x in samples_taken], .95),
            bytes=median(x.bytes for x in samples_taken))
end

prepare_call(a, b, op) = prepare_product(a, b; operation=op, max_paths=256)
generate_call(plan) = generate_product(plan; max_paths=256)
function direct_call(a, b, op)
    op == :geometric && return a * b
    op == :wedge && return wedge(a, b)
    op == :left && return left_contraction(a, b)
    error("unsupported benchmark operation")
end

function worker(family_index, output; train_count=12, test_count=12, samples=31, seed=20260927)
    family = FAMILIES[family_index]
    BLAS.set_num_threads(1)
    train_rng = Xoshiro(seed + 1000family_index)
    test_rng = Xoshiro(seed + 1000family_index + 500_000)
    used = Set{Any}()
    # Construct the entire holdout before timing. Selection never sees its costs.
    training = [make_case(train_rng, family, used) for _ in 1:train_count]
    testing = [make_case(test_rng, family, used) for _ in 1:test_count]
    isempty(intersect(Set(x.signature for x in training), Set(x.signature for x in testing))) ||
        error("train/test support leakage")
    identity_observation = observe(identity, 1)
    hasproperty(identity_observation, :compile_time) || error("Julia @timed compile counters required")
    hot_measure(identity, 1; samples=3)
    fingerprint = source_fingerprint()
    train_topologies, train_program_types = Set{Any}(), Set{Any}()
    seen_topologies, seen_program_types = Set{Any}(), Set{Any}()
    held_plans, held_programs = Any[], Any[]
    rows = NamedTuple[]
    GC.gc()
    initial_jit, initial_rss = Base.jit_total_bytes(), rss_bytes()
    for (phase, cases) in ((:train, training), (:test, testing))
        for (case_id, case) in enumerate(cases)
            a, b = case.a, case.b
            jit_start = Base.jit_total_bytes()
            direct_first = observe(direct_call, a, b, family.operation)
            prepared = observe(prepare_call, a, b, family.operation)
            plan = prepared.value
            generated = observe(generate_call, plan)
            program = generated.value
            topology = (typeof(program).parameters[1], length(plan.output_masks))
            program_type = (typeof(program), typeof(a), typeof(b))
            topology_seen = topology in seen_topologies
            program_type_seen = program_type in seen_program_types
            train_topology = topology in train_topologies
            train_program_type = program_type in train_program_types
            jit_before_call = Base.jit_total_bytes()
            first_call = observe(run_generated_product, program, a, b)
            jit_call_bytes = Base.jit_total_bytes() - jit_before_call
            first_call.value == direct_first.value || error("generated oracle failed for $(family.name)/$phase/$case_id")
            run_product(plan, a, b) == direct_first.value || error("prepared oracle failed")
            direct_hot = hot_measure(direct_call, a, b, family.operation; samples)
            plan_hot = hot_measure(prepare_call, a, b, family.operation; samples)
            generation_hot = hot_measure(generate_call, plan; samples)
            prepared_hot = hot_measure(run_product, plan, a, b; samples)
            generated_hot = hot_measure(run_generated_product, program, a, b; samples)
            push!(held_plans, plan)
            push!(held_programs, program)
            push!(seen_topologies, topology)
            push!(seen_program_types, program_type)
            if phase == :train
                push!(train_topologies, topology)
                push!(train_program_types, program_type)
            end
            GC.gc()
            row = (family=family.name, phase=String(phase), case_id=case_id,
                dimension=family.n, scalar=string(family.scalar), storage=String(family.storage),
                operation=String(family.operation), left_support=length(plan.left_masks),
                right_support=length(plan.right_masks), outputs=length(plan.output_masks), paths=length(plan.paths),
                topology_seen=topology_seen, program_type_seen=program_type_seen,
                topology_in_train=train_topology, program_type_in_train=train_program_type,
                unique_topologies=length(seen_topologies), unique_program_types=length(seen_program_types),
                prepare_first_ms=1e3prepared.time, prepare_compile_ms=1e3prepared.compile_time,
                generate_first_ms=1e3generated.time, generate_compile_ms=1e3generated.compile_time,
                generated_first_ms=1e3first_call.time, generated_compile_ms=1e3first_call.compile_time,
                generated_recompile_ms=1e3first_call.recompile_time,
                generated_first_allocated_bytes=first_call.bytes, jit_call_bytes=jit_call_bytes,
                direct_hot_us=direct_hot.us, direct_hot_bytes=direct_hot.bytes,
                prepare_hot_us=plan_hot.us, prepare_hot_bytes=plan_hot.bytes,
                generate_hot_us=generation_hot.us, generate_hot_bytes=generation_hot.bytes,
                prepared_hot_us=prepared_hot.us, prepared_hot_bytes=prepared_hot.bytes,
                generated_hot_us=generated_hot.us, generated_p95_us=generated_hot.p95_us,
                generated_hot_bytes=generated_hot.bytes,
                plan_bytes=Base.summarysize(plan), program_bytes=Base.summarysize(program),
                program_incremental_bytes=Base.summarysize((plan, program))-Base.summarysize(plan),
                retained_plans_bytes=Base.summarysize(held_plans),
                retained_joint_bytes=Base.summarysize((held_plans, held_programs)),
                jit_case_bytes=Base.jit_total_bytes()-jit_start,
                jit_cumulative_bytes=Base.jit_total_bytes()-initial_jit,
                rss_delta_bytes=rss_bytes()-initial_rss,
                source_sha256=fingerprint)
            push!(rows, row)
            println(stderr, family.name, " ", phase, " ", case_id, "/", length(cases),
                    " paths=", length(plan.paths), " shared=", topology_seen,
                    " first_ms=", round(row.generated_first_ms; digits=2))
        end
    end
    fingerprint == source_fingerprint() || error("source changed during the family campaign; rerun for a consistent fingerprint")
    open(output, "w") do io
        println(io, join(keys(first(rows)), ','))
        foreach(row -> println(io, join(values(row), ',')), rows)
    end
end

function read_rows(path)
    lines = readlines(path)
    columns = Symbol.(split(first(lines), ','))
    return [Dict(zip(columns, split(line, ','))) for line in lines[2:end]]
end

number(row, key) = parse(Float64, row[key])
function predicted_total_us(row, cap, repetitions)
    setup = number(row, :prepare_hot_us)
    if cap > 0 && number(row, :paths) <= cap
        # First-call excess is observed in the actual train-then-test JIT history.
        cold_excess = max(0., 1000number(row, :generated_first_ms)-number(row, :generated_hot_us))
        return setup + number(row, :generate_hot_us) + cold_excess + repetitions*number(row, :generated_hot_us)
    end
    return setup + repetitions*number(row, :prepared_hot_us)
end

function summarize(rows, output)
    open(output, "w") do io
        println(io, "family,repetitions,selected_cap,train_cost_us,test_cost_us,test_prepared_us,test_direct_us,test_unconditional_generated_us,test_best_per_case_us,train_topologies,test_cases,test_topology_hits,test_program_type_hits,test_compile_ms,test_jit_call_bytes,total_jit_bytes,retained_plans_bytes,retained_joint_bytes")
        for family in FAMILIES
            train = filter(row -> row[:family] == family.name && row[:phase] == "train", rows)
            test = filter(row -> row[:family] == family.name && row[:phase] == "test", rows)
            isempty(train) && continue
            for repetitions in (32, 1024, 10000)
                caps = (0, 16, 64, 128, 256)
                costs = [sum(predicted_total_us(row, cap, repetitions) for row in train) for cap in caps]
                selected = caps[argmin(costs)]
                test_cost = sum(predicted_total_us(row, selected, repetitions) for row in test)
                baseline = sum(predicted_total_us(row, 0, repetitions) for row in test)
                direct = repetitions*sum(number(row, :direct_hot_us) for row in test)
                generated = sum(predicted_total_us(row, 256, repetitions) for row in test)
                oracle = sum(min(predicted_total_us(row, 0, repetitions),
                                 predicted_total_us(row, 256, repetitions)) for row in test)
                println(io, join((family.name, repetitions, selected, minimum(costs), test_cost,
                    baseline, direct, generated, oracle, train[end][:unique_topologies], length(test),
                    count(row -> row[:topology_in_train] == "true", test),
                    count(row -> row[:program_type_in_train] == "true", test),
                    sum(number(row, :generated_compile_ms) for row in test),
                    sum(number(row, :jit_call_bytes) for row in test),
                    test[end][:jit_cumulative_bytes], test[end][:retained_plans_bytes],
                    test[end][:retained_joint_bytes]), ','))
            end
        end
    end
end

function campaign(output_directory; train_count=12, test_count=12, samples=31, seed=20260927)
    mkpath(output_directory)
    output = joinpath(output_directory, "cases.csv")
    mktempdir() do temporary
        open(output, "w") do io
            for family_index in eachindex(FAMILIES)
                partial = joinpath(temporary, "family$family_index.csv")
                run(`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(dirname(@__DIR__)) $(@__FILE__) --worker $family_index $partial $train_count $test_count $samples $seed`)
                lines = readlines(partial)
                foreach(line -> println(io, line), family_index == 1 ? lines : lines[2:end])
                flush(io)
            end
        end
    end
    rows = read_rows(output)
    summarize(rows, joinpath(output_directory, "selection.csv"))
    open(joinpath(output_directory, "environment.txt"), "w") do io
        println(io, "Julia: ", VERSION, "\nCPU: ", Sys.cpu_info()[1].model,
                "\nKernel: ", Sys.KERNEL, "\nMachine: ", Sys.MACHINE,
                "\nSeed: ", seed, "\nTrain/test cases per family: ", train_count, "/", test_count,
                "\nWarm samples per case: ", samples, "\nThreads: 1; BLAS threads: 1",
                "\nSource fingerprints: ", join(unique(row[:source_sha256] for row in rows), ";"),
                "\nBenchmark SHA256: ", bytes2hex(sha256(read(@__FILE__))))
    end
    println("Generated-program cost campaign passed: ", length(rows), " exact oracle checks; ", output_directory)
end

if !isempty(ARGS) && ARGS[1] == "--worker"
    worker(parse(Int, ARGS[2]), ARGS[3]; train_count=parse(Int, ARGS[4]),
           test_count=parse(Int, ARGS[5]), samples=parse(Int, ARGS[6]), seed=parse(Int, ARGS[7]))
elseif length(ARGS) == 1
    campaign(abspath(only(ARGS)))
else
    error("usage: julia --startup-file=no --project=. perf/generated_costs.jl OUTPUT_DIRECTORY")
end
