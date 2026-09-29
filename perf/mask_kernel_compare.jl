using PerfChecker
using BenchmarkTools
using Statistics
using SHA

length(ARGS) == 1 || error("usage: mask_kernel_compare.jl OUTPUT.csv")
const output_path = abspath(ARGS[1])
const common = joinpath(@__DIR__, "features", "common_mask_kernel.jl")
push!(LOAD_PATH, dirname(@__DIR__))
include(common)

function source_fingerprint()
    paths = sort(vcat([joinpath(dirname(@__DIR__), "Project.toml"), @__FILE__, common],
        [joinpath(root, name)
         for (root, _, files) in walkdir(joinpath(dirname(@__DIR__), "src"))
         for name in files if endswith(name, ".jl")]))
    return bytes2hex(sha256(join((relpath(path, dirname(@__DIR__)) * ":" *
                             bytes2hex(sha256(read(path))) for path in paths), "\n")))
end

const initial_fingerprint = source_fingerprint()
const setup = Dict{Tuple{Int,Symbol,Symbol},Any}()
const states = Dict{Tuple{Int,Symbol,Symbol},Any}()

function kernel_executor(planned, config, setup_callback, workload_callback)
    n, workload, representation = planned.feature.options[:kernel_case]
    key = (n, workload, representation)
    if !haskey(states, key)
        observed = @timed mask_kernel_state(n, representation, workload)
        states[key] = observed.value
        setup[key] = (; seconds=observed.time, bytes=observed.bytes)
    end
    state = states[key]
    mask_kernel_oracle(state) || error("independent mask-kernel oracle failed")
    mask_kernel_workload(state)
    trial = @benchmark mask_kernel_workload($state) samples=31 evals=1 seconds=0.2
    mask_kernel_oracle(state) || error("post-sampling oracle failed")
    qualification = Dict{String,Any}(
        "correctness" => Dict("status" => "passed", "required" => true,
                              "message" => "independent basis-inversion oracle"),
        "execution" => Dict("mode" => "shared_process_pairwise_screening"),
        "source_fingerprint" => initial_fingerprint)
    return PerfChecker.CheckerResult([PerfChecker.to_table(trial)], nothing,
        [:garamon, :mask_representation], [PerfChecker.PackageSpec(name="Garamon")],
        [qualification])
end

mktempdir() do entrypoints
    cases = [(n, workload, representation)
             for n in (64, 65, 66, 96, 127, 128)
             for workload in (:targeted, :full)
             for representation in (:fixed, :arbitrary)]
    features = FeatureSpec[]
    for (n, workload, representation) in cases
        name = Symbol(:mask_, n, :_, workload, :_, representation)
        entrypoint = joinpath(entrypoints, "$(name).jl")
        open(entrypoint, "w") do io
            println(io, "include(", repr(common), ")")
            println(io, "perf_setup = () -> mask_kernel_state($n, :$representation, :$workload)")
            println(io, "perf_workload = state -> mask_kernel_workload(state)")
            println(io, "perf_oracle = state -> mask_kernel_oracle(state)")
        end
        push!(features, FeatureSpec(name;
            description="$(n)D diagonal mask $representation, $workload",
            backend=:benchmark, entrypoint,
            comparison_key="garamon/mask-kernel/$n/$workload/v1",
            oracle=OracleSpec(function_name=:perf_oracle),
            options=Dict(:tags => [:garamon, :mask_kernel, representation],
                         :kernel_case => (n, workload, representation),
                         :samples => 31, :evals => 1, :seconds => 0.2)))
    end
    package = PackageSuite("Garamon";
        worker_environment=joinpath(@__DIR__, "runner"),
        source=dirname(@__DIR__), versions=VersionNumber[],
        dev_sources=String[], features)
    suite = SoftwareSuite(:garamon_mask_representation, [package];
        description="Pairwise fixed-width versus arbitrary-precision blade masks")
    result = run_suite(suite; profile=:quick, strict=false, executor=kernel_executor)
    unchanged = source_fingerprint() == initial_fingerprint
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, "# Julia=$VERSION, source_sha256=$initial_fingerprint, unchanged=$unchanged, samples_max=31")
        println(io, "dimension,workload,representation,status,median_us,p95_us,memory_bytes,allocs,samples,setup_ms,setup_bytes")
        for (run, key) in zip(result.runs, cases)
            observed = get(setup, key, nothing)
            if run.status == :pass && run.result isa PerfChecker.CheckerResult
                table = only(run.result.tables)
                println(io, join((key..., run.status,
                    round(median(table.times) / 1000; digits=3),
                    round(quantile(table.times, 0.95) / 1000; digits=3),
                    Int(round(median(table.memory))),
                    Int(round(median(table.allocs))), length(table.times),
                    round(observed.seconds * 1000; digits=3), observed.bytes), ","))
            else
                println(io, join((key..., run.status, "", "", "", "", "", "", ""), ","))
            end
        end
    end
    println("PerfChecker verdict: ", suite_verdict(result),
            "; cases: ", length(result.runs),
            "; source unchanged: ", unchanged,
            "; CSV: ", output_path)
    unchanged || error("source changed during mask comparison")
    suite_passed(result) || error("mask comparison has failed cases")
end
