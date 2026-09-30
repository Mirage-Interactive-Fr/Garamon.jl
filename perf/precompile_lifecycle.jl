using SHA, TOML

const LIFECYCLE_ROOT = dirname(@__DIR__)
const LIFECYCLE_KERNEL = joinpath(@__DIR__, "precompile_lifecycle_kernel.jl")
const LIFECYCLE_WORKER = joinpath(@__DIR__, "precompile_lifecycle_worker.jl")
const LIFECYCLE_UUID = "37cd90f4-a5bd-4f0c-9984-6956352c9fdc"

function lifecycle_source_hash()
    directory = joinpath(LIFECYCLE_ROOT, "src")
    paths = sort!(filter(p -> endswith(p, ".jl"), readdir(directory; join=true)))
    bytes2hex(sha256(join((basename(p)*"\n"*read(p,String) for p in paths), "\n")))
end

function lifecycle_cache_info(depot)
    files = String[]
    directory = joinpath(depot, "compiled")
    if isdir(directory)
        for (root, _, names) in walkdir(directory), name in names
            (endswith(name, ".ji") || endswith(name, ".so") ||
             endswith(name, ".dylib") || endswith(name, ".dll")) &&
                push!(files, joinpath(root, name))
        end
    end
    sort!(files)
    digest = bytes2hex(sha256(join((relpath(f, depot)*":"*bytes2hex(sha256(read(f)))
                                  for f in files), "\n")))
    (; digest, files=length(files), bytes=sum(filesize, files; init=0),
       native_bytes=sum(filesize(f) for f in files if !endswith(f, ".ji"); init=0),
       harness_bytes=sum(filesize(f) for f in files if occursin("GaramonLifecycle", f); init=0))
end

function lifecycle_environment(directory, catalogue; dependency_manifest=nothing)
    package = joinpath(directory, "GaramonLifecycle")
    mkpath(joinpath(package, "src"))
    original = TOML.parsefile(joinpath(LIFECYCLE_ROOT, "Project.toml"))
    deps = Dict("Garamon" => original["uuid"],
                "LinearAlgebra" => original["deps"]["LinearAlgebra"],
                "StaticArrays" => original["deps"]["StaticArrays"],
                "PrecompileTools" => "aea7be01-6a6a-4083-8856-8a6e6704d82a")
    open(joinpath(package, "Project.toml"), "w") do io
        TOML.print(io, Dict("name" => "GaramonLifecycle", "uuid" => LIFECYCLE_UUID,
                            "version" => "0.0.1", "deps" => deps))
    end
    # Reuse the exact pinned dependency closure, adding only a path entry for the
    # live Garamon checkout. No resolver/network/package installation is needed.
    # Pkg installs Git dependencies without their development Manifest.toml.
    # The active benchmark project still owns the exact resolved closure.
    source_manifest=joinpath(LIFECYCLE_ROOT,"Manifest.toml")
    active=Base.active_project()
    benchmark_manifest=isnothing(active) ? "" : joinpath(dirname(active),"Manifest.toml")
    manifest_path=isnothing(dependency_manifest) ?
        (isfile(source_manifest) ? source_manifest : benchmark_manifest) :
        abspath(dependency_manifest)
    isfile(manifest_path) || error("an active resolved Manifest.toml is required for the precompilation catalogue")
    manifest = TOML.parsefile(manifest_path)
    # A profiling worker can own a smaller environment than its target. Use
    # the campaign's pinned closure explicitly in that case, and reject an
    # incomplete closure before spawning the native precompilation process.
    for name in union(keys(deps), keys(original["deps"]))
        name == "Garamon" && continue
        entries=get(manifest["deps"],name,Any[])
        length(entries)==1 || error("catalogue dependency manifest must contain exactly one entry for "*name)
    end
    delete!(manifest, "project_hash")
    manifest["deps"]["Garamon"] = [Dict("uuid" => original["uuid"],
        "version" => original["version"], "path" => LIFECYCLE_ROOT,
        "deps" => sort!(collect(keys(original["deps"]))))]
    open(joinpath(package, "Manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    open(joinpath(package, "src", "GaramonLifecycle.jl"), "w") do io
        println(io, "module GaramonLifecycle\nusing PrecompileTools")
        println(io, "include(", repr(LIFECYCLE_KERNEL), ")")
        catalogue && println(io, "@compile_workload lifecycle_catalog()")
        println(io, "end")
    end
    depot = joinpath(directory, "depot")
    mkpath(depot)
    # Share immutable installed package/artifact contents, never compiled caches.
    for name in ("packages", "artifacts")
        source = findfirst(d -> isdir(joinpath(d, name)), DEPOT_PATH)
        source === nothing || symlink(joinpath(DEPOT_PATH[source], name), joinpath(depot, name))
    end
    depots = join(vcat(depot, DEPOT_PATH[2:end]), ':')
    env = Dict("JULIA_DEPOT_PATH" => depots, "JULIA_LOAD_PATH" => "@:@stdlib",
               "JULIA_PKG_PRECOMPILE_AUTO" => "0", "JULIA_NUM_THREADS" => "1",
               "JULIA_NUM_PRECOMPILE_TASKS" => "1", "JULIA_PRECOMPILE_THREADS" => "1",
               "OPENBLAS_NUM_THREADS" => "1")
    (; package, depot, env)
end

function lifecycle_run(command, env, error_log)
    started = time_ns()
    process_env = merge(env, Dict("LIFECYCLE_LAUNCH_UNIX" => string(time())))
    try
        open(error_log, "w") do io
            run(pipeline(addenv(command, process_env); stdout=devnull, stderr=io))
        end
    catch
        print(stderr, read(error_log, String))
        rethrow()
    end
    (time_ns()-started)/1e6
end

function lifecycle_csv(path, rows)
    isempty(rows) && error("empty result table")
    keys_sorted = sort!(collect(keys(first(rows))))
    open(path, "w") do io
        println(io, join(keys_sorted, ','))
        for row in rows
            println(io, join((get(row, key, "") for key in keys_sorted), ','))
        end
    end
end

function lifecycle_campaign(output; smoke=false)
    mkpath(output)
    fingerprint = lifecycle_source_hash()
    all_rows, sessions, builds = Dict[], Dict[], Dict[]
    configurations = smoke ? [("native_catalogue", true, true)] :
        [("ir_empty", false, false), ("native_empty", true, false),
         ("ir_catalogue", false, true), ("native_catalogue", true, true)]
    repetitions = smoke ? 1 : 3
    strategies = smoke ? (:generated,) : (:prepared, :generated, :workspace)
    mktempdir() do temporary
        environments = Dict()
        for (name, native, catalogue) in configurations
            environment = lifecycle_environment(joinpath(temporary, name), catalogue)
            environments[name] = environment
            build_result = joinpath(temporary, name*"-build.toml")
            expression = "using TOML; m = @timed Base.compilecache(Base.identify_package(\"GaramonLifecycle\")); " *
                         "open("*repr(build_result)*", \"w\") do io; TOML.print(io, Dict(\"precompile_ms\" => 1000m.time, \"controller_allocated_bytes\" => m.bytes)); end"
            flag = native ? "yes" : "no"
            command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --pkgimages=$flag --project=$(environment.package) -e $expression`
            wall_ms = lifecycle_run(command, environment.env, joinpath(temporary, "error.log"))
            info = lifecycle_cache_info(environment.depot)
            build = TOML.parsefile(build_result)
            merge!(build, Dict("configuration" => name, "wall_ms" => wall_ms,
                "cache_bytes" => info.bytes, "native_bytes" => info.native_bytes,
                "harness_bytes" => info.harness_bytes, "cache_files" => info.files))
            push!(builds, build)
            println("Built ", name, ": ", round(wall_ms/1000; digits=2), " s")
            flush(stdout)
        end
        # Reverse configuration order on alternate restarts to reduce drift bias.
        for repetition in 1:repetitions
            ordered = isodd(repetition) ? configurations : reverse(configurations)
            for (name, native, _) in ordered, scenario in (:known, :dynamic), strategy in strategies
                environment = environments[name]
                before = lifecycle_cache_info(environment.depot)
                result_path = joinpath(temporary, "worker.toml")
                flag = native ? "yes" : "no"
                # Use normal write-capable loading. An unchanged disk cache is
                # observed, not forced by --compiled-modules/pkgimages=existing.
                command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --compiled-modules=yes --pkgimages=$flag --project=$(environment.package) $LIFECYCLE_WORKER $scenario $strategy $repetition $result_path`
                wall_ms = lifecycle_run(command, environment.env, joinpath(temporary, "error.log"))
                result = TOML.parsefile(result_path)
                after = lifecycle_cache_info(environment.depot)
                before.digest == after.digest || error("a runtime session modified precompile cache")
                for row in result["rows"]
                    row["configuration"] = name
                    push!(all_rows, row)
                end
                metadata = result["metadata"]
                merge!(metadata, Dict("configuration" => name, "scenario" => String(scenario),
                    "strategy" => String(strategy), "repetition" => repetition,
                    "process_wall_ms" => wall_ms, "cache_unchanged" => true))
                push!(sessions, metadata)
                println("Validated ", name, " / ", scenario, " / ", strategy, " / restart ", repetition)
                flush(stdout)
            end
        end
    end
    fingerprint == lifecycle_source_hash() || error("source changed during campaign")
    lifecycle_csv(joinpath(output, "precompile-lifecycle-stages.csv"), all_rows)
    lifecycle_csv(joinpath(output, "precompile-lifecycle-sessions.csv"), sessions)
    lifecycle_csv(joinpath(output, "precompile-lifecycle-builds.csv"), builds)
    open(joinpath(output, "precompile-lifecycle-environment.txt"), "w") do io
        println(io, "Julia: ", VERSION, "\nCPU: ", Sys.cpu_info()[1].model,
                "\nSource SHA256: ", fingerprint,
                "\nManifest SHA256: ", bytes2hex(sha256(read(joinpath(LIFECYCLE_ROOT, "Manifest.toml")))),
                "\nPrecompileTools: ", TOML.parsefile(joinpath(LIFECYCLE_ROOT, "Manifest.toml"))["deps"]["PrecompileTools"][1]["version"],
                "\nPlatform: ", Sys.MACHINE, " / ", Sys.KERNEL,
                "\nJulia child command: ", Base.julia_cmd(),
                "\nSessions: ", length(sessions), "\nRows: ", length(all_rows),
                "\nSamples per hot stage: 5\nThreads: 1\nSource unchanged: true",
                "\nTemporary depots removed: true\nSmoke: ", smoke)
        for path in (@__FILE__, LIFECYCLE_KERNEL, LIFECYCLE_WORKER)
            println(io, basename(path), " SHA256: ", bytes2hex(sha256(read(path))))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoke = length(ARGS) == 2 && first(ARGS) == "--smoke"
    (length(ARGS) == 1 || smoke) || error("usage: julia --startup-file=no perf/precompile_lifecycle.jl [--smoke] OUTPUT_DIRECTORY")
    lifecycle_campaign(abspath(last(ARGS)); smoke)
end
