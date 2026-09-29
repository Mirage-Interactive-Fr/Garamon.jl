using PerfChecker, Statistics, SHA

const CM_CPP_ROOT = get(ENV,"GARAMON_CPP_ROOT",joinpath(homedir(),"Gits","garamon"))
const CM_JULIA_ROOT = dirname(@__DIR__)
const CM_BUDGET = 256 << 20

function cm_tree_bytes(directory)
    total=0
    ignore_disappeared(error) = error isa Base.IOError && error.code==Base.UV_ENOENT ? nothing : throw(error)
    for (root,_,files) in walkdir(directory;onerror=ignore_disappeared), file in files
        path=joinpath(root,file)
        if !islink(path)
            try
                total+=filesize(path)
            catch error
                ignore_disappeared(error)
            end
        end
    end
    total
end

function cm_guarded_process(command,directory,budget; working_directory=directory)
    log=joinpath(directory,"command.log")
    started=time_ns()
    peak=cm_tree_bytes(directory)
    process=nothing
    try
        open(log,"w") do io
            task=Cmd(command;dir=working_directory,detach=true)
            task=addenv(task,"TMPDIR"=>directory,"OMP_NUM_THREADS"=>"1")
            process=run(pipeline(task;stdout=io,stderr=io);wait=false)
            while process_running(process)
                peak=max(peak,cm_tree_bytes(directory))
                peak<=budget || error("C++ disk budget exceeded: $peak > $budget")
                (time_ns()-started)/1e9<=240 || error("C++ command exceeded 240 seconds")
                sleep(0.1)
            end
            wait(process)
            success(process) || error("C++ command failed: "*read(log,String))
        end
    finally
        if process!==nothing && process_running(process)
            # All children of this explicitly detached command share its group.
            ccall(:kill,Cint,(Cint,Cint),-getpid(process),Base.SIGKILL)
            wait(process)
        end
    end
    peak=max(peak,cm_tree_bytes(directory))
    peak<=budget || error("C++ disk budget exceeded at command completion")
    (;time_ms=(time_ns()-started)/1e6,peak_bytes=peak)
end

function cm_driver_source_hash()
    dir=joinpath(CM_JULIA_ROOT,"src")
    paths=sort!(filter(p->endswith(p,".jl"),readdir(dir;join=true)))
    bytes2hex(sha256(join((basename(p)*"\n"*read(p,String) for p in paths),"\n")))
end

function cm_append_cold(destination,paths)
    first_write=!isfile(destination)
    open(destination,"a") do io
        for path in paths
            isfile(path) || error("missing cold record $path")
            lines=readlines(path)
            foreach(line->println(io,line),first_write ? lines : lines[2:end])
            first_write=false
        end
    end
end

function cm_campaign(output;smoke=false,case_order=:forward)
    case_order in (:forward,:reverse) || error("case_order must be :forward or :reverse")
    mkpath(output)
    fingerprint=cm_driver_source_hash()
    cpp_revision=strip(read(`git -C $CM_CPP_ROOT rev-parse HEAD`,String))
    disk_before=read(`df -B1 $CM_CPP_ROOT`,String)
    du_before=read(`du -sb $CM_CPP_ROOT`,String)
    cold_file=joinpath(output,"cpp-matched-cold.csv")
    warm_file=joinpath(output,"cpp-matched-warm.csv")
    build_file=joinpath(output,"cpp-matched-builds.csv")
    native_file=joinpath(output,"cpp-matched-native.csv")
    samples_file=joinpath(output,"cpp-matched-samples.csv")
    native_samples_file=joinpath(output,"cpp-matched-native-samples.csv")
    any(isfile,(cold_file,warm_file,build_file,native_file,samples_file,native_samples_file)) && error("choose an output directory without prior cpp-matched CSV files")
    generator_build=joinpath(CM_CPP_ROOT,"build")
    mkpath(generator_build)
    build_rows=NamedTuple[]
    command=`cmake -S $CM_CPP_ROOT -B $generator_build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release`
    result=cm_guarded_process(command,generator_build,64<<20)
    push!(build_rows,(dimension=0,stage=:generator_configure,time_ms=result.time_ms,peak_bytes=result.peak_bytes,retained_bytes=cm_tree_bytes(generator_build),library_sha256=""))
    result=cm_guarded_process(`cmake --build $generator_build --clean-first --parallel 1`,generator_build,64<<20)
    push!(build_rows,(dimension=0,stage=:generator_compile,time_ms=result.time_ms,peak_bytes=result.peak_bytes,retained_bytes=cm_tree_bytes(generator_build),library_sha256=""))
    println("Generator compiled; retained bytes: ",cm_tree_bytes(generator_build));flush(stdout)
    open(warm_file,"w") do io
        println(io,"dimension,corpus,horizon,strategy,comparison_group,status,median_ms,p95_ms,allocated_julia_bytes,samples")
    end
    open(native_file,"w") do io
        println(io,"dimension,corpus,horizon,stage,median_ms,p95_ms,samples,peak_rss_bytes,exact")
    end
    open(samples_file,"w") do io
        println(io,"dimension,corpus,horizon,strategy,sample,time_ns,gc_time_ns,allocated_julia_bytes,julia_allocations")
    end
    open(native_samples_file,"w") do io
        println(io,"dimension,corpus,horizon,stage,sample,time_ms")
    end
    dimensions=smoke ? (3,) : (3,7)
    case_order==:reverse && (dimensions=reverse(dimensions))
    horizons=smoke ? (32,) : (1,32,1024,10000)
    corpora=smoke ? (:mixed,) : (:mixed,:vectors)
    features_total=0
    for n in dimensions
        mktempdir(;prefix="garamon-cpp-d$n-") do temporary
            symlink(joinpath(CM_CPP_ROOT,"data"),joinpath(temporary,"data"))
            generation=joinpath(temporary,"generation")
            mkpath(joinpath(generation,"output"))
            config=joinpath(CM_CPP_ROOT,"conf","e$(n)ga.conf")
            binary=joinpath(generator_build,"garamon_generator")
            result=cm_guarded_process(`$binary $config`,temporary,CM_BUDGET;working_directory=generation)
            push!(build_rows,(dimension=n,stage=:algebra_generation,time_ms=result.time_ms,peak_bytes=result.peak_bytes,retained_bytes=cm_tree_bytes(temporary),library_sha256=""))
            generated=joinpath(generation,"output","garamon_e$(n)ga","src")
            compiled=joinpath(temporary,"compiled")
            source=joinpath(CM_CPP_ROOT,"benchmark")
            command=`cmake -S $source -B $compiled -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DGARAMON_DIMENSION=$n -DGARAMON_GENERATED_SOURCE=$generated`
            result=cm_guarded_process(command,temporary,CM_BUDGET)
            push!(build_rows,(dimension=n,stage=:benchmark_configure,time_ms=result.time_ms,peak_bytes=result.peak_bytes,retained_bytes=cm_tree_bytes(temporary),library_sha256=""))
            for target in ("garamon_packed","garamon_upstream","garamon_native")
                result=cm_guarded_process(`cmake --build $compiled --target $target --parallel 1`,temporary,CM_BUDGET)
                library=joinpath(compiled,target=="garamon_native" ? target : "lib$target.so")
                push!(build_rows,(dimension=n,stage=Symbol(target*"_compile"),time_ms=result.time_ms,peak_bytes=result.peak_bytes,retained_bytes=cm_tree_bytes(temporary),library_sha256=bytes2hex(sha256(read(library)))))
            end
            packed_path=joinpath(compiled,"libgaramon_packed.so")
            native_path=joinpath(compiled,"libgaramon_upstream.so")
            cases=[(corpus,horizon,strategy) for corpus in corpora for horizon in horizons
                   for strategy in (:julia_jit,:julia_precompile,:cpp_packed,:cpp_upstream)]
            case_order==:reverse && reverse!(cases)
            features=FeatureSpec[]
            cold_paths=String[]
            common=joinpath(@__DIR__,"cpp_matched_common.jl")
            for (corpus,horizon,strategy) in cases
                id=Symbol(:cpp_,n,:_,corpus,:_,horizon,:_,strategy)
                entry=joinpath(temporary,"$id.jl")
                cold=joinpath(temporary,"$id.csv")
                push!(cold_paths,cold)
                open(entry,"w") do io
                    println(io,"include(",repr(common),")")
                    println(io,"perf_setup() = cm_setup(",n,", :",corpus,", ",horizon,", :",strategy,", ",repr(packed_path),", ",repr(native_path),", ",repr(cold),")")
                    println(io,"perf_workload(state) = cm_workload(state)\nperf_oracle(state) = cm_oracle(state)")
                end
                group=strategy==:cpp_upstream ? "unpaired-upstream-mvec" : "matched-packed-paths-v1"
                push!(features,FeatureSpec(id;backend=:benchmark,entrypoint=entry,
                    description="$group; EGA$n $corpus $horizon outputs $strategy",
                    comparison_key="garamon/cpp/$group/$n/$corpus/$horizon",
                    oracle=OracleSpec(function_name=:perf_oracle),state_policy=:reuse,
                    options=Dict(:samples=>31,:evals=>1,:seconds=>5.0)))
            end
            package=PackageSuite("Garamon";worker_environment=joinpath(@__DIR__,"runner"),
                source=CM_JULIA_ROOT,versions=VersionNumber[],dev_sources=String[],features)
            suite=SoftwareSuite(Symbol(:garamon_cpp_,n),[package])
            result=run_suite(suite;profile=:quick,strict=false)
            write_suite_json(result,joinpath(output,"cpp-matched-qualification-d$n.json"))
            features_total+=length(result.runs)
            open(warm_file,"a") do io
                for (run,(corpus,horizon,strategy)) in zip(result.runs,cases)
                    group=strategy==:cpp_upstream ? "unpaired-upstream-mvec" : "matched-packed-paths-v1"
                    if run.status==:pass && run.result isa PerfChecker.CheckerResult
                        samples=only(run.result.tables)
                        open(samples_file,"a") do raw
                            for i in eachindex(samples.times)
                                println(raw,join((n,corpus,horizon,strategy,i,samples.times[i],
                                    samples.gctimes[i],samples.memory[i],samples.allocs[i]),','))
                            end
                        end
                        println(io,join((n,corpus,horizon,strategy,group,run.status,
                            median(samples.times)/1e6,quantile(samples.times,.95)/1e6,
                            Int(round(median(samples.memory))),length(samples.times)),','))
                    else
                        println(io,join((n,corpus,horizon,strategy,group,run.status,"","","",""),','))
                    end
                end
            end
            if !suite_passed(result)
                foreach(run->run.status==:pass || println(stderr,run.planned.feature.id,": ",run.status," ",run.message),result.runs)
                error("C++ dimension $n did not validate")
            end
            cm_append_cold(cold_file,cold_paths)
            for (case,cold) in zip(cases,cold_paths)
                corpus,horizon,strategy=case
                strategy==:cpp_packed || continue
                executable=joinpath(compiled,"garamon_native")
                raw_native=joinpath(temporary,"native-samples.csv")
                cm_guarded_process(`$executable $(cold*".bin") $raw_native`,temporary,CM_BUDGET)
                open(native_file,"a") do io
                    for line in readlines(joinpath(temporary,"command.log"))[2:end]
                        println(io,n,',',corpus,',',horizon,',',line)
                    end
                end
                open(native_samples_file,"a") do io
                    for line in readlines(raw_native)[2:end]
                        println(io,n,',',corpus,',',horizon,',',line)
                    end
                end
            end
            peak=cm_tree_bytes(temporary)
            peak<=CM_BUDGET || error("dimension disk budget exceeded")
            push!(build_rows,(dimension=n,stage=:before_cleanup,time_ms=0.0,peak_bytes=peak,retained_bytes=0,library_sha256=""))
            println("Dimension ",n," validated; deleting generated sources, objects and libraries (",peak," bytes)");flush(stdout)
        end
        fingerprint==cm_driver_source_hash() || error("Julia source changed during C++ comparison")
    end
    open(build_file,"w") do io
        println(io,join(keys(first(build_rows)),','))
        foreach(row->println(io,join(values(row),',')),build_rows)
    end
    open(joinpath(output,"cpp-matched-environment.txt"),"w") do io
        println(io,"Julia: ",VERSION,"\nPerfChecker: ",pkgversion(PerfChecker),
                "\nJulia source SHA256: ",fingerprint,"\nC++ revision: ",cpp_revision,
                "\nCPU: ",Sys.cpu_info()[1].model,"\nFeatures: ",features_total,
                "\nCase order: ",case_order,
                "\nJulia optimization level: ",Base.JLOptions().opt_level,
                "\nJulia CPU target: ",unsafe_string(Base.JLOptions().cpu_target),
                "\nDimension disk budget bytes: ",CM_BUDGET,
                "\nGenerated sources and binaries retained: none\nCompiler flags: -O2 -march=native -ffp-contract=off",
                "\nClang:\n",read(`clang++ --version`,String),"Eigen: ",readchomp(`pkg-config --modversion eigen3`),
                "\nDisk before:\n",disk_before,"Disk after:\n",read(`df -B1 $CM_CPP_ROOT`,String),
                "Repository du before:\n",du_before,"Repository du after:\n",read(`du -sb $CM_CPP_ROOT`,String))
        for path in (@__FILE__,joinpath(@__DIR__,"cpp_matched_common.jl"),
                     joinpath(CM_CPP_ROOT,"benchmark","CMakeLists.txt"),
                     joinpath(CM_CPP_ROOT,"benchmark","packed.cpp"),joinpath(CM_CPP_ROOT,"benchmark","upstream.cpp"),
                     joinpath(CM_CPP_ROOT,"benchmark","native.cpp"))
            println(io,path," SHA256: ",bytes2hex(sha256(read(path))))
        end
        for dir in (CM_JULIA_ROOT,joinpath(@__DIR__,"controller"),joinpath(@__DIR__,"runner")),
            file in ("Project.toml","Manifest.toml")
            path=joinpath(dir,file)
            println(io,path," SHA256: ",isfile(path) ? bytes2hex(sha256(read(path))) : "absent")
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    smoke="--smoke" in ARGS
    case_order="--reverse" in ARGS ? :reverse : :forward
    destinations=filter(arg->!(arg in ("--smoke","--reverse")),ARGS)
    length(destinations)==1 && !startswith(only(destinations),"--") ||
        error("usage: julia --project=perf/controller perf/cpp_matched.jl [--smoke] [--reverse] OUTPUT_DIRECTORY")
    cm_campaign(abspath(only(destinations));smoke,case_order)
end
