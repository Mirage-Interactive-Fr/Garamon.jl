# Controller checks; one targeted @timed worker setup regression, no PerfChecker run_suite.
using Test
include("n04_short_identity_compare.jl")

@testset "N04 controller conditions and official runtime" begin
    @test n04_runtime_check()
    @test n04_runtime_check(version=v"1.13.1",threads=1)
    @test_throws ArgumentError n04_runtime_check(version=v"1.12.0",threads=1)
    @test_throws ArgumentError n04_runtime_check(version=v"1.14.0",threads=1)
    @test_throws ArgumentError n04_runtime_check(version=v"1.13.0",threads=2)
    @test_throws ArgumentError n04_condition(Dict{String,String}())
    @test_throws ArgumentError n04_condition(Dict("GARAMONBENCH_CONDITION"=>"unknown"))
    @test_throws ArgumentError n04_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference"))
    @test_throws ArgumentError n04_condition(Dict("GARAMONBENCH_CONDITION"=>"isolated","GARAMONBENCH_INTERFERENCE_LABEL"=>"Etendue3D"))
    @test_throws ArgumentError n04_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference","GARAMONBENCH_INTERFERENCE_LABEL"=>"a\nb"))
    @test n04_condition(Dict("GARAMONBENCH_CONDITION"=>"isolated")).label==""
    condition=n04_condition(Dict("GARAMONBENCH_CONDITION"=>"exploratory_interference","GARAMONBENCH_INTERFERENCE_LABEL"=>"Etendue3D"))
    @test condition.label=="Etendue3D" && !condition.automatically_verified_isolation
end

@testset "N04 multi-term worker diagnostics" begin
    record=n04_grid()[6551]
    mktempdir() do root
        trace=joinpath(root,"trace.toml")
        state=n04_worker_setup(record,trace,n04_source_hash())
        @test any(length(a)>1 for a in state.exact)
        details=TOML.parsefile(trace)
        @test details["oracle_exact"]
        @test length(details["exact_phase_values"])==length(state.exact)
        @test all(issorted(row) for row in details["exact_phase_values"])
    end
end

@testset "N04 stable selection and bounded resources" begin
    records=n04_grid();smoke=n04_select(records;smoke=true)
    @test length(records)==30720
    @test length(unique(r.n for r in records))==12
    @test length(smoke)==20 && length(unique(r.case_id for r in smoke))==20
    @test all(r->records[r.case_id]==r,smoke)
    @test length(unique((r.n,r.p,r.H,r.family,r.shape,r.changes) for r in smoke))==4
    @test all(count(r->r.strategy==s,smoke)==4 for s in N04_ROUTES)
    @test n04_select(records;smoke=true,reverse_order=true)==reverse(smoke)
    @test getproperty.(n04_select(records;case_ids=[30720,1]),:case_id)==[30720,1]
    @test isempty(n04_select(records))
    @test_throws ArgumentError n04_select(records;smoke=true,case_ids=[1])
    @test_throws ArgumentError n04_select(records;case_ids=[1,1])
    @test_throws ArgumentError n04_select(records;case_ids=[0])
    @test_throws ArgumentError n04_select(records;case_ids=[30721])
    @test_throws ArgumentError n04_select(records;case_ids=collect(1:21))
    @test n04_grid_hash(records)==n04_grid_hash()
    @test n04_grid_hash(reverse(records))!=n04_grid_hash(records)
    @test n04_before_next_case(239.9)
    @test !n04_before_next_case(240)
    @test_throws ArgumentError n04_before_next_case(-1)
    mktempdir() do root
        write(joinpath(root,"a"),"four")
        @test n04_disk_check(root;max_bytes=4)==4
        @test_throws ErrorException n04_disk_check(root;max_bytes=3)
    end
end

@testset "N04 exact controller oracle and output ownership" begin
    state=n04_fixture(2,1,32,:signed,:vector,:stable)
    output=n04_episode(state,:binary)
    @test n04_output_check(state,output)
    @test n04_resource_check(state,output;rss=0)
    @test !n04_output_check(state,output[1:31])
    corrupted=deepcopy(output);corrupted[1][big(0)]=1
    @test !n04_output_check(state,corrupted)
    @test !n04_output_check(state,fill(first(output),32))
    @test !n04_output_check(state,[Dict(k=>Float64(v) for (k,v) in a) for a in output])
    @test !n04_output_check(state,[state.inputs[1+mod(t-1,3)] for t in 1:32])
    @test_throws ErrorException n04_resource_check(state,output;rss=(2<<30)+1)
    @test_throws ErrorException n04_resource_check(state,output;rss=0,max_fixture=0)
    @test_throws ErrorException n04_resource_check(state,output;rss=0,max_output=0)
end

@testset "N04 provenance and prepare-only cannot measure" begin
    @test !isdefined(@__MODULE__,:run_suite)
    @test all(isfile(joinpath(dirname(@__DIR__),f)) for f in n04_source_files())
    @test length(n04_source_hash())==64
    mktempdir() do root
        write(joinpath(root,"a"),"first");write(joinpath(root,"b"),"second")
        old=n04_files_hash(root,["a","b"])
        @test old==n04_files_hash(root,["b","a"])
        write(joinpath(root,"b"),"changed")
        @test old!=n04_files_hash(root,["a","b"])
    end
    withenv("GARAMONBENCH_CONDITION"=>"isolated","GARAMONBENCH_INTERFERENCE_LABEL"=>"") do
        mktempdir() do root
            @test_throws ArgumentError n04_main([root])
            @test_throws ArgumentError n04_main([root,"--unexpected"])
            @test_throws ArgumentError n04_main([root,"--case-ids=1","--case-ids=2","--prepare-only"])
            @test_throws ArgumentError n04_main([root,"--smoke","--case-ids=1","--prepare-only"])
            metadata=n04_main([root,"--smoke","--prepare-only"])
            @test metadata["status"]=="prepared"
            @test metadata["grid_cases"]==30720 && length(metadata["case_ids"])==20
            @test metadata["measurement_condition"]=="isolated" && metadata["interference_label"]==""
            @test metadata["source_sha256"]==n04_source_hash()
            @test length(readlines(joinpath(root,"n04-replay-matrix.csv")))==30721
            @test length(readlines(joinpath(root,"n04-selection.csv")))==21
            @test !isfile(joinpath(root,"n04-samples.csv"))
            @test !isfile(joinpath(root,"n04-sources.tar.gz"))
            @test metadata==n04_main([root,"--smoke","--prepare-only"])
            @test_throws ErrorException n04_campaign(root;case_ids=[1],prepare_only=true)
            withenv("GARAMONBENCH_CONDITION"=>"exploratory_interference","GARAMONBENCH_INTERFERENCE_LABEL"=>"Etendue3D") do
                @test_throws ErrorException n04_campaign(root;smoke=true,prepare_only=true)
            end
            write(joinpath(root,"n04-samples.csv"),"existing evidence")
            @test_throws ErrorException n04_campaign(root;smoke=true,prepare_only=true)
            @test read(joinpath(root,"n04-samples.csv"),String)=="existing evidence"
        end
    end
    @test !isdefined(@__MODULE__,:run_suite)
end
