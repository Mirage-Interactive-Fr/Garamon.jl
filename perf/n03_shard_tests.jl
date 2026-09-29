using Test
include("n03_shards.jl")
include("n03_radical.jl")
@testset "N03 screen and exhaustive bounded partition" begin
    grid=n03_grid();screen=n03_cases(:screen)
    @test length(grid)==18000
    @test getproperty.(grid,:case_id)==collect(1:18000)
    @test length(screen)==756
    @test sort!(unique(getproperty.(screen,:n)))==collect(N03_SCREEN_DIMS)
    @test Set(getproperty.(screen,:r))==Set(1:3)
    @test Set(getproperty.(screen,:horizon))==Set((1,8,32))
    @test all(c->c.n>=c.r+c.s,screen)
    @test length(n03_cases(:smoke))==6
    @test n03_select(:screen;reverse_order=true)==reverse(screen)
    partitions=[getproperty.(n03_select(:full;shard=(i,563)),:case_id) for i in 1:563]
    @test maximum(length,partitions)==32
    @test minimum(length,partitions)==31
    @test sort!(vcat(partitions...))==collect(1:18000)
    @test length(unique(vcat(partitions...)))==18000
    @test getproperty.(n03_select(:full;interval=(17990,18000),reverse_order=true),:case_id)==collect(18000:-1:17990)
    @test length(n03_select(:full;prepare_only=true))==18000
    @test_throws ErrorException n03_select(:full)
    @test_throws ErrorException n03_select(:screen;shard=(1,563))
    @test_throws ErrorException n03_select(:smoke;interval=(1,2))
    for kwargs in ((;shard=(0,563)),(;shard=(564,563)),(;shard=(1,562)),(;shard=(1,18001)),
                   (;shard=(true,563)),(;interval=(0,1)),(;interval=(1,33)),(;interval=(2,1)),
                   (;interval=(18000,18001)),(;shard=(1,563),interval=(1,2)))
        @test_throws ErrorException n03_select(:full;kwargs...)
    end
    @test n03_grid_sha256()==n03_grid_sha256(copy(grid))
    @test n03_grid_sha256()!=n03_grid_sha256(reverse(grid))
    # In 1 nonradical coordinate, signed must genuinely differ from positive.
    for r in 1:3
        positive=n03_input(r+1,r,1,:positive,1)
        signed=n03_input(r+1,r,1,:signed,1)
        @test positive.g!=signed.g
        fixture=n03_fixture(r+1,r,1,:signed,1)
        @test n03_qualify(fixture,n03_episode(fixture,:radical))
        @test n03_qualify(fixture,n03_episode(fixture,:regular))
    end
    for n in (64,65,128),r in 1:3
        input=n03_input(n,r,1,:signed,1)
        @test count(==(-1.0),input.g)==1
        fixture=n03_fixture(n,r,1,:signed,1)
        @test n03_qualify(fixture,n03_episode(fixture,:radical))
        @test n03_qualify(fixture,n03_episode(fixture,:regular))
    end
end
@testset "N03 coverage audit rejects absent, duplicate, mixed, incomplete evidence" begin
    mktempdir() do root
        @test !n03_audit(root;mode=:smoke)["complete"]
        dir=joinpath(root,"one");mkpath(dir);ids=getproperty.(n03_cases(:smoke),:case_id)
        m=Dict("selected_case_ids"=>ids,"completed_case_ids"=>ids,"source_unchanged"=>true,
            "all_cases_passed"=>true,"grid_sha256"=>n03_grid_sha256(),"source_sha256"=>"synthetic-test",
            "condition"=>"isolated","interference_label"=>"","reverse_order"=>false)
        for name in ("n03-qualification.json","n03-samples.csv","n03-selection.csv")
            write(joinpath(dir,name),"synthetic structural audit fixture; no scientific qualification")
        end
        path=joinpath(dir,"n03-environment.toml")
        open(io->TOML.print(io,m),path,"w")
        @test n03_audit(root;mode=:smoke)["complete"]
        cp(dir,joinpath(root,"duplicate"))
        @test !n03_audit(root;mode=:smoke)["complete"]
        rm(joinpath(root,"duplicate");recursive=true)
        m["completed_case_ids"]=ids[1:end-1];open(io->TOML.print(io,m),path,"w")
        @test !n03_audit(root;mode=:smoke)["complete"]
        m["completed_case_ids"]=ids;m["all_cases_passed"]=false;open(io->TOML.print(io,m),path,"w")
        @test !n03_audit(root;mode=:smoke)["complete"]
        m["all_cases_passed"]=true;open(io->TOML.print(io,m),path,"w")
        rm(joinpath(dir,"n03-samples.csv"))
        @test !n03_audit(root;mode=:smoke)["complete"]
    end
end
