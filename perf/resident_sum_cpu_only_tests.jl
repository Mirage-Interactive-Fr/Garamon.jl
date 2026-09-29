using Test, Garamon, CUDA, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("resident_sum.jl")
using .ResidentSumPrototype

@testset "Exact resident sums when CUDA has no visible device" begin
    @test VERSION.major==1 && VERSION.minor==13
    @test Threads.nthreads()==4
    @test !CUDA.functional()
    for n in (8,65),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (1,32,1024),
        repetitions in (1,16)
        fixture=compact_fixture(n,family,signature)
        cpu=build_resident_sum_cpu(fixture,horizon)
        serial=resident_sum_cpu(cpu,repetitions)
        parallel=resident_sum_cpu(cpu,repetitions;threaded=true)
        @test serial==parallel
        @test serial!==parallel
        masks=first(cpu).source.plan.output_masks
        oracles=[[rank_oracle(fixture,fixture.inputs[mod1(column+shift,4)]...,
            (:all,0)) for column in 1:4] for shift in 0:3]
        for column in 1:horizon,(row,mask) in enumerate(masks)
            expected=sum(get(oracles[mod1(step,4)][mod1(column,4)],mask,Int64(0))
                for step in 1:repetitions)
            @test serial[row,column]==expected
            @test parallel[row,column]==expected
        end
        @test_throws ArgumentError build_resident_sum_gpu(fixture,horizon)
    end
end

@testset "Higher dimensions retain the CPU route without CUDA" begin
    @test !CUDA.functional()
    for n in (129,192,256,384,512,768,1024,2048,4096,8192)
        fixture=compact_fixture(n,:higher_rank,:positive)
        @test metric(first(fixture.inputs)[1].algebra) isa Diagonal
        cpu=build_resident_sum_cpu(fixture,32)
        serial=resident_sum_cpu(cpu,8)
        parallel=resident_sum_cpu(cpu,8;threaded=true)
        @test serial!==parallel
        masks=first(cpu).source.plan.output_masks
        oracles=[[rank_oracle(fixture,fixture.inputs[mod1(column+shift,4)]...,
            (:all,0)) for column in 1:4] for shift in 0:3]
        for column in 1:32,(row,mask) in enumerate(masks)
            expected=sum(get(oracles[mod1(step,4)][mod1(column,4)],mask,Int64(0))
                for step in 1:8)
            @test serial[row,column]==expected
            @test parallel[row,column]==expected
        end
        @test_throws ArgumentError build_resident_sum_gpu(fixture,32)
        @test Sys.maxrss()<=6<<30
    end
end
