using Test, Garamon, CUDA, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("resident_sum.jl")
using .ResidentSumPrototype

@testset "Exact long-session CPU and GPU resident sums" begin
    @test Threads.nthreads()==4
    @test CUDA.functional()
    for n in (8,65),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (1,32,1024),
        repetitions in (1,4,16)
        fixture=compact_fixture(n,family,signature)
        cpu=build_resident_sum_cpu(fixture,horizon)
        gpu=build_resident_sum_gpu(fixture,horizon)
        serial=resident_sum_cpu(cpu,repetitions)
        parallel=resident_sum_cpu(cpu,repetitions;threaded=true)
        device=resident_sum_gpu(gpu,repetitions)
        fused=resident_sum_gpu_fused(gpu,repetitions)
        @test serial==parallel
        @test serial==device
        @test serial==fused
        @test device==fused
        @test serial!==parallel
        @test serial!==device
        @test serial!==fused
        masks=first(cpu).source.plan.output_masks
        oracles=[[rank_oracle(fixture,fixture.inputs[mod1(column+shift,4)]...,
            (:all,0)) for column in 1:4] for shift in 0:3]
        for column in 1:horizon,(row,mask) in enumerate(masks)
            expected=sum(get(oracles[mod1(step,4)][mod1(column,4)],mask,Int64(0))
                for step in 1:repetitions)
            @test serial[row,column]==expected
            @test device[row,column]==expected
            @test fused[row,column]==expected
        end
    end
    fixture=compact_fixture(8,:higher_rank,:positive)
    @test_throws ArgumentError build_resident_sum_cpu(fixture,32;max_bytes=1)
    @test_throws ArgumentError build_resident_sum_gpu(fixture,32;max_bytes=1)
    gpu=build_resident_sum_gpu(fixture,32)
    @test_throws ArgumentError resident_sum_gpu_fused(gpu,0)
end
