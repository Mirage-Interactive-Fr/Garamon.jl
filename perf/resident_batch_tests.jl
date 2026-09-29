using Test, Garamon, CUDA, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("cpu_threaded_packed.jl")
include("gpu_packed.jl")
using .CPUThreadedPackedPrototype, .GPUPackedPrototype

function resident_test_batch(fixture,horizon)
    a,b=first(fixture.inputs)
    plan=prepare_product(a,b)
    left=[fixture.inputs[mod1(i,4)][1] for i in 1:horizon]
    right=[fixture.inputs[mod1(i,4)][2] for i in 1:horizon]
    pack_product_batch(plan,left,right)
end

@testset "Exact reusable CPU and GPU packed residents" begin
    @test Threads.nthreads()==4
    @test CUDA.functional()
    for n in (8,65),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (1,32,1024)
        fixture=compact_fixture(n,family,signature)
        batch=resident_test_batch(fixture,horizon)
        expected=run_packed_batch(batch)
        oracle=[rank_oracle(fixture,pair...,(:all,0)) for pair in fixture.inputs]
        cpu=cpu_resident_batch(batch)
        gpu=gpu_resident_batch(batch)
        @test cpu_run_serial!(cpu)===cpu.output
        @test cpu.output==expected
        @test cpu_run_threaded!(cpu)===cpu.output
        @test cpu.output==expected
        gpu_enqueue!(gpu)
        gpu_enqueue!(gpu)
        CUDA.synchronize()
        host=Array(gpu.output)
        @test host==expected
        for column in 1:horizon,(row,mask) in enumerate(batch.plan.output_masks)
            @test cpu.output[row,column]==get(oracle[mod1(column,4)],mask,Int64(0))
            @test host[row,column]==get(oracle[mod1(column,4)],mask,Int64(0))
        end
    end
    fixture=compact_fixture(8,:higher_rank,:positive)
    batch=resident_test_batch(fixture,32)
    @test_throws ArgumentError cpu_resident_batch(batch;max_bytes=0)
    @test_throws ArgumentError cpu_resident_batch(batch;max_bytes=1)
end
