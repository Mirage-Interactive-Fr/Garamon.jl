using Test, CUDA, Garamon, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("gpu_packed.jl")
using .GPUPackedPrototype

@testset "CUDA packed product prerequisites" begin
    @test VERSION.major==1 && VERSION.minor==13
    @test CUDA.functional()
    @test Threads.nthreads()==1
end

@testset "CUDA packed product preserves independent integer oracle" begin
    for n in (2,8,65,128), family in (:low_rank_high_grade,:higher_rank),
        signature in RANK_SIGNATURES, horizon in (1,32,1024)
        fixture=compact_fixture(n,family,signature)
        a,b=first(fixture.inputs)
        plan=prepare_product(a,b)
        left=[fixture.inputs[mod1(i,4)][1] for i in 1:horizon]
        right=[fixture.inputs[mod1(i,4)][2] for i in 1:horizon]
        packed=pack_product_batch(plan,left,right)
        cpu=run_packed_batch(packed)
        gpu=gpu_complete_matrix(packed)
        @test size(gpu)==size(cpu)
        @test gpu==cpu
        expected=[rank_oracle(fixture,pair...,(:all,0)) for pair in fixture.inputs]
        for j in 1:horizon, (oi,mask) in enumerate(plan.output_masks)
            @test gpu[oi,j]==get(expected[mod1(j,4)],mask,Int64(0))
        end
        resident=gpu_resident_batch(packed)
        owned=gpu_owned_matrix(resident)
        @test owned==cpu
        @test objectid(owned)!=objectid(gpu)
        owned[1,1]+=1
        @test gpu_owned_matrix(resident)==cpu
        @test_throws ArgumentError gpu_resident_batch(packed;max_bytes=1)
    end
end
