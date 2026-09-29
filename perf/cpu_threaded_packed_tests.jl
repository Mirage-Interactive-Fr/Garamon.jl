using Test, Garamon, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("cpu_threaded_packed.jl")
using .CPUThreadedPackedPrototype

function threaded_fixture_batch(fixture,horizon)
    a,b=first(fixture.inputs)
    plan=prepare_product(a,b)
    left=[fixture.inputs[mod1(i,4)][1] for i in 1:horizon]
    right=[fixture.inputs[mod1(i,4)][2] for i in 1:horizon]
    pack_product_batch(plan,left,right)
end

@testset "Exact threaded packed CPU columns" begin
    @test Threads.nthreads()==4
    for n in (2,8,65,128),family in (:low_rank_high_grade,:higher_rank),
        signature in (:positive,:mixed,:degenerate),horizon in (1,32,1024)
        fixture=compact_fixture(n,family,signature)
        batch=threaded_fixture_batch(fixture,horizon)
        expected=[rank_oracle(fixture,pair...,(:all,0)) for pair in fixture.inputs]
        reference=run_packed_batch(batch)
        threaded=run_packed_batch_threaded(batch)
        again=run_packed_batch_threaded(batch)
        @test threaded==reference
        @test again==reference
        @test threaded!==again
        @test pointer(threaded)!=pointer(again)
        @test size(threaded)==(length(batch.plan.output_masks),horizon)
        for column in 1:horizon,(row,mask) in enumerate(batch.plan.output_masks)
            @test threaded[row,column]==get(expected[mod1(column,4)],mask,Int64(0))
        end
    end
end
