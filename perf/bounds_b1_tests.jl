using Test,Garamon,LinearAlgebra
include("bounds_b1.jl")
using .BoundsB1Prototype

# Independent ordered-word Int64 oracle for diagonal geometric products.
# It neither consumes the prepared paths nor calls Garamon's product/sign code.
function b1_oracle(a,b,diagonal)
    K=keytype(a.values);result=Dict{K,Int64}()
    for (am,av) in a.values,(bm,bv) in b.values
        word=[i for i in eachindex(diagonal) if !iszero(am & (one(K)<<(i-1)))]
        value=Int64(av)*Int64(bv)
        for j in eachindex(diagonal)
            iszero(bm & (one(K)<<(j-1))) && continue
            position=length(word)+1
            while position>1 && word[position-1]>j
                value=-value;position-=1
            end
            if position>1 && word[position-1]==j
                value*=diagonal[j];deleteat!(word,position-1)
            else
                insert!(word,position,j)
            end
        end
        mask=foldl(|,(one(K)<<(i-1) for i in word);init=zero(K))
        result[mask]=get(result,mask,0)+value
    end
    filter!(pair->!iszero(last(pair)),result)
end

@testset "B1 execution environment" begin
    @test VERSION.major==1 && VERSION.minor==13
    @test Threads.nthreads()==1
    @test Base.JLOptions().check_bounds in (0,1) # auto or yes; never global no.
    println("B1 Julia=",VERSION," check_bounds=",Base.JLOptions().check_bounds," threads=",Threads.nthreads())
end

@testset "B1 exactness, metrics, changing values and owned outputs" begin
    for n in (2,8,65,128),signature in (:positive,:mixed,:degenerate),T in (Int64,Float64)
        diagonal=ones(Int64,n)
        signature==:mixed && (diagonal[2:2:n].=-1)
        signature==:degenerate && (diagonal[1]=0)
        ga=algebra(Diagonal(T.(diagonal)));K=n<=64 ? UInt64 : UInt128
        full=K((big(1)<<n)-1);masks=unique(K[0,1,full,full ⊻ K(1)])
        inputs=[(multivector(ga,Dict(m=>T((-2,-1,1,2)[mod1(i+p,4)]) for (i,m) in enumerate(masks));storage=:sparse),
            multivector(ga,Dict(m=>T((-2,-1,1,2)[mod1(2i+p,4)]) for (i,m) in enumerate(masks));storage=:sparse)) for p in 1:4]
        plan=prepare_product(inputs[1]...)
        for variant in (:checked,:inbounds)
            outputs=[b1_product(plan,a,b;variant) for (a,b) in inputs]
            @test length(unique(objectid(x.values) for x in outputs))==4
            for (result,(a,b)) in zip(outputs,inputs)
                @test result.values==b1_oracle(a,b,diagonal)
                @test result.values==run_product(plan,a,b).values
            end
            saved=copy(outputs[2].values);empty!(outputs[1].values)
            @test outputs[2].values==saved
        end
    end
end

@testset "B1 rejects altered structures before the unchecked loop" begin
    ga=algebra(Diagonal(ones(3)))
    a=multivector(ga,Dict(UInt64(i)=>Float64(i+1) for i in 0:3);storage=:sparse)
    b=multivector(ga,Dict(UInt64(i)=>Float64(4-i) for i in 0:3);storage=:sparse)
    for variant in (:checked,:inbounds)
        for field in 1:3,index in (typemin(Int),-1,0,5,typemax(Int))
            plan=prepare_product(a,b)
            path=plan.paths[1];plan.paths[1]=ntuple(i->i==field ? index : path[i],4)
            @test_throws ArgumentError b1_product(plan,a,b;variant)
        end
        for mutate in (p->pop!(p.left_masks),p->empty!(p.right_masks),p->empty!(p.output_masks),
                       p->push!(p.left_masks,first(p.left_masks)),p->push!(p.output_masks,first(p.output_masks)),
                       p->empty!(p.left_support),p->(p.output_masks[1]=UInt64(8)),
                       p->reverse!(p.right_masks),p->(p.paths[1]=(1,1,2,p.paths[1][4])),
                       p->(p.paths[1]=(1,1,1,NaN)))
            plan=prepare_product(a,b);mutate(plan)
            @test_throws ArgumentError b1_product(plan,a,b;variant)
        end
        dense_a=dense(a);dense_b=dense(b);plan=prepare_product(dense_a,dense_b)
        resize!(dense_a.values,7) # Only a trailing zero removed: support still matches.
        @test_throws DimensionMismatch b1_product(plan,dense_a,dense_b;variant)
        plan=prepare_product(a,b)
        subset=multivector(ga,Dict(UInt64(0)=>1.0);storage=:sparse)
        @test_throws ArgumentError b1_product(plan,subset,b;variant)
        @test b1_product(plan,subset,b;variant,allow_subsets=true).values==b1_oracle(subset,b,ones(Int64,3))
        zero=multivector(ga,Dict{UInt64,Float64}();storage=:sparse)
        @test isempty(b1_product(plan,zero,b;variant,allow_subsets=true).values)
        zero_plan=prepare_product(zero,b)
        @test isempty(b1_product(zero_plan,zero,b;variant).values)
        scalar_plan=prepare_product(subset,subset)
        @test length(scalar_plan.paths)<length(scalar_plan.left_masks)+length(scalar_plan.right_masks)
        @test b1_product(scalar_plan,subset,subset;variant).values==Dict(UInt64(0)=>1.0)
        @test b1_product(plan,a,b;variant).values==b1_oracle(a,b,ones(Int64,3))
    end
    @test_throws ArgumentError b1_product(prepare_product(a,b),a,b;variant=:unknown)
end

@testset "B1W canonical snapshot, long reuse and owned exact outputs" begin
    for n in (2,8,65,128),signature in (:positive,:mixed,:degenerate),T in (Int64,Float64)
        diagonal=ones(Int64,n)
        signature==:mixed && (diagonal[2:2:n].=-1)
        signature==:degenerate && (diagonal[1]=0)
        ga=algebra(Diagonal(T.(diagonal)));K=n<=64 ? UInt64 : UInt128
        full=K((big(1)<<n)-1);masks=unique(K[0,1,full,full ⊻ K(1)])
        inputs=[(multivector(ga,Dict(m=>T((-2,-1,1,2)[mod1(i+p,4)]) for (i,m) in enumerate(masks));storage=:sparse),
            multivector(ga,Dict(m=>T((-2,-1,1,2)[mod1(2i+p,4)]) for (i,m) in enumerate(masks));storage=:sparse)) for p in 1:4]
        plan=prepare_product(inputs[1]...)
        ws=b1_workspace(plan,inputs[1]...)
        direct_ws=b1_workspace(inputs[1]...)
        @test ws.plan !== plan
        outputs=[b1_workspace_product!(ws,a,b) for (a,b) in inputs]
        direct_outputs=[b1_workspace_product!(direct_ws,a,b) for (a,b) in inputs]
        singlepass_outputs=[b1_workspace_product_singlepass!(direct_ws,a,b) for (a,b) in inputs]
        @test length(unique(objectid(x.values) for x in singlepass_outputs))==4
        @test all(singlepass_outputs[i].values==direct_outputs[i].values for i in eachindex(inputs))
        @test length(unique(objectid(x.values) for x in outputs))==4
        @test length(unique(objectid(x.values) for x in direct_outputs))==4
        for (result,(a,b)) in zip(outputs,inputs)
            @test result.values==b1_oracle(a,b,diagonal)
            @test result.values==run_product(plan,a,b).values
        end
        for (result,(a,b)) in zip(direct_outputs,inputs)
            @test result.values==b1_oracle(a,b,diagonal)
        end
        saved=copy(outputs[2].values);empty!(outputs[1].values)
        @test outputs[2].values==saved
        direct_saved=copy(direct_outputs[2].values);empty!(direct_outputs[1].values)
        @test direct_outputs[2].values==direct_saved
        empty!(plan.paths);empty!(plan.left_masks)
        @test b1_workspace_product!(ws,inputs[1]...).values==b1_oracle(inputs[1]...,diagonal)
        @test b1_workspace_product!(direct_ws,inputs[1]...).values==b1_oracle(inputs[1]...,diagonal)
    end
end

@testset "B1W rejects changed contracts and false source plans" begin
    ga=algebra(Diagonal(ones(3)))
    a=multivector(ga,Dict(UInt64(i)=>Float64(i+1) for i in 0:3);storage=:sparse)
    b=multivector(ga,Dict(UInt64(i)=>Float64(4-i) for i in 0:3);storage=:sparse)
    plan=prepare_product(a,b)
    @test_throws ArgumentError b1_workspace(plan,a,b;max_bytes=1)
    @test_throws ArgumentError b1_workspace(a,b;max_bytes=1)
    for alter in (p->pop!(p.paths),p->(p.paths[1]=(1,1,1,p.paths[1][4]+1)),
                  p->reverse!(p.output_masks))
        bad=prepare_product(a,b);alter(bad)
        @test_throws ArgumentError b1_workspace(bad,a,b)
    end
    ws=b1_workspace(plan,a,b)
    subset=multivector(ga,Dict(UInt64(0)=>1.0);storage=:sparse)
    @test_throws ArgumentError b1_workspace_product!(ws,subset,b)
    dense_a=dense(a);dense_b=dense(b);dense_ws=b1_workspace(prepare_product(dense_a,dense_b),dense_a,dense_b)
    resize!(dense_a.values,7)
    @test_throws DimensionMismatch b1_workspace_product!(dense_ws,dense_a,dense_b)
    @test_throws DimensionMismatch b1_workspace(dense_a,dense_b)
    metric(ga)[1,1]=2.0
    @test_throws ArgumentError b1_workspace_product!(ws,a,b)
    ga2=algebra(Diagonal(ones(3)))
    a2=multivector(ga2,Dict(UInt64(0)=>1.0);storage=:sparse)
    b2=multivector(ga2,Dict(UInt64(0)=>2.0);storage=:sparse)
    ws2=b1_workspace(prepare_product(a2,b2),a2,b2)
    basis(ga2)[1]="changed"
    @test_throws ArgumentError b1_workspace_product!(ws2,a2,b2)
    ga3=algebra(Diagonal(ones(Int64,3)))
    ia=multivector(ga3,Dict(UInt64(0)=>Int64(2));storage=:sparse)
    ib=multivector(ga3,Dict(UInt64(0)=>Int64(3));storage=:sparse)
    iws=b1_workspace(prepare_product(ia,ib),ia,ib)
    fa=multivector(ga3,Dict(UInt64(0)=>2.0);storage=:sparse)
    @test_throws ArgumentError b1_workspace_product!(iws,fa,ib)
    ga4=algebra(Diagonal(ones(3)))
    sa=multivector(ga4,Dict(UInt64(0)=>1.0);storage=:sparse)
    sb=multivector(ga4,Dict(UInt64(0)=>2.0);storage=:sparse)
    sa.values[UInt64(8)]=3.0
    @test_throws ArgumentError b1_workspace(sa,sb)
end
