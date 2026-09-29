using Garamon, LinearAlgebra, Libdl, SHA

const CM_PATH = Tuple{Int,Int,Int,Float64}

function cm_masks(n, corpus)
    n in (3, 7) || error("bounded C++ corpus admits only dimensions 3 and 7")
    if corpus == :vectors
        masks = UInt64[1 << i for i in 0:n-1]
        return masks, copy(masks)
    elseif corpus == :mixed && n == 3
        return UInt64.(0:7), UInt64.(0:7)
    elseif corpus == :mixed
        return UInt64[0,1,8,64], UInt64[0,2,16,64]
    end
    error("unknown corpus")
end

cm_value(column, operand, slot) = Float64(1 + mod(column + operand + slot, 3))

function cm_inputs(n, corpus, horizon)
    ga = algebra(n, :ega)
    amasks, bmasks = cm_masks(n, corpus)
    left = [multivector(ga, Dict(mask => cm_value(t,1,i) for (i,mask) in enumerate(amasks));
                        storage=:sparse) for t in 1:horizon]
    right = [multivector(ga, Dict(mask => cm_value(t,2,i) for (i,mask) in enumerate(bmasks));
                         storage=:sparse) for t in 1:horizon]
    (; left, right, n, corpus, horizon)
end

cm_plan(inputs) = prepare_product(first(inputs.left), first(inputs.right); max_paths=256)
cm_pack(plan, inputs) = pack_product_batch(plan, inputs.left, inputs.right)
cm_precompile(batch) = precompile(run_packed_batch, (typeof(batch),))

# Independent integer oracle, including completeness of the requested outputs.
function cm_expected(inputs, batch)
    amasks, bmasks = cm_masks(inputs.n, inputs.corpus)
    masks = unique(UInt64[xor(a,b) for a in amasks for b in bmasks])
    Set(masks) == Set(batch.plan.output_masks) || error("plan output support is incomplete")
    positions = Dict(mask => i for (i,mask) in enumerate(batch.plan.output_masks))
    periodic = zeros(Int64, length(positions), 3)
    for t in 1:3, (ai,a) in enumerate(amasks), (bi,b) in enumerate(bmasks)
        inversions = 0
        for i in 1:inputs.n, j in 1:inputs.n
            inversions += (i > j && ((a>>(i-1))&1)==1 && ((b>>(j-1))&1)==1)
        end
        sign = isodd(inversions) ? -1 : 1
        periodic[positions[xor(a,b)],t] += sign * Int(cm_value(t,1,ai)) * Int(cm_value(t,2,bi))
    end
    Float64[periodic[i,1+mod(t-1,3)] for i in axes(periodic,1), t in 1:inputs.horizon]
end

mutable struct CmNative
    library::Ptr{Cvoid}
    context::Ptr{Cvoid}
    product::Ptr{Cvoid}
    destroy::Ptr{Cvoid}
end

function cm_native(library_path, batch)
    library = Libdl.dlopen(library_path)
    create = Libdl.dlsym(library, :garamon_native_prepare)
    destroy = Libdl.dlsym(library, :garamon_native_destroy)
    p = batch.plan
    context = ccall(create, Ptr{Cvoid},
        (Ptr{Float64},Ptr{Float64},Ptr{UInt64},Ptr{UInt64},Ptr{UInt64},Int,Int,Int,Int),
        batch.left_values, batch.right_values, p.left_masks, p.right_masks, p.output_masks,
        length(p.left_masks),length(p.right_masks),length(p.output_masks),size(batch.left_values,2))
    context == C_NULL && error("native Mvec input construction failed")
    value = CmNative(library, context, Libdl.dlsym(library,:garamon_native_product), destroy)
    finalizer(value) do object
        ccall(object.destroy, Cvoid, (Ptr{Cvoid},), object.context)
        Libdl.dlclose(object.library)
    end
    value
end

function cm_cpp_library(path)
    library = Libdl.dlopen(path)
    @assert sizeof(Int)==8 && sizeof(CM_PATH)==32
    @assert ccall(Libdl.dlsym(library,:garamon_path_size),Csize_t,()) == sizeof(CM_PATH)
    for field in 1:4
        @assert ccall(Libdl.dlsym(library,:garamon_path_offset),Csize_t,(Cint,),field-1)==fieldoffset(CM_PATH,field)
    end
    (; library, product=Libdl.dlsym(library,:garamon_packed_product))
end

function cm_validate_batch(batch)
    p = batch.plan
    isdiag(metric(p.algebra)) && diag(metric(p.algebra))==p.diagonal &&
        basis(p.algebra)==p.basis_names || error("packed plan algebra changed")
    nothing
end

function cm_cpp_packed(batch, library)
    cm_validate_batch(batch) # Same public checks as run_packed_batch.
    p = batch.plan
    n = size(batch.left_values,2)
    output = zeros(Float64,length(p.output_masks),n)
    ccall(library.product,Cvoid,
        (Ptr{CM_PATH},Int,Ptr{Float64},Ptr{Float64},Ptr{Float64},Int,Int,Int,Int),
        p.paths,length(p.paths),batch.left_values,batch.right_values,output,
        length(p.left_masks),length(p.right_masks),length(p.output_masks),n)
    output
end

function cm_cpp_upstream(batch, native)
    cm_validate_batch(batch)
    output = zeros(Float64,length(batch.plan.output_masks),size(batch.left_values,2))
    status = ccall(native.product,Cint,(Ptr{Cvoid},Ptr{Float64}),native.context,output)
    status==0 || error("upstream C++ product failed")
    output
end

function cm_precompile_frontend(batch, backend, strategy)
    target = strategy==:cpp_packed ? cm_cpp_packed : cm_cpp_upstream
    precompile(target,(typeof(batch),typeof(backend)))
end

function cm_workload(state)
    state.strategy in (:julia_jit,:julia_precompile) && return run_packed_batch(state.batch)
    state.strategy==:cpp_packed && return cm_cpp_packed(state.batch,state.backend)
    state.strategy==:cpp_upstream && return cm_cpp_upstream(state.batch,state.backend)
    error("unknown matched strategy")
end

function cm_observe(f,args...)
    @nospecialize f args
    @timed Base.invokelatest(f,args...)
end

function cm_rss(field="VmRSS")
    result=match(Regex("(?m)^"*field*":\\s+(\\d+)\\s+kB"),read("/proc/self/status",String))
    result===nothing ? -1 : 1024parse(Int,result.captures[1])
end

function cm_setup(n,corpus,horizon,strategy,packed_path,native_path,cold_path)
    BLAS.set_num_threads(1)
    cm_observe(identity,nothing)
    rows=NamedTuple[]
    function observe(stage,f,args...)
        rss_before=cm_rss()
        result=cm_observe(f,args...)
        push!(rows,(stage=stage,time_ms=1000result.time,compile_ms=1000result.compile_time,
                    allocated_bytes=result.bytes,rss_before_bytes=rss_before,
                    rss_after_bytes=cm_rss(),peak_rss_bytes=cm_rss("VmHWM")))
        result.value
    end
    inputs=observe(:inputs,cm_inputs,n,corpus,horizon)
    plan=observe(:plan,cm_plan,inputs)
    batch=observe(:packing,cm_pack,plan,inputs)
    backend=if strategy==:cpp_packed
        observe(:library_load,cm_cpp_library,packed_path)
    elseif strategy==:cpp_upstream
        observe(:native_prepare,cm_native,native_path,batch)
    elseif strategy==:julia_precompile
        observe(:explicit_precompile,cm_precompile,batch) || error("precompile did not succeed")
        nothing
    else
        nothing
    end
    state=(;batch,backend,strategy)
    if strategy in (:cpp_packed,:cpp_upstream)
        observe(:ffi_frontend_precompile,cm_precompile_frontend,batch,backend,strategy) ||
            error("C++ frontend precompile did not succeed")
    end
    result=if strategy in (:julia_jit,:julia_precompile)
        observe(:first_call,run_packed_batch,batch)
    elseif strategy==:cpp_packed
        observe(:first_call,cm_cpp_packed,batch,backend)
    else
        observe(:first_call,cm_cpp_upstream,batch,backend)
    end
    expected=cm_expected(inputs,batch)
    result==expected || error("first C++/Julia call differs from independent oracle")
    # Diagnostic repeat after first execution: structural preparation with the
    # plan-construction methods already compiled. Exclude from cold-path totals.
    observe(:plan_rebuild_warm,cm_plan,inputs)
    if !isfile(cold_path)
        open(cold_path,"w") do io
            println(io,"dimension,corpus,horizon,strategy,stage,time_ms,compile_ms,allocated_bytes,rss_before_bytes,rss_after_bytes,peak_rss_bytes")
            for row in rows
                println(io,join((n,corpus,horizon,strategy,row.stage,row.time_ms,row.compile_ms,
                                row.allocated_bytes,row.rss_before_bytes,row.rss_after_bytes,row.peak_rss_bytes),','))
            end
        end
    end
    if strategy==:cpp_packed && !isfile(cold_path*".bin")
        open(cold_path*".bin","w") do io
            write(io,Int64[length(plan.paths),length(plan.left_masks),length(plan.right_masks),
                           length(plan.output_masks),horizon])
            write(io,plan.paths)
            write(io,batch.left_values)
            write(io,batch.right_values)
            write(io,expected)
        end
    end
    (;state...,expected)
end

cm_oracle(state) = cm_workload(state)==state.expected
