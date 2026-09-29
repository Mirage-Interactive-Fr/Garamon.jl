"""Return basis vectors as values, in the order of `basis(ga)` names."""
basisvectors(ga::GeometricAlgebra; storage::Symbol=:auto) =
    [basisvector(ga, i; storage) for i in 1:dimension(ga)]

"""
    @algebra begin
        scalar = Float64
        basis = (:e₁, :e₂)
        metric = [1 0; 0 1]
    end

Create a descriptor with checked basis names. `scalar` and `basis` are optional;
the `metric` assignment is required. This macro does not bind basis names in the
caller's scope; use `basisvectors(ga)` to obtain their values.
"""
macro algebra(block)
    block isa Expr && block.head == :block ||
        throw(ArgumentError("@algebra expects a begin/end block"))
    fields = Dict{Symbol,Any}()
    for item in block.args
        item isa LineNumberNode && continue
        item isa Expr && item.head == :(=) && item.args[1] isa Symbol ||
            throw(ArgumentError("@algebra accepts only field = value assignments"))
        field = item.args[1]
        field in (:metric, :basis, :scalar, :kind) ||
            throw(ArgumentError("unsupported @algebra field: $field"))
        haskey(fields, field) && throw(ArgumentError("duplicate @algebra field: $field"))
        fields[field] = item.args[2]
    end
    haskey(fields, :metric) || throw(ArgumentError("@algebra requires metric"))
    matrix_expr = esc(fields[:metric])
    if haskey(fields, :scalar)
        matrix_expr = :(Matrix{$(esc(fields[:scalar]))}($matrix_expr))
    end
    names_expr = haskey(fields, :basis) ?
        :(String.(collect($(esc(fields[:basis]))))) : :(String[])
    kind_expr = haskey(fields, :kind) ? esc(fields[:kind]) : QuoteNode(:none)
    return :($(GlobalRef(Garamon, :algebra))($matrix_expr, $kind_expr;
               basis=$names_expr))
end

"""A small pure expression tree; source lines are kept for diagnostics."""
struct GAExpr
    operation::Symbol
    args::Tuple
    line::Int
end

function _capture_ga(ex, line::Int)
    if ex isa Symbol || ex isa Number
        return :($(GlobalRef(Garamon, :GAExpr))(:leaf, ($(esc(ex)),), $line))
    end
    ex isa Expr && ex.head == :call ||
        throw(ArgumentError("@ga supports values and calls to +, -, *, ∧, wedge, and contractions"))
    op = ex.args[1]
    op in (:+, :-, :*, :∧, :wedge, :left_contraction, :right_contraction) ||
        throw(ArgumentError("unsupported @ga operation: $op"))
    length(ex.args) >= 2 || throw(ArgumentError("@ga operation has unsupported arity"))
    if length(ex.args) > 3 && op ∉ (:+, :*, :∧, :wedge)
        throw(ArgumentError("@ga operation has unsupported arity"))
    end
    children = [_capture_ga(arg, line) for arg in ex.args[2:end]]
    if length(children) == 1
        return :($(GlobalRef(Garamon, :GAExpr))($(QuoteNode(op)),
                   ($(children[1]),), $line))
    end
    left = :($(GlobalRef(Garamon, :GAExpr))($(QuoteNode(op)),
              ($(children[1]), $(children[2])), $line))
    for child in children[3:end]
        left = :($(GlobalRef(Garamon, :GAExpr))($(QuoteNode(op)),
                  ($left, $child), $line))
    end
    return left
end

"""Capture a bounded arithmetic expression for repeated evaluation or one output."""
macro ga(ex)
    try
        return _capture_ga(ex, __source__.line)
    catch error
        error isa ArgumentError || rethrow()
        throw(ArgumentError("$(error.msg) at $(__source__.file):$(__source__.line)"))
    end
end

function _eval_ga(node::GAExpr)
    node.operation == :leaf && return node.args[1]
    args = map(_eval_ga, node.args)
    op = node.operation
    op == :+ && return +(args...)
    op == :- && return -(args...)
    op == :* && return *(args...)
    op in (:∧, :wedge) && return wedge(args...)
    op == :left_contraction && return left_contraction(args...)
    op == :right_contraction && return right_contraction(args...)
    error("invalid GA expression node at line $(node.line)")
end

function _eval_coefficient(node::GAExpr, indices)
    op = node.operation
    if op == :+ && length(node.args) == 2
        return _eval_coefficient(node.args[1], indices) +
               _eval_coefficient(node.args[2], indices)
    elseif op == :- && length(node.args) == 2
        return _eval_coefficient(node.args[1], indices) -
               _eval_coefficient(node.args[2], indices)
    elseif op in (:*, :∧, :wedge, :left_contraction, :right_contraction) &&
           length(node.args) == 2 && all(child -> child.operation == :leaf, node.args)
        left, right = node.args[1].args[1], node.args[2].args[1]
        if left isa AbstractMultiVector && right isa AbstractMultiVector
            operation = op == :* ? :geometric :
                        op in (:∧, :wedge) ? :wedge :
                        op == :left_contraction ? :left : :right
            return product_coefficient(left, right, indices; operation)
        end
    end
    value = _eval_ga(node)
    value isa AbstractMultiVector ||
        throw(ArgumentError("requested coefficient requires a multivector result"))
    return coefficient(value, indices)
end

function _eval_coefficients(node::GAExpr, requests)
    op = node.operation
    if op in (:+, :-) && length(node.args) == 2
        left = _eval_coefficients(node.args[1], requests)
        right = _eval_coefficients(node.args[2], requests)
        return op == :+ ? left .+ right : left .- right
    elseif op in (:*, :∧, :wedge, :left_contraction, :right_contraction) &&
           length(node.args) == 2 && all(child -> child.operation == :leaf, node.args)
        left, right = node.args[1].args[1], node.args[2].args[1]
        if left isa AbstractMultiVector && right isa AbstractMultiVector
            operation = op == :* ? :geometric :
                        op in (:∧, :wedge) ? :wedge :
                        op == :left_contraction ? :left : :right
            return product_coefficients(left, right, requests; operation)
        end
    end
    value = _eval_ga(node)
    value isa AbstractMultiVector ||
        throw(ArgumentError("requested coefficients require a multivector result"))
    return [coefficient(value, indices) for indices in requests]
end

function _validate_expr_algebra(node::GAExpr, first_value=nothing)
    if node.operation == :leaf
        value = node.args[1]
        value isa AbstractMultiVector || return first_value
        first_value === nothing || _same_algebra(first_value, value)
        return value
    end
    for child in node.args
        first_value = _validate_expr_algebra(child, first_value)
    end
    return first_value
end

"""
    evaluate(expr::GAExpr; output=nothing, outputs=nothing)

Evaluate the captured expression. With `output=[indices...]`, sums of binary
products propagate a single requested coefficient. With
`outputs=[[indices...], ...]`, the requests are batched for each binary product.
Other expressions use the ordinary public operations.
"""
function evaluate(expr::GAExpr; output=nothing, outputs=nothing)
    output === nothing && outputs === nothing && return _eval_ga(expr)
    output === nothing || outputs === nothing ||
        throw(ArgumentError("request either output or outputs, not both"))
    _validate_expr_algebra(expr)
    return outputs === nothing ? _eval_coefficient(expr, output) :
           _eval_coefficients(expr, collect(outputs))
end

"""
    ExpressionPlan

A bounded DAG of a captured `@ga` expression. Equal operator subtrees share a
node only when their leaves are the same captured objects. The plan stores no
calculated coefficients; every `evaluate(plan)` reads the current leaves.
Coefficient arithmetic must be pure, and leaves must not mutate concurrently
with an evaluation.
"""
struct ExpressionPlan
    nodes::Vector{GAExpr}
    children::Vector{Vector{Int}}
    root::Int
end

Base.length(plan::ExpressionPlan) = length(plan.nodes)

"""Intern identical pure subexpressions into a DAG with `max_nodes` budget."""
function prepare_expression(expr::GAExpr; max_nodes::Integer=256)
    max_nodes >= 1 || throw(ArgumentError("max_nodes must be positive"))
    nodes = GAExpr[]
    children = Vector{Int}[]
    interned = Dict{Any,Int}()
    leaf_ids = IdDict{Any,Int}()
    function visit(node::GAExpr)
        child_ids = node.operation == :leaf ? Int[] :
            Int[visit(child) for child in node.args]
        key = if node.operation == :leaf
            value = node.args[1]
            (:leaf, get!(leaf_ids, value) do
                length(leaf_ids) + 1
            end)
        else
            (node.operation, Tuple(child_ids))
        end
        return get!(interned, key) do
            length(nodes) < max_nodes ||
                throw(ArgumentError("expression exceeds max_nodes"))
            push!(nodes, node)
            push!(children, child_ids)
            length(nodes)
        end
    end
    root = visit(expr)
    return ExpressionPlan(nodes, children, root)
end

function _apply_expr(op::Symbol, args::Vector{Any}, line::Int)
    op == :+ && return +(args...)
    op == :- && return -(args...)
    op == :* && return *(args...)
    op in (:∧, :wedge) && return wedge(args...)
    op == :left_contraction && return left_contraction(args...)
    op == :right_contraction && return right_contraction(args...)
    error("invalid GA expression node at line $line")
end

function _eval_plan_full!(plan::ExpressionPlan, id::Int,
                          values::Vector{Any}, done::BitVector)
    done[id] && return values[id]
    node = plan.nodes[id]
    value = node.operation == :leaf ? node.args[1] :
        _apply_expr(node.operation,
                    Any[_eval_plan_full!(plan, child, values, done)
                        for child in plan.children[id]], node.line)
    values[id] = value
    done[id] = true
    return value
end

function _eval_plan_requests!(plan::ExpressionPlan, id::Int, requests,
                              partial::Vector{Any}, partial_done::BitVector,
                              full::Vector{Any}, full_done::BitVector)
    partial_done[id] && return partial[id]
    node = plan.nodes[id]
    kids = plan.children[id]
    op = node.operation
    result = if op in (:+, :-) && length(kids) == 2
        left = _eval_plan_requests!(plan, kids[1], requests,
                                    partial, partial_done, full, full_done)
        right = _eval_plan_requests!(plan, kids[2], requests,
                                     partial, partial_done, full, full_done)
        op == :+ ? left .+ right : left .- right
    elseif op in (:*, :∧, :wedge, :left_contraction, :right_contraction) &&
           length(kids) == 2
        left = _eval_plan_full!(plan, kids[1], full, full_done)
        right = _eval_plan_full!(plan, kids[2], full, full_done)
        if left isa AbstractMultiVector && right isa AbstractMultiVector
            operation = op == :* ? :geometric :
                        op in (:∧, :wedge) ? :wedge :
                        op == :left_contraction ? :left : :right
            product_coefficients(left, right, requests; operation)
        else
            value = _eval_plan_full!(plan, id, full, full_done)
            value isa AbstractMultiVector ||
                throw(ArgumentError("requested coefficients require a multivector result"))
            [coefficient(value, indices) for indices in requests]
        end
    else
        value = _eval_plan_full!(plan, id, full, full_done)
        value isa AbstractMultiVector ||
            throw(ArgumentError("requested coefficients require a multivector result"))
        [coefficient(value, indices) for indices in requests]
    end
    partial[id] = result
    partial_done[id] = true
    return result
end

function _expr_pair_grades(op::Symbol, ra::Int, rb::Int, n::Int)
    if op in (:∧, :wedge)
        return ra + rb <= n ? (ra + rb,) : ()
    elseif op == :left_contraction
        return ra <= rb ? (rb - ra,) : ()
    elseif op == :right_contraction
        return rb <= ra ? (ra - rb,) : ()
    end
    # In a diagonal metric, the intersection has between max(0,r+s-n)
    # and min(r,s) directions. Zero metric entries can only remove terms.
    return (ra + rb - 2intersection
            for intersection in max(0, ra + rb - n):min(ra, rb))
end

function _expr_grades(plan::ExpressionPlan, ga::GeometricAlgebra)
    n = dimension(ga)
    grades = [Set{Int}() for _ in plan.nodes]
    for id in eachindex(plan.nodes)
        node = plan.nodes[id]
        kids = plan.children[id]
        result = grades[id]
        if node.operation == :leaf
            value = node.args[1]
            if value isa AbstractMultiVector
                for (mask, _) in _terms(value)
                    push!(result, _blade_grade(mask))
                end
            elseif value isa Number
                iszero(value) || push!(result, 0)
            else
                throw(ArgumentError("recursive coefficient mode requires numeric or multivector leaves"))
            end
        elseif node.operation in (:+, :-)
            for child in kids
                union!(result, grades[child])
            end
        elseif node.operation in (:*, :∧, :wedge, :left_contraction,
                                  :right_contraction) && length(kids) == 2
            for ra in grades[kids[1]], rb in grades[kids[2]]
                union!(result, _expr_pair_grades(node.operation, ra, rb, n))
            end
        else
            throw(ArgumentError("unsupported recursive expression operation"))
        end
    end
    return grades
end

function _expr_grade_demands(plan::ExpressionPlan, ga::GeometricAlgebra,
                             targets)
    possible = _expr_grades(plan, ga)
    demanded = [Set{Int}() for _ in plan.nodes]
    for target in targets
        g = _blade_grade(target)
        g in possible[plan.root] && push!(demanded[plan.root], g)
    end
    n = dimension(ga)
    # prepare_expression stores children before parents. Reverse order thus
    # gathers every parent's demands before propagating through a child.
    for id in length(plan.nodes):-1:1
        isempty(demanded[id]) && continue
        node = plan.nodes[id]
        kids = plan.children[id]
        if node.operation in (:+, :-)
            for child in kids
                union!(demanded[child], intersect(demanded[id], possible[child]))
            end
        elseif node.operation in (:*, :∧, :wedge, :left_contraction,
                                  :right_contraction) && length(kids) == 2
            left, right = kids
            for ra in possible[left], rb in possible[right]
                any(g -> g in demanded[id],
                    _expr_pair_grades(node.operation, ra, rb, n)) || continue
                push!(demanded[left], ra)
                push!(demanded[right], rb)
            end
        end
    end
    return possible, demanded
end

function _expr_support!(plan::ExpressionPlan, id::Int,
                        support::Dict{Int,Set{K}}, ::Type{K},
                        max_support::Integer, max_pairs::Integer,
                        demanded=nothing) where K
    haskey(support, id) && return support[id]
    node = plan.nodes[id]
    masks = Set{K}()
    wanted = demanded === nothing ? nothing : demanded.grades[id]
    if node.operation == :leaf
        value = node.args[1]
        if value isa AbstractMultiVector
            for (mask, _) in _terms(value)
                wanted === nothing || _blade_grade(mask) in wanted || continue
                push!(masks, convert(K, mask))
                length(masks) <= max_support ||
                    throw(ArgumentError("expression support exceeds max_support"))
            end
        elseif value isa Number
            (iszero(value) || (wanted !== nothing && !(0 in wanted))) ||
                push!(masks, zero(K))
        else
            throw(ArgumentError("recursive coefficient mode requires numeric or multivector leaves"))
        end
    else
        kids = plan.children[id]
        op = node.operation
        if op == :- && length(kids) == 1
            union!(masks, _expr_support!(plan, kids[1], support, K,
                                         max_support, max_pairs, demanded))
        elseif op in (:+, :-) && length(kids) == 2
            for child in kids
                union!(masks, _expr_support!(plan, child, support, K,
                                             max_support, max_pairs, demanded))
            end
        elseif op in (:*, :∧, :wedge, :left_contraction,
                      :right_contraction) && length(kids) == 2
            left = _expr_support!(plan, kids[1], support, K,
                                  max_support, max_pairs, demanded)
            right = _expr_support!(plan, kids[2], support, K,
                                   max_support, max_pairs, demanded)
            big(length(left)) * length(right) <= max_pairs ||
                throw(ArgumentError("expression support pairing exceeds max_pairs"))
            for a in left, b in right
                wanted === nothing ||
                    any(g -> g in wanted,
                        _expr_pair_grades(op, _blade_grade(a),
                                          _blade_grade(b), demanded.dimension)) ||
                    continue
                out = a ⊻ b
                if op in (:∧, :wedge)
                    iszero(a & b) || continue
                elseif op == :left_contraction
                    _blade_grade(a) <= _blade_grade(b) &&
                        _blade_grade(out) == _blade_grade(b) - _blade_grade(a) ||
                        continue
                elseif op == :right_contraction
                    _blade_grade(b) <= _blade_grade(a) &&
                        _blade_grade(out) == _blade_grade(a) - _blade_grade(b) ||
                        continue
                end
                wanted === nothing || _blade_grade(out) in wanted || continue
                push!(masks, out)
                length(masks) <= max_support ||
                    throw(ArgumentError("expression support exceeds max_support"))
            end
        else
            throw(ArgumentError("unsupported recursive expression operation"))
        end
    end
    length(masks) <= max_support ||
        throw(ArgumentError("expression support exceeds max_support"))
    support[id] = masks
    return masks
end

function _eval_plan_mask!(plan::ExpressionPlan, id::Int, target::K,
                          ga::GeometricAlgebra,
                          support::Dict{Int,Set{K}},
                          values::Dict{Tuple{Int,K},Any},
                          max_support::Integer, max_pairs::Integer,
                          demanded=nothing) where K
    key = (id, target)
    haskey(values, key) && return values[key]
    node = plan.nodes[id]
    op = node.operation
    kids = plan.children[id]
    result = if op == :leaf
        leaf = node.args[1]
        if leaf isa AbstractMultiVector
            coefficient_mask(leaf, target)
        elseif leaf isa Number
            iszero(target) ? leaf : zero(leaf)
        else
            throw(ArgumentError("recursive coefficient mode requires numeric or multivector leaves"))
        end
    elseif op == :- && length(kids) == 1
        -_eval_plan_mask!(plan, kids[1], target, ga, support, values,
                          max_support, max_pairs, demanded)
    elseif op in (:+, :-) && length(kids) == 2
        left = _eval_plan_mask!(plan, kids[1], target, ga, support, values,
                                max_support, max_pairs, demanded)
        right = _eval_plan_mask!(plan, kids[2], target, ga, support, values,
                                 max_support, max_pairs, demanded)
        op == :+ ? left + right : left - right
    elseif op in (:*, :∧, :wedge, :left_contraction,
                  :right_contraction) && length(kids) == 2
        left_support = _expr_support!(plan, kids[1], support, K,
                                      max_support, max_pairs, demanded)
        right_support = _expr_support!(plan, kids[2], support, K,
                                       max_support, max_pairs, demanded)
        scan_left = length(left_support) <= length(right_support)
        scanned = scan_left ? left_support : right_support
        other = scan_left ? right_support : left_support
        big(length(scanned)) <= max_pairs ||
            throw(ArgumentError("requested coefficient exceeds max_pairs"))
        sum_value = zero(eltype(metric(ga)))
        for known in scanned
            partner = known ⊻ target
            partner in other || continue
            amask, bmask = scan_left ? (known, partner) : (partner, known)
            if op in (:∧, :wedge)
                iszero(amask & bmask) || continue
            elseif op == :left_contraction
                _blade_grade(amask) <= _blade_grade(bmask) &&
                    _blade_grade(target) ==
                    _blade_grade(bmask) - _blade_grade(amask) || continue
            elseif op == :right_contraction
                _blade_grade(bmask) <= _blade_grade(amask) &&
                    _blade_grade(target) ==
                    _blade_grade(amask) - _blade_grade(bmask) || continue
            end
            factor = _shuffle_sign(ga, amask, bmask)
            if op ∉ (:∧, :wedge)
                for direction in _indices(ga, amask & bmask)
                    factor *= metric(ga)[direction, direction]
                end
            end
            iszero(factor) && continue
            avalue = _eval_plan_mask!(plan, kids[1], amask, ga, support,
                                      values, max_support, max_pairs, demanded)
            bvalue = _eval_plan_mask!(plan, kids[2], bmask, ga, support,
                                      values, max_support, max_pairs, demanded)
            sum_value += factor * avalue * bvalue
        end
        sum_value
    else
        throw(ArgumentError("unsupported recursive expression operation"))
    end
    values[key] = result
    return result
end

function _expression_triple_operands(plan::ExpressionPlan)
    root = plan.root
    plan.nodes[root].operation == :* && length(plan.children[root]) == 2 ||
        throw(ArgumentError("join3 requires a left-associated triple product"))
    middle, last = plan.children[root]
    plan.nodes[middle].operation == :* && length(plan.children[middle]) == 2 ||
        throw(ArgumentError("join3 requires a left-associated triple product"))
    first, second = plan.children[middle]
    ids = (first, second, last)
    all(id -> plan.nodes[id].operation == :leaf, ids) ||
        throw(ArgumentError("join3 requires three multivector leaves"))
    operands = map(id -> plan.nodes[id].args[1], ids)
    all(value -> value isa AbstractMultiVector, operands) ||
        throw(ArgumentError("join3 requires three multivector leaves"))
    return operands
end

"""
Evaluate a prepared pure-expression DAG. `strategy=:recursive` propagates
requested masks through nested products in a diagonal metric.
`strategy=:recursive_grades` first propagates possible and requested grades
through the entire expression, then limits support construction accordingly.
`strategy=:join3` directly joins three multivector supports for a
left-associated geometric triple product and one or more requested outputs.
Both strategies have explicit support and pairing budgets. The default
strategy keeps the cheaper local batched path for simple expressions.
"""
function evaluate(plan::ExpressionPlan; output=nothing, outputs=nothing,
                  strategy::Symbol=:auto, max_support::Integer=1 << 12,
                  max_pairs::Integer=1 << 20)
    output === nothing || outputs === nothing ||
        throw(ArgumentError("request either output or outputs, not both"))
    strategy in (:auto, :recursive, :recursive_grades, :join3) ||
        throw(ArgumentError("unknown expression evaluation strategy"))
    full = Vector{Any}(undef, length(plan))
    full_done = falses(length(plan))
    if output === nothing && outputs === nothing
        return _eval_plan_full!(plan, plan.root, full, full_done)
    end
    first_value = _validate_expr_algebra(plan.nodes[plan.root])
    requests = outputs === nothing ? [output] : collect(outputs)
    if strategy == :join3
        a, b, c = _expression_triple_operands(plan)
        result = [triple_product_coefficient(a, b, c, indices;
                                              max_pairs) for indices in requests]
        return outputs === nothing ? only(result) : result
    end
    if strategy in (:recursive, :recursive_grades)
        max_support >= 1 && max_pairs >= 1 ||
            throw(ArgumentError("recursive expression budgets must be positive"))
        first_value === nothing &&
            throw(ArgumentError("requested coefficients require a multivector expression"))
        ga = first_value.algebra
        isdiag(metric(ga)) ||
            throw(ArgumentError("recursive expression strategy requires a diagonal metric"))
        K = _masktype(ga)
        targets = K[_mask(ga, indices) for indices in requests]
        demanded = if strategy == :recursive_grades
            _, grades = _expr_grade_demands(plan, ga, targets)
            (; grades, dimension=dimension(ga))
        else
            nothing
        end
        support = Dict{Int,Set{K}}()
        values = Dict{Tuple{Int,K},Any}()
        result = [_eval_plan_mask!(plan, plan.root, target, ga,
                                   support, values, max_support, max_pairs,
                                   demanded) for target in targets]
        return outputs === nothing ? only(result) : result
    end
    partial = Vector{Any}(undef, length(plan))
    partial_done = falses(length(plan))
    result = _eval_plan_requests!(plan, plan.root, requests,
                                  partial, partial_done, full, full_done)
    return outputs === nothing ? only(result) : result
end
