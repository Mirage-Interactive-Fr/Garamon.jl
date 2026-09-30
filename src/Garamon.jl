module Garamon

# SECTION - includes
using DelimitedFiles
using Dictionaries
using LinearAlgebra
using PrettyTables
using Random
using SparseArrays
import SparseArrays: sparse
using StaticArrays

# SECTION - exports
export algebra
export basis
export dimension
export kind
export list_geometric_algebras
export metric
export multiplicity
export object_dim
export AbstractMultiVector, DenseMultiVector, SparseMultiVector
export basisblade, basisvector, multivector, scalar, coefficient, coefficient_mask
export dense, sparse, geometric_product, wedge, left_contraction, right_contraction
export inner_product, dot_product, scalar_product
export product_coefficient
export triple_product_coefficient
export quadruple_product_coefficients
export sorted_wedge_coefficient
export product_coefficients
export outermorphism, FactorizedBlade, expand
export ReflectionChain, versor_action
export grade_involution, clifford_conjugate
export right_complement, right_uncomplement, metric_dual, metric_undual
export basisvectors, @algebra, GAExpr, @ga, evaluate
export ExpressionPlan, prepare_expression
export ProductPlan, prepare_product, run_product, batch_product
export ProductWorkspace, run_product_values!, run_product!
export TripleJoinPlan, prepare_triple_join, TripleJoinWorkspace
export run_triple_join_values!
export TripleSelectorBudget, TripleSelectionRefusal, ExactTripleSelector
export triple_strategy_features, triple_strategy_admission, choose_triple_strategy
export prepare_triple_selector, run_selected_triple!
export GeneratedProduct, generate_product, run_generated_product
export prepare_grade_product, run_grade_product
export PackedProductBatch, pack_product_batch, run_packed_batch
export unpack_product_batch
export ProductPlanCache, cached_plan!, cached_product!, cache_stats
export ModularProductPlan, prepare_modular_product, run_modular_product
export modular_geometric_product
export TopWedgePlan, prepare_top_wedge, top_wedge_coefficient
export MaskTrie, build_trie, trie_wedge, count_nodes, wedge_visits
export WedgeCoefficientPlan, prepare_wedge_coefficient, wedge_coefficient
export CliffordTrain, separable_train, train_product
export MetricDiagonalization, diagonalize_metric, to_orthogonal, from_orthogonal
export CoordinateSubspace, coordinate_subspace, project_subspace, lift_subspace
export subspace_product
export WeightedZDD, weighted_zdd, weighted_zdd_product
export weighted_zdd_coefficient, weighted_zdd_terms
export ∧
export grade, scalarpart, reverse
export active_grades, highest_grade, is_homogeneous, set_coefficient!
export has_grade, same_grade, coefficient_grade, set_grade_coefficient!
export blade_rank, blade_unrank
export clear_grade!, round_zero!, quadratic_norm, clifford_norm
export outer_primal_dual, outer_dual_primal, outer_dual_dual

# SECTION - includes
include("algebra.jl")
include("conf.jl")
include("multivectors.jl")
include("indexing.jl")
include("products.jl")
include("transformations.jl")
include("plans.jl")
include("workspace.jl")
include("triplejoin.jl")
include("generated.jl")
include("cache.jl")
include("modular.jl")
include("gradeblocks.jl")
include("materialized_trie.jl")
include("trains.jl")
include("duality.jl")
include("operations.jl")
include("dsl.jl")
include("selector.jl")
include("decomposition.jl")
include("subspace.jl")
include("weighted_zdd.jl")

end
