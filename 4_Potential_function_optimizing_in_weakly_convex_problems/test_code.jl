## Test code

## Include the main Julia file

include("BnB-PEP-potential-minimization-weakly-convex-problems.jl")

## ## Parameters to use

N = 3
R = .1
M = 1
default_obj_val_upper_bound = 1e6
ϵ_tol_feas = 1e-6
ϵ_tol_Cholesky = 1e-4

## Feasible stepsize generation

h_feas = feasible_h_generator(N, M, R; step_size_type = :Default)

# -------------------------------------------------------
## Stage 1 of the BnB-PEP Algorithm: solve the dual for the warm-starting stepsize
# -------------------------------------------------------


d_star_feas, ℓ_1_norm_λ_feas, ℓ_1_norm_τ_feas, tr_Z_sum_feas, λ_feas, τ_feas, Z_feas, L_cholesky_feas, b_feas, c_feas, idx_set_λ_effective_feas, idx_set_τ_effective_feas = solve_dual_with_known_stepsizes(N, M, h_feas; show_output = :on, ϵ_tol_feas = 1e-6,
    objective_type = :default, obj_val_upper_bound = default_obj_val_upper_bound)

## store the warm start point for computing locally optimal solution

d_star_ws, λ_ws, τ_ws, Z_ws, L_cholesky_ws, h_ws, b_ws, c_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective = d_star_feas, λ_feas, τ_feas, Z_feas, L_cholesky_feas, h_feas, b_feas, c_feas, idx_set_λ_effective_feas, idx_set_τ_effective_feas

##  Computing the bounds for computing locally optimal solution to BnB-PEP

M_tilde = 10

# Compute M_λ

d_star_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, tr_Z_sum_feas_1, λ_feas_1, τ_feas_1, Z_feas_1, L_cholesky_feas_1, b_feas_1, c_feas_1, idx_set_λ_effective_feas_1, idx_set_τ_effective_feas_1 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_λ, obj_val_upper_bound = 1.001*d_star_ws)

M_λ = M_tilde*maximum(λ_feas_1)

## Compute M_τ

d_star_feas_2, ℓ_1_norm_λ_feas_2, ℓ_1_norm_τ_feas_2, tr_Z_sum_feas_2, λ_feas_2, τ_feas_2, Z_feas_2, L_cholesky_feas_2, b_feas_2, c_feas_2, idx_set_λ_effective_feas_2, idx_set_τ_effective_feas_2 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_τ, obj_val_upper_bound = 1.001*d_star_ws)

M_τ = M_tilde*maximum(τ_feas_2)

## Compute M_Z

d_star_feas_3, ℓ_1_norm_λ_feas_3, ℓ_1_norm_τ_feas_3, tr_Z_sum_feas_3, λ_feas_3, τ_feas_3, Z_feas_3, L_cholesky_feas_3, b_feas_3, c_feas_3, idx_set_λ_effective_feas_3, idx_set_τ_effective_feas_3 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_Z, obj_val_upper_bound = 1.001*d_star_ws)

max_elm_Z_ws = []

for k in 0:N
    push!(max_elm_Z_ws, maximum(abs.(Z_feas_3[k])))
end

M_Z = M_tilde*maximum(abs.(max_elm_Z_ws))


## Compute M_cholesky

M_L_cholesky = sqrt(M_Z)

## Compute M_h

M_h = 5*M_tilde*maximum(abs.(h_ws))

## compute M_b

d_star_feas_4, ℓ_1_norm_λ_feas_4, ℓ_1_norm_τ_feas_4, tr_Z_sum_feas_4, λ_feas_4, τ_feas_4, Z_feas_4, L_cholesky_feas_4, b_feas_4, c_feas_4, idx_set_λ_effective_feas_4, idx_set_τ_effective_feas_4 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_b, obj_val_upper_bound = 1.001*d_star_ws)

M_b = M_tilde*maximum(b_feas_4)

## compute M_c

d_star_feas_5, ℓ_1_norm_λ_feas_5, ℓ_1_norm_τ_feas_5, tr_Z_sum_feas_5, λ_feas_5, τ_feas_5, Z_feas_5, L_cholesky_feas_5, b_feas_5, c_feas_5, idx_set_λ_effective_feas_5, idx_set_τ_effective_feas_5 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_c, obj_val_upper_bound = 1.001*d_star_ws)

M_c = M_tilde*max(1,2*maximum(c_feas_5)) #2*max(1,maximum(c_feas_5))

@show [M_λ M_τ M_h M_Z M_L_cholesky M_b M_c]

## sparsify the solution for warm-starting locally optimal solver

d_star_feas, ℓ_1_norm_λ_feas, ℓ_1_norm_τ_feas, tr_Z_sum_feas, λ_feas, τ_feas, Z_feas, L_cholesky_feas, b_feas, c_feas, idx_set_λ_effective_feas, idx_set_τ_effective_feas =  solve_dual_with_known_stepsizes(N, M, h_ws;
    show_output = :on,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_sparse_sol,
    obj_val_upper_bound = d_star_ws)

# ---------------------------------------------------
## Stage 2 of the BnB-PEP Algorithm: compute the locally optimal point
# ----------------------------------------------------


obj_val_loc_opt,
λ_loc_opt,
τ_loc_opt,
Z_loc_opt,
L_cholesky_loc_opt,
b_loc_opt,
c_loc_opt,
Θ_loc_opt,
h_loc_opt,
idx_set_λ_loc_opt_effective,
idx_set_τ_loc_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N,
    M,
    R,
    # solution to warm-start (Θ is warm-started internally)
    # -----------------------------------------------------
    d_star_ws,
    λ_ws,
    τ_ws,
    Z_ws,
    L_cholesky_ws,
    h_ws,
    b_ws,
    c_ws,
    idx_set_λ_ws_effective,
    idx_set_τ_ws_effective,
    # bounds on the variables (M_Θ is computed internally)
    # ----------------------------------------------------
    M_λ,
    M_τ,
    M_h,
    M_Z,
    M_L_cholesky,
    M_b,
    M_c;
    # options
    # -------
    solution_type = :find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    local_solver = :knitro, #:ipopt, # :ipopt, # other option :knitro
    knitro_multistart = :off, # other option :on (only if :knitro solver is used)
    knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
    reduce_index_set_for_dual_variables = :for_warm_start_only,
    reduce_index_set_for_L_cholesky = :off, # the other option is :on
    positive_step_size = :on, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :off, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation,
    quadratic_equality_modeling = :exact, #:through_ϵ, # other option is :exact
    cholesky_modeling = :definition, #:formula, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-6, # tolerance for feasibility
    ϵ_tol_Cholesky = 0.0005, # tolerance for determining which elements of L_cholesky_ws is zero
    maxCutCount = 1e4, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :off, # wheather to polish the solution to get better precision, the other option is :off,
    M_Θ_factor = M_tilde, # factor by which to magnify the internal M_Θ
    impose_pattern = :on
)

## Test the linear constraint
#
# k = 0
#
# 𝐰, 𝐠, 𝐟 = data_generator_potential_pep(h_loc_opt,k; input_type = :stepsize_constant)
#
# term_1 = -(b_loc_opt[k+1]*a_vec(-1,3,𝐟) - b_loc_opt[k]*a_vec(-1,2,𝐟))
#
# term_2 = sum(τ_loc_opt[i_k_idx(i,k)]*a_vec(i,-1,𝐟) for i in 0:3)
#
# term_3 = sum(λ_loc_opt[i_j_k_λ]*a_vec(i_j_k_λ.i,i_j_k_λ.j,𝐟) for i_j_k_λ in λ_loc_opt.axes[1] if i_j_k_λ.k == k)

# Test the relationship between b and λ (comment out later)

# k = 2
#
# b_loc_opt[k] - (4-λ_loc_opt[i_j_k_idx(2,0,k)] + b_loc_opt[k+1])
#
# b_loc_opt[k+1] - λ_loc_opt[i_j_k_idx(2,3,k)]
#
# (2*h_loc_opt[k]*λ_loc_opt[i_j_k_idx(2,3,k)])-(λ_loc_opt[i_j_k_idx(2,0,k)])
#
# c_loc_opt[k] - (h_loc_opt[k]^2*b_loc_opt[k+1])
#
# b_loc_opt[k] - (4+((1-(2*h_loc_opt[k]))*b_loc_opt[k+1]))
#
# τ_loc_opt[i_k_idx(2,k)] - (b_loc_opt[k]-b_loc_opt[k+1])
#
# bFlippedAnalytic(k,h) = (2/h)*(1-(1-2h)^k)
#
# bAnalytic(k,h) =  (2/h)*(1-(1-2h)^(N+1-k))
#
# b_loc_opt
#
# bAnalytic(N+1,h_loc_opt[k])
#
# bAnalyticArray = [bAnalytic(i,h_loc_opt[0]) for i in 0:N+1]
#
# [b_loc_opt[i] for i in 0:N+1] - bAnalyticArray

## Store the solution to be warm-started for a next step

d_star_ws, λ_ws, τ_ws, Z_ws, L_cholesky_ws, h_ws, b_ws, c_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective = obj_val_loc_opt, λ_loc_opt, τ_loc_opt, Z_loc_opt, L_cholesky_loc_opt, h_loc_opt, b_loc_opt, c_loc_opt, idx_set_λ_loc_opt_effective, idx_set_τ_loc_opt_effective

h_ws = h_loc_opt

## update the entries of the bounds based on the heuristic

##  Computing the bounds for computing locally optimal solution to BnB-PEP

M_tilde = 1.01

## compute M_λ

d_star_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, tr_Z_sum_feas_1, λ_feas_1, τ_feas_1, Z_feas_1, L_cholesky_feas_1, b_feas_1, c_feas_1, idx_set_λ_effective_feas_1, idx_set_τ_effective_feas_1 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_λ, obj_val_upper_bound = 1.001*d_star_ws)

M_λ = M_tilde*maximum(λ_feas_1)

## Compute M_τ

d_star_feas_2, ℓ_1_norm_λ_feas_2, ℓ_1_norm_τ_feas_2, tr_Z_sum_feas_2, λ_feas_2, τ_feas_2, Z_feas_2, L_cholesky_feas_2, b_feas_2, c_feas_2, idx_set_λ_effective_feas_2, idx_set_τ_effective_feas_2 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_τ, obj_val_upper_bound = 1.001*d_star_ws)

M_τ = M_tilde*maximum(τ_feas_2)

## Compute M_Z

d_star_feas_3, ℓ_1_norm_λ_feas_3, ℓ_1_norm_τ_feas_3, tr_Z_sum_feas_3, λ_feas_3, τ_feas_3, Z_feas_3, L_cholesky_feas_3, b_feas_3, c_feas_3, idx_set_λ_effective_feas_3, idx_set_τ_effective_feas_3 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_Z, obj_val_upper_bound = 1.001*d_star_ws)

max_elm_Z_ws = []

for k in 0:N
    push!(max_elm_Z_ws, maximum(abs.(Z_feas_3[k])))
end

M_Z = M_tilde*maximum(abs.(max_elm_Z_ws))


## Compute M_cholesky

M_L_cholesky = sqrt(M_Z)

## Compute M_h

M_h = 5*M_tilde*maximum(abs.(h_ws))

## compute M_b

d_star_feas_4, ℓ_1_norm_λ_feas_4, ℓ_1_norm_τ_feas_4, tr_Z_sum_feas_4, λ_feas_4, τ_feas_4, Z_feas_4, L_cholesky_feas_4, b_feas_4, c_feas_4, idx_set_λ_effective_feas_4, idx_set_τ_effective_feas_4 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_b, obj_val_upper_bound = 1.001*d_star_ws)

M_b = M_tilde *maximum(b_feas_4)

## compute M_c

d_star_feas_5, ℓ_1_norm_λ_feas_5, ℓ_1_norm_τ_feas_5, tr_Z_sum_feas_5, λ_feas_5, τ_feas_5, Z_feas_5, L_cholesky_feas_5, b_feas_5, c_feas_5, idx_set_λ_effective_feas_5, idx_set_τ_effective_feas_5 =  solve_dual_with_known_stepsizes(N, M, h_ws; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type =  :find_M_c, obj_val_upper_bound = 1.001*d_star_ws)

M_c = M_tilde*max(1,2*maximum(c_feas_5)) #2*max(1,maximum(c_feas_5))



## Compute global lower bound via lazy callback

obj_val_glb_lbd,
λ_glb_lbd,
τ_glb_lbd,
Z_glb_lbd,
L_cholesky_glb_lbd,
b_glb_lbd,
c_glb_lbd,
Θ_glb_lbd,
h_glb_lbd,
idx_set_λ_glb_lbd_effective,
idx_set_τ_glb_lbd_effective = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N,
    M,
    R,
    # solution to warm-start (Θ is warm-started internally)
    # -----------------------------------------------------
    d_star_ws,
    λ_ws,
    τ_ws,
    Z_ws,
    L_cholesky_ws,
    h_ws,
    b_ws,
    c_ws,
    idx_set_λ_ws_effective,
    idx_set_τ_ws_effective,
    # bounds on the variables (M_Θ is computed internally)
    # ----------------------------------------------------
    M_λ,
    M_τ,
    M_h,
    M_Z,
    M_L_cholesky,
    M_b,
    M_c;
    # options
    # -------
    solution_type = :find_globally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    local_solver = :ipopt, # :ipopt, # other option :knitro
    knitro_multistart = :off, # other option :on (only if :knitro solver is used)
    knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
    reduce_index_set_for_dual_variables = :for_warm_start_only, # other options are :on and :off
    reduce_index_set_for_L_cholesky = :off, # the other option is :on
    positive_step_size = :on, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :on, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation,
    quadratic_equality_modeling = :exact, #:through_ϵ, #:through_ϵ, # other option is :exact
    cholesky_modeling = :formula, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-6, # tolerance for feasibility
    ϵ_tol_Cholesky = 0.0005, # tolerance for determining which elements of L_cholesky_ws is zero
    maxCutCount = 1e3, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :off, # wheather to polish the solution to get better precision, the other option is :off,
    M_Θ_factor = 1.1, # factor by which to magnify the internal M_Θ
    impose_pattern = :on
)


# ----------------------------------------------------
## Stage 3 of the BnB-PEP Algorithm: find the globally optimal solution to the BnB-PEP-QCQP
# ----------------------------------------------------


obj_val_glb_opt,
λ_glb_opt,
τ_glb_opt,
Z_glb_opt,
L_cholesky_glb_opt,
b_glb_opt,
c_glb_opt,
Θ_glb_opt,
h_glb_opt,
idx_set_λ_glb_opt_effective,
idx_set_τ_glb_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N,
    M,
    R,
    # solution to warm-start (Θ is warm-started internally)
    # -----------------------------------------------------
    d_star_ws,
    λ_ws,
    τ_ws,
    Z_ws,
    L_cholesky_ws,
    h_ws,
    b_ws,
    c_ws,
    idx_set_λ_ws_effective,
    idx_set_τ_ws_effective,
    # bounds on the variables (M_Θ is computed internally)
    # ----------------------------------------------------
    M_λ,
    M_τ,
    M_h,
    M_Z,
    M_L_cholesky,
    M_b,
    M_c;
    # options
    # -------
    solution_type = :find_globally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    local_solver = :ipopt, # :ipopt, # other option :knitro
    knitro_multistart = :off, # other option :on (only if :knitro solver is used)
    knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
    reduce_index_set_for_dual_variables = :off, # :for_warm_start_only,
    reduce_index_set_for_L_cholesky = :off, # the other option is :on
    positive_step_size = :on, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :off, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation,
    quadratic_equality_modeling = :exact, #:through_ϵ, # other option is :exact
    cholesky_modeling = :formula, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-6, # tolerance for feasibility
    ϵ_tol_Cholesky = 0.0005, # tolerance for determining which elements of L_cholesky_ws is zero
    maxCutCount = 1e3, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :on, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = obj_val_glb_lbd, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :off, # wheather to polish the solution to get better precision, the other option is :off,
    M_Θ_factor = 1.1, # factor by which to magnify the internal M_Θ
    impose_pattern = :on # Caution if impose_pattern == :on, keep reduce_index_set_for_dual_variables = :off and reduce_index_set_for_L_cholesky = :off, and vice versa
)



##
