## Test code

## Load the code

include("BnB-PEP-gradient-reduction-scvx-smooth.jl")


## Input parameters and feasible stepsize generation

μ = 0.1
L = 1
N = 2
R = 1
default_obj_val_upper_bound = 1e6


h_test, α_test = feasible_h_α_generator(N, μ, L; step_size_type = :Default)

## Solve primal with feasible stepsize

p_feas_1, G_feas_1, Ft_feas_1 = solve_primal_with_known_stepsizes(N, μ, L, α_test, R; show_output = :on)

# -------------------------------------------------------
## Stage 1 of the BnB-PEP Algorithm: solve the dual for the warm-starting stepsize
# -------------------------------------------------------

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, μ, L, α_test, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)


## Compute bound based on SDP relaxation of the BnB-PEP, uncomment if you prefer this method (and comment the heuristic one)

# M_λ, M_α, M_Z, M_L_cholesky, M_ν = bound_generator_through_SDP_relaxation(N, μ, L, R,  ν_feas_1; show_output = :off, obj_val_upper_bound = d_feas_1)

## Compute bound based on heuristic method

M_tilde = 50

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, μ, L, α_feas_1, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_λ,
    obj_val_upper_bound = 1.001*p_feas_1)

M_λ =M_tilde*maximum(λ_feas_1)

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, μ, L, α_feas_1, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_Z,
    obj_val_upper_bound = 1.001*p_feas_1)

M_Z = M_tilde*maximum(Z_feas_1[i,i] for i in 1:N+2)

M_L_cholesky = sqrt(M_Z)

M_α = (5*M_tilde)/L

M_ν = ν_feas_1

@show [M_λ M_α M_Z M_L_cholesky M_ν]

## Sparsify the solution

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, μ, L, α_feas_1, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_sparse_sol,
    obj_val_upper_bound = 1.001*p_feas_1)


## Store the warm start point for computing locally optimal solution

d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective = d_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective


# ---------------------------------------------------
## Stage 2 of the BnB-PEP Algorithm: compute the locally optimal point
# ----------------------------------------------------

obj_val_loc_opt, λ_loc_opt, ν_loc_opt, Z_loc_opt, L_cholesky_loc_opt, α_loc_opt, idx_set_λ_loc_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N, μ, L, R,
    # solution to warm-start
    # ----------------------
    d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective,
    # bounds on the variables
    # ----------------------
    M_λ, M_α, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type = :find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    local_solver = :ipopt, # other option :knitro
    reduce_index_set_for_λ = :for_warm_start_only,
    # options for reduce_index_set_for_λ
    # (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective, and will turn the BnB-PEP solver into a heuristic),
    # (ii) :off , this will define λ and warm-start over the full index set
    # (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
    quadratic_equality_modeling = :exact,
    cholesky_modeling = :definition,
    ϵ_tol_feas = 1e-6, # tolerance for feasibility
    polish_solution = :on # wheather to polish the solution to get better precision, the other option is :off
)

## Store the solution to be warm-started for a next step

d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective = obj_val_loc_opt, λ_loc_opt, ν_loc_opt, Z_loc_opt, L_cholesky_loc_opt, α_loc_opt, idx_set_λ_loc_opt_effective

## Update the entries of the bounds

# Based on the SDP relaxation of the BnB-PEP (uncomment if you prefer this method, and the comment the heuristic method below)
# M_λ, M_α, M_Z, M_L_cholesky, M_ν = bound_generator_through_SDP_relaxation(N, μ, L, R, ν_ws; show_output = :off, obj_val_upper_bound = d_star_ws)


# Bounds based on the heuristic

M_tilde = 1.01

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, μ, L, α_ws, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_Z,
    obj_val_upper_bound = 1.001*d_star_ws)

M_Z = M_tilde*maximum(Z_feas_1[i,i] for i in 1:N+2)

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, α_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, μ, L, α_ws, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_λ,
    obj_val_upper_bound = 1.001*d_star_ws)

M_λ = M_tilde* maximum(λ_feas_1)

M_L_cholesky = sqrt(M_Z)

M_α = 5*M_tilde*maximum(abs.(α_loc_opt))

M_ν = ν_ws# ν_feas_1

@show [M_λ M_α M_Z M_L_cholesky M_ν]


## Compute the global lower bound via the lazy callback procedure

obj_val_glb_lbd, λ_glb_lbd, ν_glb_lbd, Z_glb_lbd, L_cholesky_glb_lbd, α_glb_lbd, idx_set_λ_glb_lbd_effective = BnB_PEP_solver(
    # different parameters to be used
    # -------------------------------
    N, μ, L, R,
    # solution to warm-start
    # ----------------------
    d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective,
    # bounds on the variables
    # ----------------------
    M_λ, M_α, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type =  :find_globally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    reduce_index_set_for_λ = :on, # :for_warm_start_only,
    # options for reduce_index_set_for_λ
    # (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective, and will turn the BnB-PEP solver into a heuristic),
    # (ii) :off , this will define λ and warm-start over the full index set
    # (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
    positive_step_size = :off, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :on, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
    quadratic_equality_modeling = :exact, #:through_ϵ,
    cholesky_modeling = :definition, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-4, # tolerance for feasibility
    maxCutCount=1e6, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    heuristic_solution_submit = :off, # other option is :on, turning it on means that at the node of the spatial branch and bound tree we will take a look at the relaxed solution and if it satisfies certain condition, we will submit a heuristic solution
    polish_solution = :on # wheather to polish the solution to get better precision, the other option is :off
)

# ----------------------------------------------------
## Stage 3 of the BnB-PEP Algorithm: find the globally optimal solution to the BnB-PEP-QCQP
# ----------------------------------------------------


obj_val_glb_opt, λ_glb_opt, ν_glb_opt, Z_glb_opt, L_cholesky_glb_opt, α_glb_opt, idx_set_λ_glb_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # -------------------------------
    N, μ, L, R,
    # solution to warm-start
    # ----------------------
    d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective,
    # bounds on the variables
    # ----------------------
    M_λ, M_α, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type =  :find_globally_optimal, #:find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    reduce_index_set_for_λ = :on, #:for_warm_start_only, #:on,#:for_warm_start_only, # other option :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective, and will turn the BnB-PEP solver into a heuristic),
    reduce_index_set_for_L_cholesky = :on, #other option is :off,
    bound_impose = :on, # if this is :from_warm_start_sol, then from the warm_start solution we compute lower and upper bounds for the decision variables,
    quadratic_equality_modeling = :exact, #other option is :through_ϵ,
    cholesky_modeling = :formula, #other option is :definition,
    ϵ_tol_feas = 1e-4, # tolerance for cholesky feasibility
    global_lower_bound_given = :on, # wheather is a global lower bound is given, other option is :off
    global_lower_bound = obj_val_glb_lbd, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on # wheather to polish the solution to get better precision, the other option is :off
    )

## Optimal stepsize

h_glb_opt = compute_h_from_α(α_glb_opt, N, μ, L)
