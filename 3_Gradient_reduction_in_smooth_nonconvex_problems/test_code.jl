## Test code

## Include the main Julia file

include("BnB_PEP_gradient_reduction_smooth_ncvx.jl")

## Parameters to use

L = 1
N = 3
R = 1
default_obj_val_upper_bound = 1e6

## Feasible stepsize generation

h_test, α_test = feasible_h_α_generator(N, L; step_size_type = :Default)
## Solve primal with feasible stepsize

p_feas_1, G_feas_1, Ft_feas_1 = solve_primal_with_known_stepsizes(N, R, L, α_test; show_output = :off)


# -------------------------------------------------------
## Stage 1 of the BnB-PEP Algorithm: solve the dual for the warm-starting stepsize
# -------------------------------------------------------


d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_test;
    show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)

@info "primal-dual gap = $(p_feas_1-d_feas_1)"


##  Computing the bounds for computing locally optimal solution to BnB-PEP

M_tilde = 50

# compute M_λ

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_feas_1;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_λ,
    obj_val_upper_bound = 1.001*p_feas_1)

M_λ =M_tilde*maximum(λ_feas_1)

# compute M_τ

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_feas_1;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_τ,
    obj_val_upper_bound = 1.001*p_feas_1)

M_τ = M_tilde*maximum(τ_feas_1)

# compute M_η

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_feas_1;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_η,
    obj_val_upper_bound = 1.001*p_feas_1)

M_η = M_tilde*maximum(η_feas_1)

# compute M_Z

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_feas_1;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_Z,
    obj_val_upper_bound = 1.001*p_feas_1)


M_Z = M_tilde*max(1,maximum(Z_feas_1[i,i] for i in 1:N+2))

# compute M_L_cholesky

M_L_cholesky = sqrt(M_Z)

# compute M_α

M_α = (5*M_tilde)/L

# compute M_ν

M_ν = ν_feas_1

@show [M_λ M_τ M_η M_α M_Z M_L_cholesky M_ν]

## sparsify the solution for warm-starting locally optimal solver

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_feas_1;
    show_output = :on,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_sparse_sol,
    obj_val_upper_bound =  p_feas_1)


## store the warm start point for computing locally optimal solution


d_star_ws, λ_ws, τ_ws, η_ws, ν_ws, Z_ws, L_cholesky_ws, Θ_ws, α_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective, idx_set_η_ws_effective = d_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective


# ---------------------------------------------------
## Stage 2 of the BnB-PEP Algorithm: compute the locally optimal point
# ----------------------------------------------------


obj_val_loc_opt, λ_loc_opt, τ_loc_opt, η_loc_opt, ν_loc_opt, Z_loc_opt, L_cholesky_loc_opt, Θ_loc_opt, α_loc_opt, idx_set_λ_loc_opt_effective, idx_set_τ_loc_opt_effective, idx_set_η_loc_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N, L, R,
    # solution to warm-start (Θ is warm-started internally)
    # -----------------------------------------------------
    d_star_ws, λ_ws, τ_ws, η_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective, idx_set_η_ws_effective,
    # bounds on the variables (M_Θ is computed internally)
    # ----------------------------------------------------
    M_λ, M_τ, M_η, M_α, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type = :find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    local_solver = :ipopt, # other option :knitro
    reduce_index_set_for_dual_variables = :off, #other option is :for_warm_start_only,
    positive_step_size = :on, # other option is :off
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
    quadratic_equality_modeling = :exact, #:through_ϵ, # :exact,
    cholesky_modeling = :definition,
    ϵ_tol_feas = 1e-6, # tolerance for Cholesky feasibility
    polish_solution = :on, # wheather to polish the solution to get better precision, the other option is :off
    M_Θ_factor = M_tilde, # upper bound factor for Θ
    impose_pattern = :off # wheather to impose the pattern found by solving smaller BnB-PEPs to global optimality
)

# Store the solution to be warm-started for a next step


## values for the tables

using LatexPrint

@show round(obj_val_loc_opt, digits = 7)

h_loc_opt = compute_h_from_α(α_loc_opt, N, L)

h_matlab_format = round.(OffsetArrays.no_offset_view(h_loc_opt),digits = 6)
h_latex = lap(h_matlab_format)

@show h_latex



## Store the solution to be warm-started for a next step

d_star_ws, λ_ws, τ_ws, η_ws, ν_ws, Z_ws, L_cholesky_ws, Θ_ws, α_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective, idx_set_η_ws_effective = obj_val_loc_opt, λ_loc_opt, τ_loc_opt, η_loc_opt, ν_loc_opt, Z_loc_opt, L_cholesky_loc_opt, Θ_loc_opt, α_loc_opt, idx_set_λ_loc_opt_effective, idx_set_τ_loc_opt_effective, idx_set_η_loc_opt_effective

## Update the entries of the bounds based on the heuristic

##  Computing the bounds for computing locally optimal solution to BnB-PEP

M_tilde = 1.01

# compute M_λ

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_ws;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_λ,
    obj_val_upper_bound = 1.001*d_star_ws)

M_λ = M_tilde*maximum(λ_feas_1)

# compute M_τ

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_ws;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_τ,
    obj_val_upper_bound = 1.001*d_star_ws)

M_τ = M_tilde*maximum(τ_feas_1)

# compute M_η

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_ws;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_η,
    obj_val_upper_bound = 1.001*d_star_ws)

M_η = M_tilde*maximum(η_feas_1)

# compute M_Z

d_feas_1, ℓ_1_norm_λ_feas_1, ℓ_1_norm_τ_feas_1, ℓ_1_norm_η_feas_1, tr_Z_feas_1, λ_feas_1, τ_feas_1, η_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, Θ_feas_1, α_feas_1, idx_set_λ_feas_1_effective, idx_set_τ_feas_1_effective, idx_set_η_1_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_ws;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_Z,
    obj_val_upper_bound = 1.001*d_star_ws)


M_Z = M_tilde*max(1,maximum(Z_feas_1[i,i] for i in 1:N+2))

# compute M_L_cholesky

M_L_cholesky = sqrt(M_Z)

# compute M_α

M_α = (5*M_tilde)/L

# compute M_ν

M_ν = ν_feas_1

@show [M_λ M_τ M_η M_α M_Z M_L_cholesky M_ν]



## Compute global lower bound via lazy callback

obj_val_glb_lbd, λ_glb_lbd, τ_glb_lbd, η_glb_lbd, ν_glb_lbd, Z_glb_lbd, L_cholesky_glb_lbd, Θ_glb_lbd, α_glb_lbd, idx_set_λ_glb_lbd_effective, idx_set_τ_glb_lbd_effective, idx_set_η_glb_lbd_effective  = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N, L, R,
    # solution to warm-start (Θ is warm-started internally)
    # -----------------------------------------------------
    d_star_ws, λ_ws, τ_ws, η_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective, idx_set_η_ws_effective,
    # bounds on the variables (M_Θ is computed internally)
    # ----------------------------------------------------
    M_λ, M_τ, M_η, M_α, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type =  :find_globally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    reduce_index_set_for_dual_variables = :on, # :for_warm_start_only,
    positive_step_size = :on, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :on, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
    quadratic_equality_modeling = :exact, #:through_ϵ,
    cholesky_modeling = :definition, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-6, # tolerance for Cholesky feasibility
    maxCutCount=1e6, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on, # wheather to polish the solution to get better precision, the other option is :off
    M_Θ_factor = M_tilde, # upper bound factor for Θ
    impose_pattern = :off # wheather to impose the pattern found by solving smaller BnB-PEPs to global optimality
)


# ----------------------------------------------------
## Stage 3 of the BnB-PEP Algorithm: find the globally optimal solution to the BnB-PEP-QCQP
# ----------------------------------------------------


obj_val_glb_opt, λ_glb_opt, τ_glb_opt, η_glb_opt, ν_glb_opt, Z_glb_opt, L_cholesky_glb_opt, Θ_glb_opt, α_glb_opt, idx_set_λ_glb_opt_effective, idx_set_τ_glb_opt_effective, idx_set_η_glb_opt_effective= BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N, L, R,
    # solution to warm-start (Θ is warm-started internally)
    # -----------------------------------------------------
    d_star_ws, λ_ws, τ_ws, η_ws, ν_ws, Z_ws, L_cholesky_ws, α_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective, idx_set_η_ws_effective,
    # bounds on the variables (M_Θ is computed internally)
    # ----------------------------------------------------
    M_λ, M_τ, M_η, M_α, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type =  :find_globally_optimal, #:find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    reduce_index_set_for_dual_variables = :off, #:on,
    reduce_index_set_for_L_cholesky = :off, #:on,
    positive_step_size = :on, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    bound_impose = :on, # if this is :from_warm_start_sol, then from the warm_start solution we compute lower and upper bounds for the decision variables, [💀 : TODO] if this is :uniform_bound, then we impose a user specified bound on the variables
    quadratic_equality_modeling = :exact, #:through_ϵ, # :exact, #:through_ϵ,
    cholesky_modeling = :formula, #:definition,
    ϵ_tol_feas = 1e-4, # tolerance for Cholesky feasibility
    global_lower_bound_given = :on, # wheather is a global lower bound is given
    global_lower_bound = obj_val_glb_lbd, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on, # wheather to polish the solution to get better precision, the other option is :off
    M_Θ_factor = 5*M_tilde,
    impose_pattern = :on
    )

## Values for the tables

@show round(obj_val_glb_opt, digits = 7)

h_glb_opt = compute_h_from_α(α_glb_opt, N, L)

h_matlab_format = round.(OffsetArrays.no_offset_view(h_glb_opt),digits = 6)

h_glb_opt_latex = lap(h_matlab_format)

@show h_glb_opt_latex
