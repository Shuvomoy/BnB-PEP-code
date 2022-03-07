## Test code

## Include the main Julia file

include("BnB_PEP_reducing_function_value_AWM.jl")

## Parameters to use

L = 1
N = 2
R = 1
default_obj_val_upper_bound = 1e6

## Feasible stepsize generation

h_test = feasible_h_generator(N, L; step_size_type = :Default)

## Solve primal with feasible stepsizes

p_feas_1, G_feas_1, Ft_feas_1 = solve_primal_with_known_stepsizes(N, L, h_test, R; show_output = :on)

## -------------------------------------------------------
# Stage 1 of the BnB-PEP Algorithm: solve the dual for the warm-starting stepsize
## -------------------------------------------------------


d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_test, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)

##  Computing the bounds for computing locally optimal solution to BnB-PEP

## Using SDP relaxation based bound, comment the block out if using heuristic based bound

# M_λ, M_h, M_Z, M_L_cholesky, M_ν = bound_generator_through_SDP_relaxation(N, L, R, ν_feas_1; show_output = :off, obj_val_upper_bound = d_feas_1)


##  Using Heuristic bound, comment the block out if using bound based on SDP relaxation
# -------------------------------------------------------------------------

M_tilde = 50

# Compute M_λ

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_feas_1, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_λ,
    obj_val_upper_bound = 1.001*p_feas_1)

M_λ = M_tilde*maximum(λ_feas_1)

# Compute M_Z

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_feas_1, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_Z,
    obj_val_upper_bound = 1.001*p_feas_1)

M_Z = M_tilde*maximum(Z_feas_1[i,i] for i in 1:N+2)

# Compute M_L_cholesky

M_L_cholesky = sqrt(M_Z)

# Compute M_h

M_h = (500*M_tilde)/L

# Compute M_ν

M_ν = ν_feas_1

@show [M_λ M_h M_Z M_L_cholesky M_ν]

## Sparsify the solution for warm-starting locally optimal solver

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_feas_1, R;  show_output = :off,
    ϵ_tol_feas = 1e-8,
    objective_type = :find_sparse_sol,
    obj_val_upper_bound = 1.001*p_feas_1)

## Store the warm start point for computing locally optimal solution

d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, h_ws, idx_set_λ_ws_effective = d_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective

# ---------------------------------------------------
## Stage 2 of the BnB-PEP Algorithm: compute the locally optimal point
# ----------------------------------------------------


obj_val_loc_opt, λ_loc_opt, ν_loc_opt, Z_loc_opt, L_cholesky_loc_opt, h_loc_opt, idx_set_λ_loc_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N, L, R,
    # solution to warm-start
    # ----------------------
    d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, h_ws, idx_set_λ_ws_effective,
    # bounds on the variables
    # ----------------------
    M_λ, M_h, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type = :find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    local_solver = :knitro, #:ipopt, # other option :knitro
    knitro_multistart = :off,
    knitro_multi_algorithm = :off,
    reduce_index_set_for_λ = :for_warm_start_only,
    # options for reduce_index_set_for_λ
    # (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective, and will turn the BnB-PEP solver into a heuristic),
    # (ii) :off , this will define λ and warm-start over the full index set
    # (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
    quadratic_equality_modeling = :exact,
    cholesky_modeling = :definition,
    ϵ_tol_feas = 1e-6, # tolerance for Cholesky decomposition,
    polish_solution = :off # wheather to polish the solution to get better precision, the other option is :off
)


## Values for the tables

using LatexPrint

@show round(obj_val_loc_opt, digits = 6)

h_matlab_format = round.(OffsetArrays.no_offset_view(h_loc_opt),digits = 6)
h_latex = lap(h_matlab_format)

obj_val_loc_opt_refined,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_loc_opt, R;  show_output = :on,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)

@show h_latex

println(h_matlab_format)


## Store the solution to be warm-started for a next step

d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, h_ws, idx_set_λ_ws_effective = obj_val_loc_opt, λ_loc_opt, ν_loc_opt, Z_loc_opt, L_cholesky_loc_opt, h_loc_opt, idx_set_λ_loc_opt_effective


## Update the variable bounds

## Update the variable bounds based on SDP relaxation, comment out the block if using heuristic based variable bound

# M_λ, M_h, M_Z, M_P, M_ν = bound_generator_through_SDP_relaxation(N, L, R, ν_feas_1; show_output = :off, obj_val_upper_bound = d_star_ws)

## Update the variable bounds based on SDP relaxation, comment out the block if using SDP relaxation based bound

M_tilde = 1.01

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_ws, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_Z,
    obj_val_upper_bound = 1.001*d_star_ws)

M_Z = M_tilde*maximum(Z_feas_1[i,i] for i in 1:N+2)

d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_ws, R;  show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :find_M_λ,
    obj_val_upper_bound = 1.001*d_star_ws)

M_λ = M_tilde* maximum(λ_feas_1)

M_L_cholesky = sqrt(M_Z)

M_h = 5*M_tilde*maximum(abs.(h_loc_opt))

M_ν = ν_ws# ν_feas_1

@show [M_λ M_h M_Z M_L_cholesky M_ν]

## Compute global lower bound via lazy callback

obj_val_glb_lbd, λ_glb_lbd, ν_glb_lbd, Z_glb_lbd, L_cholesky_glb_lbd, h_glb_lbd, idx_set_λ_glb_lbd_effective = BnB_PEP_solver(
    # different parameters to be used
    # -------------------------------
    N, L, R,
    # solution to warm-start
    # ----------------------
    d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, h_ws, idx_set_λ_ws_effective,
    # bounds on the variables
    # ----------------------
    M_λ, M_h, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type =  :find_globally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    reduce_index_set_for_λ = :on, #:off, # :for_warm_start_only,
    # options for reduce_index_set_for_λ
    # (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective, and will turn the BnB-PEP solver into a heuristic),
    # (ii) :off , this will define λ and warm-start over the full index set
    # (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
    positive_step_size = :off, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :on, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :on, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
    quadratic_equality_modeling = :exact, #:through_ϵ,
    cholesky_modeling = :definition, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-6, # tolerance for cholesky feasibility
    maxCutCount=1e6, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on # wheather to polish the solution to get better precision, the other option is :off
)

## Time to compute the globally optimal solution

obj_val_glb_opt, λ_glb_opt, ν_glb_opt, Z_glb_opt, L_cholesky_glb_opt, h_glb_opt, idx_set_λ_glb_opt_effective = BnB_PEP_solver(
    # different parameters to be used
    # -------------------------------
    N, L, R,
    # solution to warm-start
    # ----------------------
    d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, h_ws, idx_set_λ_ws_effective,
    # bounds on the variables
    # ----------------------
    M_λ, M_h, M_Z, M_L_cholesky, M_ν;
    # options
    # -------
    solution_type =  :find_globally_optimal, #:find_locally_optimal, # other option :find_globally_optimal
    show_output = :on, # other option :on
    reduce_index_set_for_λ = :on, # other option :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective, and will turn the BnB-PEP solver into a heuristic),
    reduce_index_set_for_L_cholesky = :on, #other option is :on,
    bound_impose = :on, # if this is :from_warm_start_sol, then from the warm_start solution we compute lower and upper bounds for the decision variables,
    quadratic_equality_modeling = :exact, #:through_ϵ,
    cholesky_modeling = :definition, #:formula, #:definition,
    ϵ_tol_feas = 1e-6, # tolerance for cholesky feasibility
    global_lower_bound_given = :on, # wheather is a global lower bound is given
    global_lower_bound = obj_val_glb_lbd, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :off # wheather to polish the solution to get better precision, the other option is :off
    )

## Print the globally optimal stepsize

@show round(obj_val_glb_opt, digits = 6)


h_matlab_format = round.(OffsetArrays.no_offset_view(h_glb_opt),digits = 6)
h_glb_opt_latex = lap(h_matlab_format)

@show h_glb_opt_latex
