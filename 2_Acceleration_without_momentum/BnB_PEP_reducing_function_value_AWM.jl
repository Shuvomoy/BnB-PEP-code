
# using Weave
# cd("C:\\Users\\shuvo\\Google Drive\\GitHub\BnB-PEP-code-all\\1_Gradient_reduction_in_strongly_convex_smooth_problems\\") # directory that contains the .jmd file
# tangle("[Polished]_BnB-PEP-gradient-reduction-scvx-smooth.jmd", informat = "markdown")


## Load the packages:
# -------------------
using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, KNITRO

## Load the pivoted Cholesky finder
# ---------------------------------
include("code_to_compute_pivoted_cholesky.jl")


## Some helper functions

# construct e_i in R^n
function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

# this symmetric outer product is used when a is constant, b is a JuMP variable
function ⊙(a,b)
    return ((a*b') .+ transpose(a*b')) ./ 2
end

# this symmetric outer product is for computing ⊙(a,a) where a is a JuMP variable
function ⊙(a)
    return a*transpose(a)
end

# function to compute cardinality of a vector
function compute_cardinality(x, ϵ_sparsity)
    n = length(x)
    card_x = 0
    for i in 1:n
        if abs(x[i]) >=  ϵ_sparsity
            card_x = card_x + 1
        end
    end
    return card_x
end

# function to compute rank of a matrix
function compute_rank(X, ϵ_sparsity)
    eigval_array_X = eigvals(X)
    rnk_X = 0
    n = length(eigval_array_X)
    for i in 1:n
        if abs(eigval_array_X[i]) >= ϵ_sparsity
            rnk_X = rnk_X + 1
        end
    end
    return rnk_X
end


# Options for these function are
# step_size_type = :Default => will create a last step of 1/(L) rest will be zero
# step_size_type = :Random => will create a random stepsize

function feasible_h_generator(N, L; step_size_type = :Default)

    # construct h
    # -----------
    h = OffsetArray(zeros(N), 0:N-1)

    if step_size_type == :Default
        for i in 0:N-1
            h[i]= 1 # because we have defined h[i]/L in the FSFOM, h[i]=1 will correspond to a stepsize of 1/L
        end
    elseif step_size_type == :Random
        for i in 0:N-1
            h[i] = Uniform(0, 1) # same
        end
    end

    return h

end


function data_generator_function(N, L, h; input_type = :stepsize_constant)

    # define all the bold vectors
    # --------------------------

    # define 𝐱_0 and 𝐱_star

    𝐱_0 = e_i(N+2, 1)

    𝐱_star = zeros(N+2, 1)

    # define 𝐠_0, 𝐠_1, …, 𝐠_N

    # first we define the 𝐠 vectors,
    # index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

    # 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]
    𝐠 = OffsetArray(zeros(N+2, N+2), 1:N+2, -1:N)
    for i in 0:N
        𝐠[:,i] = e_i(N+2, i+2)
    end

    # time to define the 𝐟 vectors

    # 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

    𝐟 = OffsetArray(zeros(N+1, N+2), 1:N+1, -1:N)

    for i in 0:N
        𝐟[:,i] = e_i(N+1, i+1)
    end

    if input_type == :stepsize_constant

        # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}] ∈ 𝐑^(N+2 × N+2)
        𝐱 = OffsetArray(zeros(N+2, N+2), 1:N+2, -1:N)

        # assign values next using our formula for 𝐱_k
        𝐱[:,0] = 𝐱_0

        for k in 1:N
            𝐱[:,k] = 𝐱[:,0] - (1/L) .* ( sum( h[i] .* 𝐠[:,i] for i in 0:k-1) )
        end

    elseif input_type == :stepsize_variable

        𝐱 = [𝐱_star 𝐱_0]

        # assign values next using our formula for 𝐱_k
        for k in 1:N
            𝐱_k = 𝐱_0  - (1/L)*( sum( h[i] .* 𝐠[:,i] for i in 0:k-1) )
            𝐱 = [𝐱 𝐱_k]
        end

        # make 𝐱 an offset array to make our life comfortable
        𝐱 = OffsetArray(𝐱, 1:N+2, -1:N)

    end

    return 𝐱, 𝐠, 𝐟

end


# Index set creator function for the dual variables λ

struct i_j_idx # correspond to (i,j) pair, where i,j ∈ I_N_⋆
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
end

# We have dual variable λ={λ_ij}_{i,j} where i,j ∈ I_N_star
# The following function creates the maximal index set for λ

function index_set_constructor_for_dual_vars_full(I_N_star)

    # construct the index set for λ
    idx_set_λ = i_j_idx[]
    for i in I_N_star
        for j in I_N_star
            if i!=j
                push!(idx_set_λ, i_j_idx(i,j))
            end
        end
    end

    return idx_set_λ

end

# The following function will return the effective index set of a known λ i.e., those indices of  that are  λ  that are non-zero.

function effective_index_set_finder(λ ; ϵ_tol = 0.0005)

    # the variables λ are of the type DenseAxisArray whose index set can be accessed using _.axes and data via _.data syntax

    idx_set_λ_current = (λ.axes)[1]

    idx_set_λ_effective = i_j_idx[]

    # construct idx_set_λ_effective

    for i_j_λ in idx_set_λ_current
        if abs(λ[i_j_λ]) >= ϵ_tol # if λ[i,j] >= ϵ, where ϵ is our cut off for accepting nonzero
            push!(idx_set_λ_effective, i_j_λ)
        end
    end

    return idx_set_λ_effective

end

# The following function will return the zero index set of a known L_cholesky i.e., those indices of  that are  L  that are zero. 💀 Note that for λ we are doing the opposite.

function zero_index_set_finder_L_cholesky(L_cholesky; ϵ_tol = 1e-4)
    n_L_cholesky, _ = size(L_cholesky)
    zero_idx_set_L_cholesky = []
    for i in 1:n_L_cholesky
        for j in 1:n_L_cholesky
            if i >= j # because i<j has L_cholesky[i,j] == 0 for lower-triangual structure
                if abs(L_cholesky[i,j]) <= ϵ_tol
                    push!(zero_idx_set_L_cholesky, (i,j))
                end
            end
        end
    end
    return zero_idx_set_L_cholesky
end


A_mat(i,j,h,𝐠,𝐱) = ⊙(𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat(i,j,h,𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat(i,j,𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
a_vec(i,j,𝐟) = 𝐟[:, j] - 𝐟[:, i]


function solve_primal_with_known_stepsizes(N, L, h, R; show_output = :off)

    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 =  data_generator_function(N, L, h; input_type = :stepsize_constant)

    # number of points etc
    # --------------------

    I_N_star = -1:N
    dim_G = N+2
    dim_Ft = N+1


    # define the model
    # ----------------

    model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model_primal_PEP_with_known_stepsizes, G[1:dim_G, 1:dim_G], PSD)

    # construct Ft (this is just transpose of F)
    @variable(model_primal_PEP_with_known_stepsizes, Ft[1:dim_Ft])

    # define objective
    # ----------------

    @objective(model_primal_PEP_with_known_stepsizes, Max,
    Ft'*a_vec(-1,N, 𝐟)
    )

    # interpolation constraint
    # ------------------------

    for i in I_N_star, j in I_N_star
        if i != j
            @constraint(model_primal_PEP_with_known_stepsizes, Ft'*a_vec(i,j,𝐟) + tr(G*A_mat(i,j,h,𝐠,𝐱)) + ((1/(2*(L)))* tr(G*C_mat(i,j,𝐠))) <= 0
            )
        end
    end


    # initial condition
    # -----------------

    @constraint( model_primal_PEP_with_known_stepsizes, tr(G*B_mat(0,-1,h,𝐱)) <= R^2 )

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_primal_PEP_with_known_stepsizes)
    end

    optimize!(model_primal_PEP_with_known_stepsizes)

    # store and return the solution
    # -----------------------------

    if termination_status(model_primal_PEP_with_known_stepsizes) != MOI.OPTIMAL
        @warn "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_with_known_stepsizes)

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end


# In this function, the most important option is objective type:
# 0) :default will minimize ν*R^2 (this is the dual of the primal pep for a given stepsize)
# other options are
# 1) :find_sparse_sol, this will find a sparse solution given a particular stepsize and objective value upper bound
# 2) :find_M_λ , find the upper bound for the λ variables by maximizing ||λ||_1 for a given stepsize and particular objective value upper bound
# 3) :find_M_Z, find the upper bound for the entries of the slack matrix Z, by maximizing tr(Z) for for a given stepsize and particular objective value upper bound

function solve_dual_PEP_with_known_stepsizes(N, L, h, R;
    show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)

    # data generator
    # --------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, h; input_type = :stepsize_constant)

    # Number of points etc
    # --------------------

    I_N_star = -1:N
    dim_Z = N+2

    # define the model
    # ----------------

    model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # define the variables
    # --------------------

    # define the index set of λ
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)

    # define λ
    @variable(model_dual_PEP_with_known_stepsizes, λ[idx_set_λ] >= 0)

    # define ν

    @variable(model_dual_PEP_with_known_stepsizes, ν >= 0)

    # define Z ⪰ 0
    @variable(model_dual_PEP_with_known_stepsizes, Z[1:dim_Z, 1:dim_Z], PSD)


    if objective_type == :default

        @info "[🐒 ] Minimizing the usual performance measure"

        @objective(model_dual_PEP_with_known_stepsizes, Min,  ν*R^2)

    elseif objective_type == :find_sparse_sol

        @info "[🐮 ] Finding a sparse dual solution given the objective value upper bound"

        @objective(model_dual_PEP_with_known_stepsizes, Min, sum(λ[i_j] for i_j in idx_set_λ))

        @constraint(model_dual_PEP_with_known_stepsizes, ν*R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_λ

        @info "[🐷 ] Finding upper bound on the entries of λ for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(λ[i_j] for i_j in idx_set_λ))

        @constraint(model_dual_PEP_with_known_stepsizes, ν*R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_Z

        @info "[🐯 ] Finding upper bound on the entries of Z for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, tr(Z))

        @constraint(model_dual_PEP_with_known_stepsizes, ν*R^2 <= obj_val_upper_bound)

    end

    # add the linear constraint
    # -------------------------

    # the constraint is: ∑ λ[i,j] a[i,j] = 0
    # note that in the code i_j_λ = (i,j), i_j_λ.i = i, i_j_λ.j = j
    @constraint(model_dual_PEP_with_known_stepsizes,   sum(λ[i_j_λ]*a_vec(i_j_λ.i,i_j_λ.j,𝐟) for i_j_λ in idx_set_λ) - a_vec(-1,N,𝐟) .== 0)

    # add the LMI constraint
    # ----------------------

    @constraint(model_dual_PEP_with_known_stepsizes,
    ν*B_mat(0,-1,h,𝐱)  + sum(λ[i_j_λ]*A_mat(i_j_λ.i,i_j_λ.j,h,𝐠,𝐱) for i_j_λ in idx_set_λ) +
    (1/(2*(L)))*sum(λ[i_j_λ]*C_mat(i_j_λ.i,i_j_λ.j,𝐠) for i_j_λ in idx_set_λ)
    .==
    Z
    )

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_dual_PEP_with_known_stepsizes)
    end

    optimize!(model_dual_PEP_with_known_stepsizes)

    if termination_status(model_dual_PEP_with_known_stepsizes) != MOI.OPTIMAL
		 @info "💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀"
        @error "model_dual_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_dual_PEP_with_known_stepsizes)
    end

    # store the solutions and return
    # ------------------------------

    # store λ_opt

    λ_opt = value.(λ)

    # store ν_opt

    ν_opt = value.(ν)

    # store Z_opt

    Z_opt = value.(Z)

    # compute cholesky

    L_cholesky_opt =  compute_pivoted_cholesky_L_mat(Z_opt)

    if norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 1e-6
        @info "checking the norm bound"
        @warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf))"
    end

    # effective index sets for the dual variables λ, μ, ν

    idx_set_λ_effective = effective_index_set_finder(λ_opt ; ϵ_tol = 0.0005)

    # store objective

    ℓ_1_norm_λ = sum(λ_opt)
    tr_Z = tr(Z_opt)
    original_performance_measure = ν_opt*R^2

    # return all the stored values

    return original_performance_measure, ℓ_1_norm_λ, tr_Z, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h, idx_set_λ_effective

end


# We also provide a function to check if in a particular feasible solution, these bounds are violated

function bound_violation_checker_BnB_PEP(
    # input point
    # -----------
    d_star_sol, λ_sol, ν_sol, Z_sol, L_cholesky_sol, h_sol,
    # input bounds
    # ------------
    λ_lb, λ_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, h_lb, h_ub;
    # options
    # -------
    show_output = :on,
    computing_global_lower_bound = :off
    )

    if show_output == :on
        @show [minimum(λ_sol)  maximum(λ_sol)  λ_ub]
        @show [ν_lb ν_sol ν_ub]
        @show [Z_lb minimum(Z_sol)   maximum(Z_sol)  Z_ub]
        @show [L_cholesky_lb  minimum(L_cholesky_sol)  maximum(L_cholesky_sol) L_cholesky_ub]
        @show [h_lb minimum(h_sol) maximum(h_sol) h_ub]
    end

    # bound satisfaction flag

    bound_satisfaction_flag = 1

    # verify bound for λ
    if !(maximum(λ_sol) < λ_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found λ is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for ν: this is not necessary because this will be ensured due to our objective function being ν R^2
    if !(maximum(ν_sol) <= ν_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found ν is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for Z
    if !(Z_lb -  1e-8 < minimum(Z_sol) && maximum(Z_sol) < Z_ub + 1e-8)
        @error "found Z is violating the input bound"
        bound_satisfaction_flag = 0
    end

    if computing_global_lower_bound == :off
        # verify bound for L_cholesky
        if !(L_cholesky_lb -  1e-8 < minimum(L_cholesky_sol) && maximum(L_cholesky_sol) < L_cholesky_ub +  1e-8)
            @error "found L_cholesky is violating the input bound"
            bound_satisfaction_flag = 0
        end
    elseif computing_global_lower_bound == :on
        @info "no need to check bound on L_cholesky"
    end

    # # verify bound for objective value
    # if abs(obj_val_sol-BnB_PEP_cost_lb) <= ϵ_tol_sol
    #     @error "found objective value is violating the input bound"
    #     bound_satisfaction_flag = 0
    # end

    if bound_satisfaction_flag == 0
        @error "[💀 ] some bound is violated, increase the bound intervals "
    elseif bound_satisfaction_flag == 1
        @info "[😅 ] all bounds are satisfied by the input point, rejoice"
    end

    return bound_satisfaction_flag

end



function BnB_PEP_solver(
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
    show_output = :off, # other option :on
    local_solver = :ipopt, # other option :knitro
    knitro_multistart = :off, # other option :on (only if :knitro solver is used)
    knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
    reduce_index_set_for_λ = :for_warm_start_only,
    # options for reduce_index_set_for_λ
    # (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective),
    # (ii) :off , this will define λ and warm-start over the full index set
    # (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
    reduce_index_set_for_L_cholesky = :off, # the other option is :on
    positive_step_size = :off, # other option is :on (i.e., making it :on will enforce the stepsize to be non-negative, which will turn BnB-PEP solver into a heuristic), 💀 turning it :on is not recommended
    find_global_lower_bound_via_cholesky_lazy_constraint = :off, # if this on, then we model Z = L_cholesky*L_cholesky^T via lazy constraint (the goal is to find a lower bound to BnB PEP)
    bound_impose = :off, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation,
    quadratic_equality_modeling = :through_ϵ, # other option is :exact
    #  quadratic_equality_modeling == :exact models a nonconvex quadratic constraint x^T P x + q^T x + r == 0 exactly in JuMP
    #  quadratic_equality_modeling == : :through_ϵ models the constraint x^T P x + q^T x + r == 0 as two constraints:
    # x^T P x + q^T x + r <= ϵ_tol_feas, and
    #  x^T P x + q^T x + r >= -ϵ_tol_feas,
    # where ϵ_tol_feas is our tolerance for feasibility. This is recommended while solving using Gurobi
    cholesky_modeling = :formula, # : formula impelements the equivalent representation of Z = L_cholesky*L_cholesky^T via formulas, the other option is :definition, that directly model Z = L_cholesky*L_cholesky^T
    ϵ_tol_feas = 1e-6, # tolerance for feasibility
    ϵ_tol_Cholesky = 0.0005, # tolerance for determining which elements of L_cholesky_ws is zero
    maxCutCount=1e3, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on, # wheather to polish the solution to get better precision, the other option is :off
    )

    # Number of points
    # ----------------

    I_N_star = -1:N
    dim_Z = N+2

    # *************
    # declare model
    # -------------
    # *************

    if solution_type == :find_globally_optimal

        @info "[🐌 ] globally optimal solution finder activated, solution method: spatial branch and bound"

        BnB_PEP_model =  Model(Gurobi.Optimizer)
        # using direct_model results in smaller memory allocation
        # we could also use
        # Model(Gurobi.Optimizer)
        # but this requires more memory

        set_optimizer_attribute(BnB_PEP_model, "NonConvex", 2)
        # "NonConvex" => 2 tells Gurobi to use its nonconvex algorithm

        set_optimizer_attribute(BnB_PEP_model, "MIPFocus", 3)
        # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more
        # attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

        # 🐑: other Gurobi options one can play with
        # ------------------------------------------

        # turn off all the heuristics
        # set_optimizer_attribute(BnB_PEP_model, "Heuristics", 0)
        # set_optimizer_attribute(BnB_PEP_model, "RINS", 0)

        # other termination epsilons for Gurobi
        # set_optimizer_attribute(BnB_PEP_model, "MIPGapAbs", 1e-4)

        set_optimizer_attribute(BnB_PEP_model, "MIPGap", 1e-2) # 99% optimal solution, because Gurobi will provide a result associated with a global lower bound within this tolerance, by polishing the result, we can find the exact optimal solution by solving a convex SDP

        # set_optimizer_attribute(BnB_PEP_model, "FuncPieceRatio", 0) # setting "FuncPieceRatio" to 0, will ensure that the piecewise linear approximation of the nonconvex constraints lies below the original function

        # set_optimizer_attribute(BnB_PEP_model, "Threads", 64) # how many threads to use at maximum
        #
        set_optimizer_attribute(BnB_PEP_model, "FeasibilityTol", 1e-4)
        #
        # set_optimizer_attribute(BnB_PEP_model, "OptimalityTol", 1e-4)

    elseif solution_type == :find_locally_optimal

        @info "[🐙 ] locally optimal solution finder activated, solution method: interior point method"

        if local_solver == :knitro

            @info "[🚀 ] activating KNITRO"

            # BnB_PEP_model = Model(optimizer_with_attributes(KNITRO.Optimizer, "convex" => 0,  "strat_warm_start" => 1))

            BnB_PEP_model = Model(
                optimizer_with_attributes(
                KNITRO.Optimizer,
                "convex" => 0,
                "strat_warm_start" => 1,
                # the last settings below are for larger N
                # you can comment them out if preferred but not recommended
                "honorbnds" => 1,
                # "bar_feasmodetol" => 1e-3,
                #"feastol" => 1e-4,
                #"infeastol" => 1e-12,
                #"opttol" => 1e-4
                )
            )

            if knitro_multistart == :on
                set_optimizer_attribute(BnB_PEP_model, "ms_enable", 1)
                set_optimizer_attribute(BnB_PEP_model, "par_numthreads", 8)
                set_optimizer_attribute(BnB_PEP_model, "par_msnumthreads", 8)
                set_optimizer_attribute(BnB_PEP_model, "ms_maxsolves", 8)
            end

            if knitro_multi_algorithm == :on
                set_optimizer_attribute(BnB_PEP_model, "algorithm", 5)
                set_optimizer_attribute(BnB_PEP_model, "ma_terminate", 0)
            end

        elseif local_solver == :ipopt

            @info "[🎃 ] activating IPOPT"

            BnB_PEP_model = Model(Ipopt.Optimizer)

            # very high precision solution in IPOPT can be obtained via the following code, comment them out if high precision is not required

            set_optimizer_attribute(BnB_PEP_model, "constr_viol_tol", 1e-6)

            set_optimizer_attribute(BnB_PEP_model, "dual_inf_tol", 1e-6)

            set_optimizer_attribute(BnB_PEP_model, "compl_inf_tol", 1e-6)

            set_optimizer_attribute(BnB_PEP_model, "tol", 1e-10)

            set_optimizer_attribute(BnB_PEP_model, "max_iter", 5000)

        end
    end

    # ************************
    # define all the variables
    # ------------------------
    # ************************

    @info "[🎉 ] defining the variables"

    # define λ
    # --------

    if reduce_index_set_for_λ == :off
        # define λ over the full index set
        idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
    elseif reduce_index_set_for_λ == :on
        # define λ over a reduced index set, idx_set_λ_ws_effective, which is the effective index set of λ_ws
        idx_set_λ = idx_set_λ_ws_effective
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
    elseif reduce_index_set_for_λ == :for_warm_start_only
        # this :for_warm_start_only option is same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
        idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)
        idx_set_λ_ws = idx_set_λ_ws_effective
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
    end

    # define ν
    # --------

    @variable(BnB_PEP_model, ν >= 0)

    # define Z
    # --------

    @variable(BnB_PEP_model, Z[1:dim_Z, 1:dim_Z], Symmetric)

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off

        # define the cholesky matrix of Z: L_cholesky
        # -------------------------------------------
        @variable(BnB_PEP_model, L_cholesky[1:dim_Z, 1:dim_Z])

    end

    # define the stepsize matrix h
    # ----------------------------
    if positive_step_size == :off
        @variable(BnB_PEP_model,  h[0:N-1])
    elseif positive_step_size == :on
        @variable(BnB_PEP_model, h[0:N-1] >= 0)
    end

    # [👲 ] insert warm-start values for all the variables
    # ----------------------------------------------------

    @info "[👲 ] warm-start values for all the variables"

    # warm start for λ
    # ----------------
    if reduce_index_set_for_λ == :for_warm_start_only
        for i_j_λ in idx_set_λ_ws
            set_start_value(λ[i_j_λ], λ_ws[i_j_λ])
        end
        for i_j_λ in setdiff(idx_set_λ, idx_set_λ_ws)
            set_start_value(λ[i_j_λ], 0.0)
        end
    else
        for i_j_λ in idx_set_λ
            set_start_value(λ[i_j_λ], λ_ws[i_j_λ])
        end
    end

    # warm start for ν
    # ----------------

    set_start_value(ν, ν_ws)

    # warm start for Z
    # ----------------

    for i in 1:dim_Z
        for j in 1:dim_Z
            set_start_value(Z[i,j], Z_ws[i,j])
        end
    end

    # warm start for L_cholesky
    # ------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off
        for i in 1:dim_Z
            for j in 1:dim_Z
                set_start_value(L_cholesky[i,j], L_cholesky_ws[i,j])
            end
        end
    end

    # warm start for h
    # ----------------

    for i in 0:N-1
            set_start_value(h[i], h_ws[i])
    end


    # ************
    # [🎇 ] add objective
    # -------------
    # *************

    @info "[🎇 ] adding objective"

    @objective(BnB_PEP_model, Min, ν*R^2)

    # Adding an upper bound for the objective function

    @constraint(BnB_PEP_model,  ν*R^2 <= 1.001*d_star_ws) # this 1.001 factor gives some slack

    # Adding a lower bound for the objective function (if given)
    if global_lower_bound_given == :on
        @constraint(BnB_PEP_model,  ν*R^2 >= global_lower_bound)
    end

    # ******************************
    # [🎍 ] add the data generator function
    # *******************************

    @info "[🎍 ] adding the data generator function to create 𝐱, 𝐠, 𝐟"

    dim_𝐱 = N+2

    𝐱_0 = e_i(dim_𝐱, 1)

    𝐱_star = zeros(dim_𝐱, 1)

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, h; input_type = :stepsize_variable)

    B_mat_0_minus1 = ⊙(𝐱_0-𝐱_star, 𝐱_0-𝐱_star)

    # *******************
    # add the constraints
    # *******************


    # add the linear constraint
    # -------------------------

    @info "[🎋 ] adding linear constraint"

    # the constraint is: ∑ λ[i,j] a[i,j] = 0
    # note that in the code i_j_λ = (i,j), i_j_λ.i = i, i_j_λ.j = j
    @constraint(BnB_PEP_model, sum(λ[i_j_λ]*a_vec(i_j_λ.i,i_j_λ.j,𝐟) for i_j_λ in idx_set_λ) - a_vec(-1,N,𝐟) .== 0)

    # add the LMI constraint
    # ----------------------

    @info "[🎢 ] adding LMI constraint"

    if quadratic_equality_modeling == :exact


        # modeling of the LMI constraint through vectorization (works same)
        # ------------------------------------
        @constraint(BnB_PEP_model,
        vectorize(
            ν*B_mat_0_minus1 + sum(λ[i_j_λ]*A_mat(i_j_λ.i,i_j_λ.j,h,𝐠,𝐱) for i_j_λ in idx_set_λ) +
    (1/(2*(L)))*sum(λ[i_j_λ]*C_mat(i_j_λ.i,i_j_λ.j,𝐠) for i_j_λ in idx_set_λ)
    - Z,
        SymmetricMatrixShape(dim_Z)
        ) .== 0
        )

    elseif quadratic_equality_modeling == :through_ϵ

        # modeling of the LMI constraint through vectorization and ϵ_tol_feas
        # ---------------------------------------

        # part 1: models
        # (dual related terms) - Z <= ϵ_tol_feas*ones(dim_Z,dim_z)
        @constraint(BnB_PEP_model,
        vectorize(
        ν*B_mat_0_minus1  + sum(λ[i_j_λ]*A_mat(i_j_λ.i,i_j_λ.j,h,𝐠,𝐱) for i_j_λ in idx_set_λ) +
    (1/(2*(L)))*sum(λ[i_j_λ]*C_mat(i_j_λ.i,i_j_λ.j,𝐠) for i_j_λ in idx_set_λ)
    - Z - ϵ_tol_feas*ones(dim_Z,dim_Z),
        SymmetricMatrixShape(dim_Z)
        ) .<= 0
        )

        # part 2: models
        # (dual related terms) - Z >= -ϵ_tol_feas*ones(dim_Z,dim_z)
        @constraint(BnB_PEP_model,
        vectorize(
        ν*B_mat_0_minus1  + sum(λ[i_j_λ]*A_mat(i_j_λ.i,i_j_λ.j,h,𝐠,𝐱) for i_j_λ in idx_set_λ) +
    (1/(2*(L)))*sum(λ[i_j_λ]*C_mat(i_j_λ.i,i_j_λ.j,𝐠) for i_j_λ in idx_set_λ)
    - Z + ϵ_tol_feas*ones(dim_Z,dim_Z),
        SymmetricMatrixShape(dim_Z)
        ) .>= 0
        )

    else

        @error "something is not right in LMI modeling"

        return

    end


    # implementation through ϵ_tol_feas

    # add valid constraints for Z ⪰ 0
    # -------------------------------

    @info "[🎩 ] adding valid constraints for Z"

    # diagonal components of Z are non-negative
    for i in 1:dim_Z
        @constraint(BnB_PEP_model, Z[i,i] >= 0)
    end

    # the off-diagonal components satisfy:
    # (∀i,j ∈ dim_Z: i != j) -(0.5*(Z[i,i] + Z[j,j])) <= Z[i,j] <=  (0.5*(Z[i,i] + Z[j,j]))

    for i in 1:dim_Z
        for j in 1:dim_Z
            if i != j
                @constraint(BnB_PEP_model, Z[i,j] <= (0.5*(Z[i,i] + Z[j,j])) )
                @constraint(BnB_PEP_model, -(0.5*(Z[i,i] + Z[j,j])) <= Z[i,j] )
            end
        end
    end

    # add cholesky related constraints
    # --------------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off

        @info "[🎭 ] adding cholesky matrix related constraints"

        # Two constraints to define the matrix L_cholesky to be a lower triangular matrix
        # -------------------------------------------------

        # upper off-diagonal terms of L_cholesky are zero

        for i in 1:dim_Z
            for j in 1:dim_Z
                if i < j
                    # @constraint(BnB_PEP_model, L_cholesky[i,j] .== 0)
                    fix(L_cholesky[i,j], 0; force = true)
                end
            end
        end

        # diagonal components of L_cholesky are non-negative

        for i in 1:dim_Z
            @constraint(BnB_PEP_model, L_cholesky[i,i] >= 0)
        end

    end

    # time to implement Z = L*L^T constraint
    # --------------------------------------

    if cholesky_modeling == :definition && find_global_lower_bound_via_cholesky_lazy_constraint == :off

        if quadratic_equality_modeling == :exact

            # direct modeling through definition and vectorization
            # ---------------------------------------------------
            @constraint(BnB_PEP_model, vectorize(Z - (L_cholesky * L_cholesky'), SymmetricMatrixShape(dim_Z)) .== 0)

        elseif quadratic_equality_modeling == :through_ϵ

            # definition modeling through vectorization and ϵ_tol_feas

            # part 1: models Z-L_cholesky*L_cholesky <= ϵ_tol_feas*ones(dim_Z,dim_Z)
            @constraint(BnB_PEP_model, vectorize(Z - (L_cholesky * L_cholesky') - ϵ_tol_feas*ones(dim_Z,dim_Z), SymmetricMatrixShape(dim_Z)) .<= 0)

            # part 2: models Z-L_cholesky*L_cholesky >= -ϵ_tol_feas*ones(dim_Z,dim_Z)

            @constraint(BnB_PEP_model, vectorize(Z - (L_cholesky * L_cholesky') + ϵ_tol_feas*ones(dim_Z,dim_Z), SymmetricMatrixShape(dim_Z)) .>= 0)

        else

            @error "something is not right in Cholesky modeling"

            return

        end


    elseif cholesky_modeling == :formula && find_global_lower_bound_via_cholesky_lazy_constraint == :off

        # Cholesky constraint 1
        # (∀ j ∈ dim_Z) L_cholesky[j,j]^2 + ∑_{k∈[1:j-1]} L_cholesky[j,k]^2 == Z[j,j]

        for j in 1:dim_Z
            if j == 1
                @constraint(BnB_PEP_model, L_cholesky[j,j]^2 == Z[j,j])
            elseif j > 1
                @constraint(BnB_PEP_model, L_cholesky[j,j]^2+sum(L_cholesky[j,k]^2 for k in 1:j-1) == Z[j,j])
            end
        end

        # Cholesky constraint 2
        # (∀ i,j ∈ dim_Z: i > j) L_cholesky[i,j] L_cholesky[j,j] + ∑_{k∈[1:j-1]} L_cholesky[i,k] L_cholesky[j,k] == Z[i,j]

        for i in 1:dim_Z
            for j in 1:dim_Z
                if i>j
                    if j == 1
                        @constraint(BnB_PEP_model, L_cholesky[i,j]*L_cholesky[j,j]  == Z[i,j])
                    elseif j > 1
                        @constraint(BnB_PEP_model, L_cholesky[i,j]*L_cholesky[j,j] + sum(L_cholesky[i,k]*L_cholesky[j,k] for k in 1:j-1) == Z[i,j])
                    end
                end
            end
        end

    elseif find_global_lower_bound_via_cholesky_lazy_constraint == :on

        # set_optimizer_attribute(BnB_PEP_model, "FuncPieces", -2) # FuncPieces = -2: Bounds the relative error of the approximation; the error bound is provided in the FuncPieceError attribute. See https://www.gurobi.com/documentation/9.1/refman/funcpieces.html#attr:FuncPieces

        # set_optimizer_attribute(BnB_PEP_model, "FuncPieceError", 0.1) # relative error

        set_optimizer_attribute(BnB_PEP_model, "MIPFocus", 1) # focus on finding good quality feasible solution

        # add initial cuts
        num_cutting_planes_init = 2*dim_Z^2
        cutting_plane_array = randn(dim_Z,num_cutting_planes_init)
        num_cuts_array_rows, num_cuts = size(cutting_plane_array)
        for i in 1:num_cuts
            d_cut = cutting_plane_array[:,i]
            d_cut = d_cut/norm(d_cut,2) # normalize the cutting plane vector
            @constraint(BnB_PEP_model, tr(Z*(d_cut*d_cut')) >= 0)
        end

        cutCount=0
        # maxCutCount=1e3

        # add the lazy callback function
        # ------------------------------
        function add_lazy_callback(cb_data)
            if cutCount<=maxCutCount
                Z0 = zeros(dim_Z,dim_Z)
                for i=1:dim_Z
                    for j=1:dim_Z
                        Z0[i,j]=callback_value(cb_data, Z[i,j])
                    end
                end
                if eigvals(Z0)[1]<=-0.01
                    u_t = eigvecs(Z0)[:,1]
                    u_t = u_t/norm(u_t,2)
                    con3 = @build_constraint(tr(Z*u_t*u_t') >=0.0)
                    MOI.submit(BnB_PEP_model, MOI.LazyConstraint(cb_data), con3)
                    # noPSDCuts+=1
                end
                cutCount+=1
            end
        end

        # submit the lazy constraint
        # --------------------------
        MOI.set(BnB_PEP_model, MOI.LazyConstraintCallback(), add_lazy_callback)


    end

    # impose bound on the variables if bound_impose == :on

    if bound_impose == :on
        @info "[🌃 ] finding bound on the variables"

        # store the values

        λ_lb = 0
        λ_ub = M_λ
        ν_lb = 0
        ν_ub = ν_ws
        Z_lb = -M_Z
        Z_ub = M_Z
        L_cholesky_lb = -M_L_cholesky
        L_cholesky_ub = M_L_cholesky
        h_lb = -M_h
        h_ub = M_h

        # set bound for λ
        # ---------------
        # set_lower_bound.(λ, λ_lb): done in definition
        set_upper_bound.(λ, λ_ub)

        # set bound for ν
        # ---------------
        # set_lower_bound.(ν, ν_lb): done in definition
        set_upper_bound(ν, ν_ub)

        # set bound for Z
        # ---------------
        for i in 1:dim_Z
            for j in 1:dim_Z
                set_lower_bound(Z[i,j], Z_lb)
                set_upper_bound(Z[i,j], Z_ub)
            end
        end

        # set bound for L_cholesky
        # ------------------------

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off
            # need only upper bound for the diagonal compoments, as the lower bound is zero from the model
            for i in 1:N+2
                set_upper_bound(L_cholesky[i,i], L_cholesky_ub)
            end
            # need to bound only components, L_cholesky[i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
            for i in 1:N+2
                for j in 1:N+2
                    if i > j
                        set_lower_bound(L_cholesky[i,j], L_cholesky_lb)
                        set_upper_bound(L_cholesky[i,j], L_cholesky_ub)
                    end
                end
            end
        end

        # set bound for h
        # ---------------
        set_lower_bound.(h, h_lb)
        set_upper_bound.(h, h_ub)

    end

    # impose the effective index set of L_cholesky if reduce_index_set_for_L_cholesky  == :on and we are not computing a global lower bound
    # ------------------------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off && reduce_index_set_for_L_cholesky == :on
        zis_Lc = zero_index_set_finder_L_cholesky(L_cholesky_ws; ϵ_tol = ϵ_tol_Cholesky)
        for k in 1:length(zis_Lc)
            fix(L_cholesky[CartesianIndex(zis_Lc[k])], 0; force = true)
        end
    end


    # time to optimize
    # ----------------

    @info "[🙌 	🙏 ] model building done, starting the optimization process"

    if show_output == :off
        set_silent(BnB_PEP_model)
    end

    optimize!(BnB_PEP_model)

    @info "BnB_PEP_model has termination status = " termination_status(BnB_PEP_model)

    if (solution_type == :find_locally_optimal && termination_status(BnB_PEP_model) == MOI.LOCALLY_SOLVED) || (solution_type ==:find_globally_optimal && termination_status(BnB_PEP_model) == MOI.OPTIMAL )

        # store the solutions and return
        # ------------------------------

        @info "[😻 ] optimal solution found done, store the solution"

        # store λ_opt

        λ_opt = value.(λ)

        # store ν_opt

        ν_opt = value.(ν)

        # store h_opt

        h_opt = value.(h)

        # store Z_opt

        Z_opt = value.(Z)

        # store L_cholesky

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off

            L_cholesky_opt = value.(L_cholesky)

            if norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 10^-4
                @warn "||Z - L_cholesky*L_cholesky^T|| = $(norm(Z_opt -  L_cholesky_opt*L_cholesky_opt', Inf))"
            end

        elseif find_global_lower_bound_via_cholesky_lazy_constraint == :on

            L_cholesky_opt = compute_pivoted_cholesky_L_mat(Z_opt)

            # in this case doing the cholesky check does not make sense, because we are not aiming to find a psd Z_opt

            # if norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 10^-4
            #     @info "checking the norm bound"
            #     @warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf))"
            # end

        end

        obj_val = objective_value(BnB_PEP_model)

    else

        @warn "[🙀 ] could not find an optimal solution, returning the warm-start point"

        obj_val, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt, idx_set_λ_opt_effective = d_star_ws, λ_ws, ν_ws, Z_ws, L_cholesky_ws, h_ws, idx_set_λ_ws_effective


    end

    if polish_solution == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :off # note that if we are finding a global lower bound, then polishing the solution would not make sense

        @info "[🎣 ] polishing and sparsifying the solution"

        obj_val,  _, _, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt, _ = solve_dual_PEP_with_known_stepsizes(N, L, h_opt, R;  show_output = :off,
        ϵ_tol_feas = 1e-6, objective_type = :default, obj_val_upper_bound = 1.0001*obj_val)

        obj_val_sparse,  _, _, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt, _ = solve_dual_PEP_with_known_stepsizes(N, L, h_opt, R;  show_output = :off,
        ϵ_tol_feas = 1e-6, objective_type = :find_sparse_sol, obj_val_upper_bound = (1+(1e-6))*obj_val)

    end

    # find the effective index set of the found λ

    idx_set_λ_opt_effective = effective_index_set_finder(λ_opt ; ϵ_tol = 0.0005)

    @info "[🚧 ] for λ, only $(length(idx_set_λ_opt_effective)) components out of $(length(idx_set_λ)) are non-zero for the optimal solution"


    @info "[💹 ] warm-start objective value = $d_star_ws, and objective value of found solution = $obj_val"

    # verify if any of the imposed bounds are violated

    if bound_impose == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :off
        bound_satisfaction_flag = bound_violation_checker_BnB_PEP(obj_val, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt, λ_lb, λ_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, h_lb, h_ub; show_output = :on,     computing_global_lower_bound = :off)
    elseif bound_impose == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :on
        bound_satisfaction_flag = bound_violation_checker_BnB_PEP(obj_val, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt, λ_lb, λ_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, h_lb, h_ub; show_output = :on,     computing_global_lower_bound = :on)
    end

    # time to return all the stored values

    return obj_val, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt, idx_set_λ_opt_effective

end


# the following function will compute w = vec(h, ν, λ) and provide index selectors from math to vec and vec to math.
function vectorize_h_ν_λ(h, ν, λ, idx_set_λ)

    k = 0

    vec_all_var = Vector{VariableRef}() # this is the vectorized version of all variables

    # vectorize h

    index_math2vec = OrderedDict()

    for i in 0:N-1
          k = k+1
          vec_all_var= [vec_all_var; h[i]]
          index_math2vec[(:h, i)] = k
    end

    # vectorize ν

    k = k+1
    vec_all_var= [vec_all_var; ν]
    index_math2vec[(:ν, 1)] = k

    # vectorize λ

    for i_j in idx_set_λ
        k = k+1
        vec_all_var = [vec_all_var; λ[i_j]]
        index_math2vec[(:λ, (i_j.i, i_j.j))] = k
    end

    # reverse the dictionary index_math2vec
    index_vec2math = OrderedDict(value => key for (key, value) in index_math2vec)

    return vec_all_var, index_math2vec, index_vec2math

end

# usage:
# w, index_math2vec, index_vec2math = vectorize_h_ν_λ(h, ν, λ, idx_set_λ)
# such that
# w[index_math2vec[(:h,i)]]=w[k]=h[i]
# h[index_vec2math[k]] = w[k]
# and so on


## Write the  𝔍_i matrix creator

function 𝔍_mat(i, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    if i == -1
        dim_𝐱 = length(𝐱_0)
        𝔍_i = zeros(dim_𝐱, len_w+dim_𝐱)
        return 𝔍_i
    elseif i == 0
        dim_𝐱 = length(𝐱_0)
        𝔍_i = zeros(dim_𝐱, len_w+dim_𝐱)
        𝔍_i[1:dim_𝐱,1:dim_𝐱] = I(dim_𝐱)
        return 𝔍_i
    elseif i >= 1 && i <= N
        dim_𝐱 = length(𝐱_0)
        𝔍_i = zeros(dim_𝐱, len_w+dim_𝐱)
        𝔍_i_part_2 = zeros(dim_𝐱, len_w)
        𝔍_i[1:dim_𝐱,1:dim_𝐱] = I(dim_𝐱)
        for j in 0:i-1
            term = (-1/L)*( ( 𝐠[:,j] ) * transpose(e_i(len_w, index_math2vec[(:h, j)])) )
            𝔍_i_part_2 = 𝔍_i_part_2 + term
        end
        𝔍_i[1:dim_𝐱,dim_𝐱+1:len_w+dim_𝐱] = 𝔍_i_part_2
        return 𝔍_i
    end
end

# usage:
# test = 𝔍_mat(2,  𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)


## Write the 𝔊_i_j and ℌ_i_j matrix creator

function 𝔊_ℌ_mat(i, j, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    dim_𝐱 = length(𝐱_0)
    𝔍_i = 𝔍_mat(i, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    𝔍_j =  𝔍_mat(j, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    𝔊_i_j_ℌ_i_j = 𝔍_i - 𝔍_j
    𝔊_i_j = 𝔊_i_j_ℌ_i_j[1:dim_𝐱, 1:dim_𝐱]
    ℌ_i_j = 𝔊_i_j_ℌ_i_j[1:dim_𝐱, dim_𝐱+1:len_w+dim_𝐱]
    return 𝔊_i_j, ℌ_i_j
end

# usage:
# 𝔊_1_2, ℌ_1_2 = 𝔊_ℌ_mat(1, 2, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)


## Write the ℍ_i_j_k_ℓ matrix creator

function constituents_of_B_h(i, j, k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    𝔊_i_j, ℌ_i_j = 𝔊_ℌ_mat(i, j, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    # k-th row of 𝔊_i_j, ℌ_i_j
    𝔤_ij_k = 𝔊_i_j[k, :]
    𝔥_ij_k = ℌ_i_j[k,:]
    # ℓ-th row of 𝔊_i_j, ℌ_i_j
    𝔤_ij_ℓ = 𝔊_i_j[ℓ,:]
    𝔥_ij_ℓ = ℌ_i_j[ℓ,:]
    # define c_ij_k, c_ij_ℓ
    c_ij_k = (𝔤_ij_k'*𝐱_0)[1]
    c_ij_ℓ =  (𝔤_ij_ℓ'*𝐱_0)[1]
    ℍ_ij_kℓ = zeros(len_w, len_w)
    for i_tilde in 1:len_w
        for j_tilde in 1:len_w
            ℍ_ij_kℓ[i_tilde, j_tilde] = 0.5*((𝔥_ij_k[i_tilde]*𝔥_ij_ℓ[j_tilde])+(𝔥_ij_k[j_tilde]*𝔥_ij_ℓ[i_tilde]))
        end
    end
    return c_ij_k, c_ij_ℓ, 𝔥_ij_k, 𝔥_ij_ℓ, ℍ_ij_kℓ
end

# # usage:
# i = 1
# j = -1
# k = 1
# ℓ = 2
#
# c_ij_k, c_ij_ℓ, 𝔥_ij_k, 𝔥_ij_ℓ, ℍ_ij_kℓ = constituents_of_B_h(i, j, k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
#
# c_Nstar_k, c_Nstar_ℓ, 𝔥_Nstar_k, 𝔥_Nstar_ℓ, ℍ_Nstar_kℓ = constituents_of_B_h(N, -1,  k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
#
# B_N_star_h_k_ℓ = (c_Nstar_k*c_Nstar_ℓ) + (c_Nstar_k*𝔥_Nstar_ℓ'*w) + (c_Nstar_ℓ*𝔥_Nstar_k'*w) + tr(ℍ_Nstar_kℓ * W)


## B_N_star_h_mat

function B_N_star_h_mat(k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, w, W, index_math2vec)
    c_Nstar_k, c_Nstar_ℓ, 𝔥_Nstar_k, 𝔥_Nstar_ℓ, ℍ_Nstar_kℓ = constituents_of_B_h(N, -1,  k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    B_N_star_h_k_ℓ = (c_Nstar_k*c_Nstar_ℓ) + (c_Nstar_k*𝔥_Nstar_ℓ'*w) + (c_Nstar_ℓ*𝔥_Nstar_k'*w) + tr(ℍ_Nstar_kℓ * W)
    return B_N_star_h_k_ℓ
end

# usage
# B_N_star_h_mat(k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, w, W, index_math2vec)


## Constructs the elements of λ_A_h

function constituents_of_λ_A_h(i, j, k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)

    𝔊_i_j, ℌ_i_j = 𝔊_ℌ_mat(i, j, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
    # k-th row of 𝔊_i_j, ℌ_i_j
    𝔤_ij_k = 𝔊_i_j[k,:]
    𝔥_ij_k = ℌ_i_j[k,:]
    # ℓ-th row of 𝔊_i_j, ℌ_i_j
    𝔤_ij_ℓ = 𝔊_i_j[ℓ,:]
    𝔥_ij_ℓ = ℌ_i_j[ℓ,:]
    # define c_ij_k, c_ij_ℓ
    c_ij_k = (𝔤_ij_k'*𝐱_0)[1]
    c_ij_ℓ =  (𝔤_ij_ℓ'*𝐱_0)[1]

    c_tilde_ij_kℓ = 0.5*( ( c_ij_ℓ* (𝐠[:,j])[k] ) + ( c_ij_k* (𝐠[:,j])[ℓ] ) )

    d_ij = e_i(len_w, index_math2vec[(:λ, (i, j))])

    𝕊_ij_kℓ = zeros(len_w, len_w)

    for i_tilde in 1:len_w
        for j_tilde in 1:len_w
            q_ij_kℓ_itilde = 0.5 *  ( ((𝐠[:,j])[k]* 𝔥_ij_ℓ[i_tilde]) + ((𝐠[:,j])[ℓ]* 𝔥_ij_k[i_tilde]) )
            q_ij_kℓ_jtilde = 0.5 *  ( ((𝐠[:,j])[k]* 𝔥_ij_ℓ[j_tilde]) + ((𝐠[:,j])[ℓ]* 𝔥_ij_k[j_tilde]) )
            term_itilde_jtilde = d_ij[i_tilde] * q_ij_kℓ_jtilde
            term_jtilde_itilde = d_ij[j_tilde] * q_ij_kℓ_itilde
            𝕊_ij_kℓ[i_tilde, j_tilde] = 0.5*(term_itilde_jtilde + term_jtilde_itilde)
        end
    end

    return c_tilde_ij_kℓ, d_ij, 𝕊_ij_kℓ

end

# usage:
# c_tilde_ij_kℓ, d_ij, 𝕊_ij_kℓ = constituents_of_λ_A_h(1, 0, 2, 2, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)


## Construct sum_λij_Aij_k_ℓ_over_ij_mat

function sum_λij_Aij_k_ℓ_over_ij_mat(k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, w, W, index_math2vec)
    vec_1 = zeros(len_w)
    mat_1 = zeros(len_w,len_w)
    for i in -1:N
        for j in -1:N
            if i != j
                c_tilde_ij_kℓ, d_ij, 𝕊_ij_kℓ = constituents_of_λ_A_h(i, j, k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, index_math2vec)
                vec_1 = vec_1 + (c_tilde_ij_kℓ*d_ij)
                mat_1 = mat_1 + 𝕊_ij_kℓ
            end
        end
    end
    sum_λij_Aij_k_ℓ_over_ij = dot(vec_1,w) + tr(mat_1*W)
    return sum_λij_Aij_k_ℓ_over_ij
end

# usage

# sum_λij_Aij_k_ℓ_over_ij_mat(1, 2, 𝐱_0, 𝐱, 𝐠, len_w, N, w, W, index_math2vec)


## Code to solve the semidefinite relaxation
# ------------------------------------------

function SDP_relaxation_solver_for_bound_generation(N, L, R;
    c_λ=1, c_h=0, c_Z=0,
    objective_type = :compute_bound,
    # other option is :original_performance_measure
    show_output = :off,
    obj_val_upper_bound = default_obj_val_upper_bound)

    ϵ_tol_feas = 1e-6

    if c_λ == 1 && c_h == 0 && c_Z == 0
        @info "computing M_λ"
    elseif c_λ == 0 && c_h == 1 && c_Z == 0
        @info "computing M_h"
    elseif c_λ == 0 && c_h == 0 && c_Z == 1
        @info "computing M_Z"
    else
        @error "exactly one of c_λ, c_h, c_Z, must be one, the other must be zero"
    end

    I_N_star = -1:N

    dim_Z = N+2

    model_lifted = Model(optimizer_with_attributes(Mosek.Optimizer))

    # define the variables
    # --------------------

    # define the index set of λ
    idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)

    # define λ
    @variable(model_lifted, λ[idx_set_λ] >= 0)

    # define ν
    @variable(model_lifted, ν >= 0)

    # define Z ⪰ 0
    @variable(model_lifted, Z[1:dim_Z, 1:dim_Z], PSD)

    @variable(model_lifted, h[i = 0:N-1])

    @variable(model_lifted, M_λ >= 0)

    @variable(model_lifted, M_h >= 0)

    @variable(model_lifted, M_Z >= 0)

    # bound constraints
    # -----------------

    if c_λ == 1 && c_h == 0 && c_Z == 0

        for i_j in idx_set_λ
            @constraint(model_lifted, λ[i_j] <= M_λ)
        end

    elseif c_λ == 0 && c_h == 1 && c_Z == 0
        for i in 0:N-1

            @constraint(model_lifted, h[i] <= M_h)
            @constraint(model_lifted,  h[i] >= -M_h)
        end

    elseif c_λ == 0 && c_h == 0 && c_Z == 1

        # for i in 1:dim_Z
        #     for j in 1:dim_Z
        #         @constraint(model_lifted, Z[i,j] <= M_Z)
        #         @constraint(model_lifted, Z[i,j] >= -M_Z)
        #     end
        # end

        # Because the diagonal entries of a matrix is >= 0, and Z_{i,j} <= (Z_{i,i}+Z_{j,j})/2,
        # we can bound just bound the diagonal entries

        for i in 1:dim_Z
            @constraint(model_lifted, Z[i,i] <= M_Z)
        end

    else

        @error "exactly one of c_λ, c_h, c_Z, must be one, the other must be zero"

    end

    # test to see if the relaxation gives the same solution if we fix h

    # define w

    w, index_math2vec, index_vec2math = vectorize_h_ν_λ(h, ν, λ, idx_set_λ)

    len_w = length(w)

    # define W

    @variable(model_lifted, W[1:len_w, 1:len_w], Symmetric)

    # ******************************
    # [🎍 ] add the data generator function
    # *******************************

    dim_𝐱 = N+2

    𝐱_0 = e_i(dim_𝐱, 1)

    𝐱_star = zeros(dim_𝐱, 1)

    𝐱, 𝐠, 𝐟 = data_generator_function(N, L, h; input_type = :stepsize_variable)

    # add the objective
    # -----------------

    if objective_type == :compute_bound

        # @objective(model_lifted, Max, (c_λ*M_λ) + (c_h*M_h) + (c_Z*M_Z)) #Min,  ν*R^2)

        @objective(model_lifted, Max, (c_λ*M_λ) + (c_h*M_h) + (c_Z*tr(Z))) #Min,  ν*R^2)

    elseif  objective_type == :original_performance_measure

        @objective(model_lifted, Min,  ν*R^2)

    end

    # add the linear constraint (need to fix)
    # -------------------------

    @constraint(model_lifted,   sum(λ[i_j_λ]*a_vec(i_j_λ.i,i_j_λ.j,𝐟) for i_j_λ in idx_set_λ) - a_vec(-1,N,𝐟) .== 0)

    # add the LMI constraint (need to fix)
    # ----------------------

    for k in 1:dim_𝐱
        for ℓ in 1:k

            sum_λij_Aij_k_ℓ_over_ij = sum_λij_Aij_k_ℓ_over_ij_mat(k, ℓ, 𝐱_0, 𝐱, 𝐠, len_w, N, w, W, index_math2vec)

            @constraint( model_lifted, (ν*⊙(𝐱_0 - 𝐱_star, 𝐱_0 - 𝐱_star) )[k,ℓ] + (1/(2*L))*(sum(λ[i_j_λ]*C_mat(i_j_λ.i,i_j_λ.j,𝐠) for i_j_λ in idx_set_λ))[k,ℓ] + sum_λij_Aij_k_ℓ_over_ij == Z[k,ℓ] )
        end
    end

    # confine the search in the space of known upper bound
    # ----------------------------------------------------

    @constraint(model_lifted, ν*R^2 <= obj_val_upper_bound)

    # add the Schur complement constraint
    # -----------------------------------

    @constraint(model_lifted, [W w; w' 1] in PSDCone())

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_lifted)
    end

    optimize!(model_lifted)

    # store the solution and return

    if objective_type == :compute_bound

        M_λ_opt = value(M_λ)

        M_Z_opt = value(M_Z)

        M_h_opt = value(M_h)

        return (c_λ*M_λ_opt) + (c_h*M_h_opt) + (c_Z*M_Z_opt)

    elseif  objective_type == :original_performance_measure

        # store λ_opt

        λ_opt = value.(λ)

        # store ν_opt

        ν_opt = value.(ν)

        # store h_opt

        h_opt = value.(h)

        # store Z_opt

        Z_opt = value.(Z)

        # store L_cholesky

        L_cholesky_opt = compute_pivoted_cholesky_L_mat(Z_opt)

        if norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 1e-6
            @info "checking the norm bound"
            @warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf))"
        end

        obj_val = objective_value(model_lifted)

        return obj_val, λ_opt, ν_opt, Z_opt, L_cholesky_opt, h_opt

    end

end


# L = 1
# N = 5
# R = 1
# h_test = feasible_h_generator(N, L; step_size_type = :Default)
#
# default_obj_val_upper_bound = 1e6
# #
# p_feas_1, G_feas_1, Ft_feas_1 = solve_primal_with_known_stepsizes(N, L, h_test, R; show_output = :on)
# #
# d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_test, R;  show_output = :off,
#     ϵ_tol_feas = 1e-6,
#     objective_type = :default,
#     obj_val_upper_bound = default_obj_val_upper_bound)
# #
# #
# obj_val_sdp_rlx, λ_opt_sdp_rlx, ν_opt_sdp_rlx, Z_opt_sdp_rlx, L_cholesky_opt_sdp_rlx, h_opt_sdp_rlx = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=0, c_h=0, c_Z=1,
#     objective_type =  :original_performance_measure,
#     # other option is :original_performance_measure
#     show_output = :off,
#     obj_val_upper_bound = default_obj_val_upper_bound)
#
# #
# M_λ = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=1, c_h=0, c_Z=0,
#     show_output = :off,
#     obj_val_upper_bound = d_feas_1)
#
# M_h = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=0, c_h=1, c_Z=0,
#     show_output = :off,
#     obj_val_upper_bound = d_feas_1)
#
# M_Z = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=0, c_h=0, c_Z=1,
#     show_output = :off,
#     obj_val_upper_bound = d_feas_1)


## Function that generates bounds on the variables from solving the SDP relaxation of the QCQO
# ---------------------------------------------------------------------

function bound_generator_through_SDP_relaxation(N, L, R, ν_feas;
    show_output = :off,
    obj_val_upper_bound = default_obj_val_upper_bound)

    M_λ = SDP_relaxation_solver_for_bound_generation(N, L, R;
    c_λ=1, c_h=0, c_Z=0,
    show_output = :off,
    obj_val_upper_bound = default_obj_val_upper_bound)

    M_α = SDP_relaxation_solver_for_bound_generation(N, L, R;
    c_λ=0, c_h=1, c_Z=0,
    show_output = :off,
    obj_val_upper_bound = default_obj_val_upper_bound)

    M_Z = SDP_relaxation_solver_for_bound_generation(N, L, R;
    c_λ=0, c_h=0, c_Z=1,
    show_output = :off,
    obj_val_upper_bound = default_obj_val_upper_bound)

    @info "computing M_P"

    M_P = sqrt(M_Z)

    M_ν = ν_feas

    return M_λ, M_α, M_Z, M_P, M_ν

end


# L = 1
# N = 5
# R = 1
# h_test = feasible_h_generator(N, L; step_size_type = :Default)
#
# default_obj_val_upper_bound = 1e6
# #
# p_feas_1, G_feas_1, Ft_feas_1 = solve_primal_with_known_stepsizes(N, L, h_test, R; show_output = :on)
# #
# d_feas_1,  ℓ_1_norm_λ_feas_1, tr_Z_feas_1, λ_feas_1, ν_feas_1, Z_feas_1, L_cholesky_feas_1, h_feas_1, idx_set_λ_feas_1_effective = solve_dual_PEP_with_known_stepsizes(N, L, h_test, R;  show_output = :off,
#     ϵ_tol_feas = 1e-6,
#     objective_type = :default,
#     obj_val_upper_bound = default_obj_val_upper_bound)
# #
#
# M_λ = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=1, c_h=0, c_Z=0,
#     show_output = :off,
#     obj_val_upper_bound = d_feas_1)
#
# M_h = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=0, c_h=1, c_Z=0,
#     show_output = :off,
#     obj_val_upper_bound = d_feas_1)
#
# M_Z = SDP_relaxation_solver_for_bound_generation(N, L, R;
#     c_λ=0, c_h=0, c_Z=1,
#     show_output = :off,
#     obj_val_upper_bound = d_feas_1)
#
# M_λ, M_h, M_Z, M_P, M_ν = bound_generator_through_SDP_relaxation(N, L, R, ν_feas_1; show_output = :off, obj_val_upper_bound = d_feas_1)

