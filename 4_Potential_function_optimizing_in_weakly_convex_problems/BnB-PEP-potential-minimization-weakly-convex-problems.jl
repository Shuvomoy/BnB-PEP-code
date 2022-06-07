
# using Weave
# cd("directory that contains the .jmd file") # directory that contains the .jmd file
# tangle("file_name.jmd", informat = "markdown")


## Load the packages:
# ------------------
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
# step_size_type = :Default => will create a last step of (R/M)/sqrt(N+1) rest will be zero

function feasible_h_generator(N, M, R; step_size_type = :Default)

    # construct h
    # -----------

    h = OffsetArray(zeros(N+1), 0:N)
    if step_size_type == :Default
        for i in 0:N
            h[i] = (R/M)/(sqrt(N+1))
        end
    end

    return h

end


## Data generator function
# ------------------------

function data_generator_potential_pep(h,k; input_type = :stepsize_constant)

    dim_𝐰 = 5
    dim_𝐠 = 5
    dim_𝐟 = 4
    N_pts = 5 # number of points corresponding to [x_⋆ x_k x_{k+1} y_k y_{k+1}]

    h_k = h[k]

    𝐰_0 = e_i(dim_𝐰, 1)

    𝐰_star = zeros(dim_𝐰, 1)

    # 𝐠 = [𝐠_{-1}=𝐠_⋆ | 𝐠_0 | 𝐠_1 | 𝐠_2 | 𝐠_3] ∈ 𝐑^(5×5)
    𝐠 =  OffsetArray(zeros(dim_𝐠, N_pts), 1:5, -1:3)

    # 𝐟  = [𝐟_{-1}=𝐟_⋆ | 𝐟_0 | 𝐟_1 | 𝐟_2 | 𝐟_3 ] ∈ 𝐑^(4×5)
    𝐟 = OffsetArray(zeros(dim_𝐟, N_pts), 1:4, -1:3)

    # construct 𝐠
    for k in 0:3
        𝐠[:,k] = e_i(dim_𝐠, k+2)
    end

    # construct 𝐟

    for k in 0:3
        𝐟[:,k] = e_i(dim_𝐟, k+1)
    end

    if input_type == :stepsize_constant

        # 𝐰 = [𝐱_{⋆} = 𝐰{-2} ∣ 𝐲_{⋆} = 𝐰_{-1} ∣  𝐰_0 𝐰_1 ... 𝐰_1 ∣ 𝐰_{1+1} ... 𝐰_{21+1}]

        𝐰 = OffsetArray(zeros(dim_𝐰, N_pts), 1:5, -1:3)

        # define 𝐰_0 which corresponds to x_0

        𝐰[:,0] = 𝐰_0

        𝐰[:,1] = 𝐰_0 - (h_k*𝐠[:,0])

        𝐰[:,2] = 𝐰_0 - (0.5*𝐠[:,2])

        𝐰[:,3] = 𝐰_0 - (h_k*𝐠[:,0]) - (0.5*𝐠[:,3])

    elseif input_type == :stepsize_variable

        # create a normal matrix first containing elements of 𝐰

        𝐰 = [𝐰_star 𝐰_0 (𝐰_0 - (h_k*𝐠[:,0])) (𝐰_0 - (0.5*𝐠[:,2])) (𝐰_0 - (h_k*𝐠[:,0]) - (0.5*𝐠[:,3]))]

        # make 𝐰 an offset array to make our life comfortable

        𝐰 = OffsetArray(𝐰, 1:5, -1:3)

    end

    # time to return

    return 𝐰, 𝐠, 𝐟

end


# Index set creator function for the dual variables λ, τ

struct i_j_k_idx
    i::Int64 # corresponds to index i ∈ [-1:3]
    j::Int64 # corresponds to index j ∈ [-1:3]
    k::Int64 # corresponds to index k ∈ [0:N]
end

struct i_k_idx # correspond i in some set
    i::Int64  # corresponds to index i ∈ [-1:3]
    k::Int64 # corresponds to index k ∈ [0:N]
end

# We have 3 dual variables that operate over mulitple index sets
# λ_{i,j,k} where i,j ∈ [-1:3], k ∈ [0:N]
# τ_{i,k} where  i ∈ [0:3], k ∈ [0:N]
# so we write a function to construct these index sets

function index_set_constructor_for_dual_vars_full(N)

    # construct the index set for λ
    idx_set_λ = i_j_k_idx[]
    for k in 0:N
    for i in -1:3
        for j in -1:3
            if i!=j
                push!(idx_set_λ, i_j_k_idx(i,j,k))
            end
        end
    end
    end

    # construct the index set for τ, both of which would have the same index set for the full case

    idx_set_τ = i_k_idx[]
    for k in 0:N
    for i in 0:3
        push!(idx_set_τ, i_k_idx(i,k))
    end
    end

    return idx_set_λ, idx_set_τ

end

function effective_index_set_finder(λ, τ; ϵ_tol = 0.0005)

    # the variables λ, τ, η are of the type DenseAxisArray whose index set can be accessed using _.axes and data via _.data syntax

    idx_set_λ_current = (λ.axes)[1]

    idx_set_τ_current = (τ.axes)[1]

    idx_set_λ_effective = i_j_k_idx[]

    idx_set_τ_effective = i_k_idx[]

    # construct idx_set_λ_effective

    for i_j_k_λ in idx_set_λ_current
        if abs(λ[i_j_k_λ]) >= ϵ_tol # if λ[i,j,k] >= ϵ, where ϵ is our cut off for accepting nonzero
            push!(idx_set_λ_effective, i_j_k_λ)
        end
    end

    # construct idx_set_τ_effective

    for i_k_τ in idx_set_τ_current
        if abs(τ[i_k_τ]) >= ϵ_tol
            push!(idx_set_τ_effective, i_k_τ)
        end
    end


    return idx_set_λ_effective, idx_set_τ_effective

end

# The following function will return the zero index set of a known L_cholesky i.e., those indices of  that are  L  that are zero. 💀 Note that for λ we are doing the opposite.

# function zero_index_set_finder_L_cholesky(L_cholesky, dim_Z, N; ϵ_tol = 1e-4)
#     zero_idx_set_L_cholesky = []
#     for k in 0:N
#         for i in 1:dim_Z
#             for j in 1:dim_Z
#                 if i >= j # because i<j has L_cholesky[i,j] == 0 for lower-triangual structure
#                     if abs(L_cholesky[i,j,k]) <= ϵ_tol
#                         push!(zero_idx_set_L_cholesky, (i,j,k))
#                     end
#                 end
#             end
#         end
#     end
#     return zero_idx_set_L_cholesky
# end

function zero_index_set_finder_L_cholesky(L_cholesky_k; ϵ_tol = 1e-4)
    n_L_cholesky, _ = size(L_cholesky_k)
    zero_idx_set_L_cholesky = []
    for i in 1:n_L_cholesky
        for j in 1:n_L_cholesky
            if i >= j # because i<j has L_cholesky_k[i,j] == 0 for lower-trianguar structure
                if abs(L_cholesky_k[i,j]) <= ϵ_tol
                    push!(zero_idx_set_L_cholesky, (i,j))
                end
            end
        end
    end
    return zero_idx_set_L_cholesky
end


function index_set_zero_entries_dual_variables(N, idx_set_λ, idx_set_τ)

    idx_set_nz_λ = []

    for k in 0:N
        idx_set_nz_λ = [idx_set_nz_λ; i_j_k_idx(0, 2, k); i_j_k_idx(0, 3, k); i_j_k_idx(2, 0, k); i_j_k_idx(2, 3, k)]
    end

    idx_set_zero_λ = setdiff(idx_set_λ, idx_set_nz_λ ) # this is the index set where the entries of λ will be zero

    idx_set_nz_τ = []

    for k in 0:N
        idx_set_nz_τ = [idx_set_nz_τ; i_k_idx(2,k)]
    end

    idx_set_zero_τ = setdiff(idx_set_τ, idx_set_nz_τ)

    return idx_set_zero_λ, idx_set_zero_τ

end


A_mat(i,j,k,h,𝐠,𝐰) = ⊙(𝐠[:,j], 𝐰[:,i]-𝐰[:,j])
B_mat(i,j,k,h,𝐰) = ⊙(𝐰[:,i]-𝐰[:,j], 𝐰[:,i]-𝐰[:,j])
C_mat(i,j,𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
a_vec(i,j,𝐟) = 𝐟[:, j] - 𝐟[:, i]
# another important function to find proper index of Θ given (i,j,k) pair
index_finder_Θ(i,j,k,idx_set_λ) = findfirst(isequal(i_j_k_idx(i,j,k)), idx_set_λ)



# In this function, the most important option is objective type:
# 0) :default will minimize ν*R^2 (this is the dual of the primal pep for a given stepsize)
# other options are
# 1) :find_sparse_sol, this will find a sparse solution given a particular stepsize and objective value upper bound
# 2) :find_M_λ , find the upper bound for the λ variables by maximizing ||λ||_1 for a given stepsize and particular objective value upper bound
# 3) :find_M_τ, find the upper bound for the τ variables by maximizing || τ ||_1
# 4) :find_M_Z, find the upper bound for the entries of the slack matrix Z, by maximizing tr(Z) for for a given stepsize and particular objective value upper bound
# 5) :find_M_b, find the upper bound for the entries of b
# 5) :find_M_c, find the upper bound for the entries of c

function solve_dual_with_known_stepsizes(N, M, h; show_output = :off, ϵ_tol_feas = 1e-6,
    objective_type = :default, obj_val_upper_bound = default_obj_val_upper_bound)

    dim_Z = 5


    model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    idx_set_λ, idx_set_τ = index_set_constructor_for_dual_vars_full(N)

    # define λ
    # --------
    @variable(model_dual_PEP_with_known_stepsizes, λ[idx_set_λ] >= 0)

    # define τ (the variable corresponding to lower bound)
    # -------
    @variable(model_dual_PEP_with_known_stepsizes, τ[idx_set_τ] >= 0)

    # Define the potential function coefficients b_i that ranges from 0:N+1

    @variable(model_dual_PEP_with_known_stepsizes, b[0:N+1] >= 0)

    # Define the potential function coefficients c_i that ranges from 0:N

    @variable(model_dual_PEP_with_known_stepsizes, c[0:N] >= 0)

    # Define the array of slack matrices Z[:,:,k] for k ∈ [0:N]

    # Z_unoffset = model_dual_PEP_with_known_stepsizes[:Z] = reshape(
    # hcat([
    # @variable(model_dual_PEP_with_known_stepsizes, [1:5, 1:5], PSD, base_name = "Z$(i)") for i in 0:N
    # ]...),
    # 5, 5, N+1) # here the index of Z_unoffset is Z_unoffset[ℓ] for ℓ ∈ [1,N+1]
    #
    # Z = OffsetArray(Z_unoffset, 1:5, 1:5, 0:N) # we do this, so that we can access Z^[k] as in math, where k ∈ [0:N]

    Z = Dict{Int64,Array{VariableRef,2}}(
              key => @variable(model_dual_PEP_with_known_stepsizes, [1:dim_Z, 1:dim_Z], PSD, base_name = "Z$(key)")
              for key in 0:N
          )

    # add objective
    # -------------

    if objective_type == :default

        @info "[🐒 ] Minimizing the usual performance measure"

        @objective(model_dual_PEP_with_known_stepsizes, Min, ( M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) )

    elseif objective_type == :find_sparse_sol

        @info "[🐮 ] Finding a sparse dual solution given the objective value upper bound"

        @objective(model_dual_PEP_with_known_stepsizes, Min, sum(τ[i] for i in idx_set_τ) + sum(λ[i_j] for i_j in idx_set_λ) )

        @constraint(model_dual_PEP_with_known_stepsizes, ( (M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) ) <= obj_val_upper_bound)

    elseif objective_type == :find_M_λ

        @info "[🐷 ] Finding upper bound on the entries of λ for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(λ[i_j] for i_j in idx_set_λ))

        @constraint(model_dual_PEP_with_known_stepsizes, ( (M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) ) <= obj_val_upper_bound)

    elseif objective_type == :find_M_τ

        @info "[🐷 ] Finding upper bound on the entries of τ for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(τ[i] for i in idx_set_τ))

        @constraint(model_dual_PEP_with_known_stepsizes, ( (M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) ) <= obj_val_upper_bound)

    elseif objective_type == :find_M_Z

        @info "[🐷 ] Finding upper bound on the entries of Z for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(tr(Z[i]) for i in 0:N))

        @constraint(model_dual_PEP_with_known_stepsizes, ( (M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) ) <= obj_val_upper_bound)

    elseif objective_type == :find_M_b

        @info "[🐷 ] Finding upper bound on the entries of b for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(b[i] for i in 0:N+1))

        @constraint(model_dual_PEP_with_known_stepsizes, ( (M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) ) <= obj_val_upper_bound)

    elseif objective_type == :find_M_c

        @info "[🐷 ] Finding upper bound on the entries of c for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(c[i] for i in 0:N))

        @constraint(model_dual_PEP_with_known_stepsizes, ( (M^2*(sum(c[i] for i in 0:N)) + (b[0]*R^2) ) /(N+1) ) <= obj_val_upper_bound )

    else

        @error "please input the correct option for objective_type"

        return

    end

    # add the constraints in loop now

    for k in 0:N

        # data generator

        𝐰, 𝐠, 𝐟 = data_generator_potential_pep(h,k; input_type = :stepsize_constant)

        # add the Linear constraint

        # -q^[k] + ∑_{i ∈ [0:3]} τ_i^[k] a[i,⋆] + ∑_{i,j ∈ [0:3]∪{⋆}} λ_{ij}^[k] a[i,j] =0
        # where q^[k] = b[k+1]*a[⋆,3] - b[k]*a[⋆,2]

        @constraint(model_dual_PEP_with_known_stepsizes,
        -(b[k+1]*a_vec(-1,3,𝐟) - b[k]*a_vec(-1,2,𝐟))
        + sum(τ[i_k_idx(i,k)]*a_vec(i,-1,𝐟) for i in 0:3)
        + sum(λ[i_j_k_λ]*a_vec(i_j_k_λ.i,i_j_k_λ.j,𝐟) for i_j_k_λ in idx_set_λ if i_j_k_λ.k == k) .== 0
        )

        # add the LMI constraint

        # -Q^[k] +  ∑_{i,j ∈ [0:3]∪{⋆}} λ_{ij}^[k] (A_{i,j}(h_k) - 0.5*B_{i,j}(h_k)) = Z[:,:,k]
        # where
        # Q^[k] = C_{2,⋆} + b[k+1]*B_{1,3}(h_k) - b[k]*B_{0,2}(h_k) - c[k]*C_{0,⋆}
        # note that B_{1,3}(h_k) = ⊙(0.5 𝐠_3, 0.5 𝐠_3)
        # B_{0,2}(h_k) = ⊙(0.5 𝐠_2, 0.5 𝐠_2)

        @constraint(model_dual_PEP_with_known_stepsizes,
        -( C_mat(2,-1,𝐠) + (b[k+1]*⊙(0.5*𝐠[:,3], 0.5*𝐠[:,3])) - (b[k]*⊙(0.5*𝐠[:,2],0.5*𝐠[:,2])) - (c[k]*C_mat(0,-1,𝐠)) ) +
         sum( λ[i_j_k_λ]*( A_mat(i_j_k_λ.i,i_j_k_λ.j,i_j_k_λ.k,h,𝐠,𝐰) - (0.5*B_mat(i_j_k_λ.i,i_j_k_λ.j,i_j_k_λ.k,h,𝐰)) ) for i_j_k_λ in idx_set_λ if i_j_k_λ.k == k )
        .== Z[k]
        )

    end

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_dual_PEP_with_known_stepsizes)
    end

    optimize!(model_dual_PEP_with_known_stepsizes)

    # @show termination_status(model_dual_PEP_with_known_stepsizes)

    if termination_status(model_dual_PEP_with_known_stepsizes) != MOI.OPTIMAL
        @info "model_dual_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_dual_PEP_with_known_stepsizes)
    end

    # store the solutions and return
    # ------------------------------

    # store λ_opt

    λ_opt = value.(λ)

    # store τ_opt

    τ_opt = value.(τ)

    # store Z_opt

    Z_opt = Dict{Int64,Array{Float64,2}}(
              key => value.(Z[key])
              for key in 0:N
          )

    # store b_opt

    b_opt = value.(b)

    # store c_opt

    c_opt = value.(c)

    # compute array of L_cholesky's

    # L_cholesky_opt = Dict{Int64,Array{Float64,2}}(
    #           key => compute_pivoted_cholesky_L_mat(Z_opt[key])
    #           for key in 0:N
    #       )

    L_cholesky_opt = Dict{Int64,Array{Float64,2}}(
            key =>  (cholesky(Z_opt[key]; check = false).L)
            for key in 0:N
        )

    for k in 0:N

        cholesky_error = norm(Z_opt[k] - (L_cholesky_opt[k])*(L_cholesky_opt[k])', Inf)

        if   cholesky_error > 1e-5
            @info "checking the norm bound"
            @warn "||Z - L*L^T|| = $(cholesky_error)"
        end

    end

    # Θ_{ij}^[k] is computed directly in the BnB-PEP solver along with its upper bound M_Θ

    # compute effective index sets for λ and τ

    idx_set_λ_effective, idx_set_τ_effective = effective_index_set_finder(λ_opt, τ_opt; ϵ_tol = 0.0005)

    # store objective value and other goodies

    ℓ_1_norm_λ = sum(λ_opt)

    ℓ_1_norm_τ = sum(τ_opt)

    tr_Z_sum = sum(tr(Z_opt[i]) for i in 0:N)

    original_performance_measure = ( M^2*(sum(c_opt[i] for i in 0:N)) + (b_opt[0]*R^2) ) /(N+1)

    return original_performance_measure, ℓ_1_norm_λ, ℓ_1_norm_τ, tr_Z_sum, λ_opt, τ_opt, Z_opt, L_cholesky_opt, b_opt, c_opt, idx_set_λ_effective, idx_set_τ_effective

end


function bound_violation_checker_BnB_PEP(
    N,
    # input point
    # -----------
    d_star_sol, λ_sol, τ_sol, Z_sol, L_cholesky_sol,  b_sol, c_sol, Θ_sol,  h_sol,
    # input bounds
    # ------------
    λ_lb, λ_ub, τ_lb, τ_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, Θ_lb, Θ_ub, h_lb, h_ub, b_lb, b_ub, c_lb, c_ub,
    # index set of λ is required for internal calculation
    idx_set_λ
    ;
    # options
    # -------
    show_output = :on,
    computing_global_lower_bound = :off
    )

    # compute minimum and maximum elements of Z, L_cholesky, and Θ

    max_elm_Z = []
    min_elm_Z = []
    max_elm_L_cholesky = []
    min_elm_L_cholesky = []
    max_elm_Θ = []
    min_elm_Θ = []

    for k in 0:N
        push!(max_elm_Z, maximum(Z_sol[k]))
        push!(min_elm_Z, minimum(Z_sol[k]))
        push!(max_elm_L_cholesky, maximum(L_cholesky_sol[k]))
        push!(min_elm_L_cholesky, minimum(L_cholesky_sol[k]))
    end

    max_elm_Θ = []
    min_elm_Θ = []

    for i_j_k_λ in idx_set_λ
        push!(max_elm_Θ, maximum(Θ_sol[i_j_k_λ]))
        push!(min_elm_Θ, maximum(Θ_sol[i_j_k_λ]))
    end

    if show_output == :on
        @show [minimum(λ_sol)  maximum(λ_sol)  λ_ub]
        @show [minimum(τ_sol)  maximum(τ_sol)  τ_ub]
        @show [Z_lb minimum(min_elm_Z)   maximum(max_elm_Z)  Z_ub]
        @show [L_cholesky_lb  minimum(min_elm_L_cholesky)  maximum(max_elm_L_cholesky) L_cholesky_ub]
        @show [minimum(b_sol) maximum(b_sol) b_ub]
        @show [minimum(c_sol) maximum(c_sol) c_ub]
        @show [h_lb minimum(h_sol) maximum(h_sol) h_ub]
        @show [Θ_lb minimum(min_elm_Θ) maximum(max_elm_Θ) Θ_ub]
    end

    # bound satisfaction flag

    bound_satisfaction_flag = 1

    # verify bound for λ
    if !(maximum(λ_sol) < λ_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found λ is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for b
    if !(maximum(b_sol) < b_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found b is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for c
    if !(maximum(c_sol) < c_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found c is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for τ
    if !(maximum(τ_sol) < τ_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found τ is violating the input bound"
        bound_satisfaction_flag = 0
    end

    if computing_global_lower_bound == :off

        # verify bound for Z
        if !(Z_lb -  1e-8 < minimum(min_elm_Z) &&  maximum(max_elm_Z) < Z_ub + 1e-8)
            @error "found Z is violating the input bound"
            bound_satisfaction_flag = 0
        end

        # verify bound for L_cholesky
        if computing_global_lower_bound == :off
            if !(L_cholesky_lb -  1e-8 < minimum(min_elm_L_cholesky) && maximum(max_elm_L_cholesky) < L_cholesky_ub +  1e-8)
                @error "found L_cholesky is violating the input bound"
                bound_satisfaction_flag = 0
            end
        elseif computing_global_lower_bound == :on
            @info "no need to check bound on L_cholesky"
        end

    end

    # verify bound for h
    if !(h_lb -  1e-8 < minimum(h_sol) && maximum(h_sol) < h_ub + 1e-8)
        @error "found h is violating the input bound"
        bound_satisfaction_flag = 0
    end


    # verify bound for Θ
    if !(Θ_lb -  1e-8 < minimum(min_elm_Θ) && maximum(max_elm_Θ) < Θ_ub + 1e-8)
        @error "found Z is violating the input bound"
        bound_satisfaction_flag = 0
    end

    if bound_satisfaction_flag == 0
        @error "[💀 ] some bound is violated, increase the bound intervals"
    elseif bound_satisfaction_flag == 1
        @info "[😅 ] all bounds are satisfied by the input point, rejoice"
    end

    return bound_satisfaction_flag

end


# BnB-PEP-solver function for potential function

function BnB_PEP_solver(
    # different parameters to be used
    # ------------------------------
    N,
    L,
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
    show_output = :off, # other option :on
    local_solver = :ipopt, # other option :knitro
    knitro_multistart = :off, # other option :on (only if :knitro solver is used)
    knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
    reduce_index_set_for_dual_variables = :for_warm_start_only,
    # options for reduce_index_set_for_dual_variables
    # (i) :on (making it :on will make force a dual variable 0, if the corresponding index lies int idx_set_variable_ws),
    # (ii) :off , this will define the dual variables and warm-start over the full index set (not recommended)
    # (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define dual variables over the full index set, but warm-start from a dual_variable that has reduced index set (recommended)
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
    maxCutCount = 1e3, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on, # wheather to polish the solution to get better precision, the other option is :off,
    M_Θ_factor = 100, # factor by which to magnify the internal M_Θ
    impose_pattern = :off # other option is :on, if it is turned on then we impose the pattern found by solving BnB-PEP from solving N=1,2,3
)

    # dimension of Z[k]
    dim_Z = 5



    ## Declare model

    if solution_type == :find_globally_optimal

        @info "[🐌 ] globally optimal solution finder activated, solution method: spatial branch and bound"

        BnB_PEP_model = Model(Gurobi.Optimizer)
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
        # set_optimizer_attribute(BnB_PEP_model, "FeasibilityTol", 1e-2)
        #
        # set_optimizer_attribute(BnB_PEP_model, "OptimalityTol", 1e-4)

    elseif solution_type == :find_locally_optimal

        @info "[🐙 ] locally optimal solution finder activated, solution method: interior point method"

        if local_solver == :knitro

            @info "[🚀 ] activating KNITRO"

            BnB_PEP_model = Model(
                optimizer_with_attributes(
                KNITRO.Optimizer,
                "convex" => 0,
                "strat_warm_start" => 1,
                # the last settings below are for larger N
                # you can comment them out if preferred but not recommended
                "honorbnds" => 1,
                # "bar_feasmodetol" => 1e-3,
                "feastol" => 1e-4,
                "infeastol" => 1e-12,
                "opttol" => 1e-4)
            )

            if knitro_multistart == :on
                set_optimizer_attribute(BnB_PEP_model, "ms_enable", 1)
                set_optimizer_attribute(BnB_PEP_model, "par_numthreads", 8)
                set_optimizer_attribute(BnB_PEP_model, "par_msnumthreads", 8)
                # set_optimizer_attribute(BnB_PEP_model, "ms_maxsolves", 200)
            end

            if knitro_multi_algorithm == :on
                set_optimizer_attribute(BnB_PEP_model, "algorithm", 5)
                set_optimizer_attribute(BnB_PEP_model, "ma_terminate", 0)
            end

        elseif local_solver == :ipopt

            @info "[🎃 ] activating IPOPT"

            BnB_PEP_model = Model(Ipopt.Optimizer)

            set_optimizer_attribute(BnB_PEP_model, "constr_viol_tol", 1e-4)

            set_optimizer_attribute(BnB_PEP_model, "dual_inf_tol", 1e-4)

            set_optimizer_attribute(BnB_PEP_model, "compl_inf_tol", 1e-4)

            set_optimizer_attribute(BnB_PEP_model, "tol", 1e-5)

            set_optimizer_attribute(BnB_PEP_model, "acceptable_tol", 1e-4)

            set_optimizer_attribute(BnB_PEP_model, "max_iter", 5000)

        end
    end

    ## Define all the variables

    @info "[🎉 ] defining the variables"

    ## define λ, τ
    # -----------

    if reduce_index_set_for_dual_variables == :off
        # define λ,  τ over the full index set
        idx_set_λ, idx_set_τ = index_set_constructor_for_dual_vars_full(N)
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
        @variable(BnB_PEP_model, τ[idx_set_τ] >= 0)
    elseif reduce_index_set_for_dual_variables == :on
        # define λ over a reduced index set, idx_set_λ_ws_effective, which is the effective index set of λ_ws, and so on for other variables
        idx_set_λ = idx_set_λ_ws_effective
        idx_set_τ = idx_set_τ_ws_effective
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
        @variable(BnB_PEP_model, τ[idx_set_τ] >= 0)
    elseif reduce_index_set_for_dual_variables == :for_warm_start_only
        # this :for_warm_start_only option is same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set, and so on for other variables
        idx_set_λ, idx_set_τ = index_set_constructor_for_dual_vars_full(N)
        idx_set_λ_ws = idx_set_λ_ws_effective
        idx_set_τ_ws = idx_set_τ_ws_effective
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
        @variable(BnB_PEP_model, τ[idx_set_τ] >= 0)
    end

    # define b
    # --------

    @variable(BnB_PEP_model, b[0:N+1] >= 0)

    # define c
    # -------

    @variable(BnB_PEP_model, c[0:N] >= 0)


    # Define the stepsize matrix h
    # ----------------------------

    if positive_step_size == :off
        @variable(BnB_PEP_model, h[i = 0:N])
    elseif positive_step_size == :on
        @variable(BnB_PEP_model, h[i = 0:N] >= 0)
    end

    # 💀 fixing h

    for k in 1:N
        @constraint(BnB_PEP_model, h[k] == h[0])
    end

    # Z[k], L_cholesky[k], and Θ[k] where k∈[0:N]  are defined as JuMP dictionaries

    Z = Dict{Int64,Array{VariableRef,2}}(
        key => @variable(
            BnB_PEP_model,
            [1:dim_Z, 1:dim_Z],
            Symmetric,
            base_name = "Z$(key)"
        ) for key = 0:N
    )

    L_cholesky = Dict{Int64,Array{VariableRef,2}}(
        key => @variable(BnB_PEP_model, [1:dim_Z, 1:dim_Z], base_name = "L_cholesky$(key)") for key = 0:N
    )

    Θ = Dict{i_j_k_idx,Array{VariableRef,2}}(
        key => @variable(
            BnB_PEP_model,
            [1:dim_Z, 1:dim_Z],
            Symmetric,
            base_name = "Θ$(key)"
        ) for key in idx_set_λ
    )

    @info "[👲 ] warm-start values for all the variables"


    # warm start for λ, τ
    # -------------------
    if reduce_index_set_for_dual_variables == :for_warm_start_only
        # warm start for λ
        for i_j_k_λ in idx_set_λ_ws
            set_start_value(λ[i_j_k_λ], λ_ws[i_j_k_λ])
        end
        for i_j_k_λ in setdiff(idx_set_λ, idx_set_λ_ws)
            set_start_value(λ[i_j_k_λ], 0.0)
        end
        # warm start for τ
        for i_k_τ in idx_set_τ_ws
            set_start_value(τ[i_k_τ], τ_ws[i_k_τ])
        end
        for i_k_τ in setdiff(idx_set_τ, idx_set_τ_ws)
            set_start_value(τ[i_k_τ], 0.0)
        end
    else
        # warm start for λ
        for i_j_k_λ in idx_set_λ
            set_start_value(λ[i_j_k_λ], λ_ws[i_j_k_λ])
        end
        # warm start for τ
        for i_k_τ in idx_set_τ
            set_start_value(τ[i_k_τ], τ_ws[i_k_τ])
        end
    end

    # warm start for Z
    # ----------------

    for k = 0:N
        set_start_value.(Z[k], Z_ws[k])
    end

    # warm start for L_cholesky
    # -------------------------

    for k = 0:N
        set_start_value.(L_cholesky[k], L_cholesky_ws[k])
    end

    # warm start for h
    # ----------------

    for i = 0:N
        set_start_value(h[i], h_ws[i])
    end

    # warm start for Θ
    # ----------------

    Θ_ws = Dict{i_j_k_idx,Array{Float64,2}}()

    for i_j_k_λ in idx_set_λ
        k = i_j_k_λ.k
        i = i_j_k_λ.i
        j = i_j_k_λ.j
        𝐰_ws, 𝐠_ws, 𝐟_ws =
            data_generator_potential_pep(h_ws, k; input_type = :stepsize_constant)
        Θ_ws[i_j_k_λ] = ⊙(𝐰_ws[:, i] - 𝐰_ws[:, j], 𝐰_ws[:, i] - 𝐰_ws[:, j])
		set_start_value.(Θ[i_j_k_λ] ,Θ_ws[i_j_k_λ])
    end


    # warm start b
    # -----------

    set_start_value.(b, b_ws)

    # warm start c
    # ------------

    set_start_value.(c, c_ws)

    # compute M_Θ
    # -----------

    max_elm_Θ_ws = []

    for i_j_k_λ in idx_set_λ
        push!(max_elm_Θ_ws, maximum(abs.(Θ_ws[i_j_k_λ])))
    end

    M_Θ = M_Θ_factor * max(1, maximum(abs.(max_elm_Θ_ws)))

    # ************
    # [🎇 ] add objective
    # -------------
    # *************

    @info "[🎇 ] adding objective"

    @objective(
        BnB_PEP_model,
        Min,
        (M^2 * (sum(c[i] for i = 0:N)) + (b[0] * R^2)) / (N + 1)
    )


    # adding an upper bound for objective function

    @constraint(
        BnB_PEP_model,
        ((M^2 * (sum(c[i] for i = 0:N)) + (b[0] * R^2)) / (N + 1)) <= 1.001 * d_star_ws
    )

    # Adding a lower bound for the objective function (if given)
    if global_lower_bound_given == :on
        @constraint(
            BnB_PEP_model,
            ((M^2 * (sum(c[i] for i = 0:N)) + (b[0] * R^2)) / (N + 1)) >=
            global_lower_bound
        )
    end

    # ************************************
    # [🎇 ] add constraints in a loop
    # -------------
    # *************************************

    for k = 0:N

        # load the data generator for k
        # -----------------------------

        𝐰, 𝐠, 𝐟 = data_generator_potential_pep(h, k; input_type = :stepsize_variable)

        # add linear constraints
        # ----------------------


        if length(findall(x -> x.k == k, idx_set_λ)) > 0 && length(findall(x -> x.k == k, idx_set_τ)) > 0

            @constraint(
            BnB_PEP_model,
            -(b[k+1] * a_vec(-1, 3, 𝐟) - b[k] * a_vec(-1, 2, 𝐟)) +
            sum(τ[i_k_τ] * a_vec(i_k_τ.i, -1, 𝐟) for i_k_τ in idx_set_τ if i_k_τ.k == k) +
            sum(
            λ[i_j_k_λ] * a_vec(i_j_k_λ.i, i_j_k_λ.j, 𝐟) for
            i_j_k_λ in idx_set_λ if i_j_k_λ.k == k
                ) .== 0
                )

        elseif length(findall(x -> x.k == k, idx_set_λ)) > 0 && length(findall(x -> x.k == k, idx_set_τ)) == 0

            @constraint(
                BnB_PEP_model,
                -(b[k+1] * a_vec(-1, 3, 𝐟) - b[k] * a_vec(-1, 2, 𝐟)) +
                sum(
                λ[i_j_k_λ] * a_vec(i_j_k_λ.i, i_j_k_λ.j, 𝐟) for
                i_j_k_λ in idx_set_λ if i_j_k_λ.k == k
                    ) .== 0
                    )

        elseif length(findall(x -> x.k == k, idx_set_λ)) == 0 && length(findall(x -> x.k == k, idx_set_τ)) > 0

                @constraint(
                    BnB_PEP_model,
                    -(b[k+1] * a_vec(-1, 3, 𝐟) - b[k] * a_vec(-1, 2, 𝐟)) +
                    sum(τ[i_k_τ] * a_vec(i_k_τ.i, -1, 𝐟) for i_k_τ in idx_set_τ if i_k_τ.k == k) .== 0
                    )

        elseif length(findall(x -> x.k == k, idx_set_λ)) == 0 && length(findall(x -> x.k == k, idx_set_τ)) == 0

                @constraint(
                    BnB_PEP_model,
                    -(b[k+1] * a_vec(-1, 3, 𝐟) - b[k] * a_vec(-1, 2, 𝐟))  .== 0
                    )

        else

            @error "something is not right in linear constraint creation"

        end



        # @show sum(τ[i_k_τ] * a_vec(i_k_τ.i, -1, 𝐟) for i_k_τ in idx_set_τ if i_k_τ.k == k)

        # @info sum(
        #     λ[i_j_k_λ] * a_vec(i_j_k_λ.i, i_j_k_λ.j, 𝐟) for
        #     i_j_k_λ in idx_set_λ if i_j_k_λ.k == k
        # )

        # add the LMI constraint
        # ----------------------

        if length(findall(x -> x.k == k, idx_set_λ)) > 0

            @constraint(
            BnB_PEP_model,
            vectorize(
            -(
            C_mat(2, -1, 𝐠) + (b[k+1] * ⊙(0.5 * 𝐠[:, 3], 0.5 * 𝐠[:, 3])) -
            (b[k] * ⊙(0.5 * 𝐠[:, 2], 0.5 * 𝐠[:, 2])) - (c[k] * C_mat(0, -1, 𝐠))
            ) + sum(
            λ[i_j_k_λ] *
            (A_mat(i_j_k_λ.i, i_j_k_λ.j, i_j_k_λ.k, h, 𝐠, 𝐰) - (0.5 * Θ[i_j_k_λ]))
            for i_j_k_λ in idx_set_λ if i_j_k_λ.k == k
                ) - Z[k],
                SymmetricMatrixShape(dim_Z),
                ) .== 0
            )

        elseif length(findall(x -> x.k == k, idx_set_λ)) == 0

            @constraint(
            BnB_PEP_model,
            vectorize(
            -(
            C_mat(2, -1, 𝐠) + (b[k+1] * ⊙(0.5 * 𝐠[:, 3], 0.5 * 𝐠[:, 3])) -
            (b[k] * ⊙(0.5 * 𝐠[:, 2], 0.5 * 𝐠[:, 2])) - (c[k] * C_mat(0, -1, 𝐠))
            )  - Z[k],
            SymmetricMatrixShape(dim_Z),
            ) .== 0
            )

        else

            @error "something is not rigth in the quadaratic constraint creation"

        end



        for i_j_k_λ in idx_set_λ
            if i_j_k_λ.k == k
                @constraint(
                    BnB_PEP_model,
                    vectorize(
                        Θ[i_j_k_λ] - B_mat(i_j_k_λ.i, i_j_k_λ.j, i_j_k_λ.k, h, 𝐰),
                        SymmetricMatrixShape(dim_Z),
                    ) .== 0
                )
            end
        end



        # Add valid constraints for Z^[k] ⪰ 0

        # diagonal components of Z are non-negative

        for i = 1:dim_Z
            @constraint(BnB_PEP_model, Z[k][i, i] >= 0)
        end

        # the off-diagonal components satisfy:
        # (∀i,j ∈ dim_Z: i != j) -(0.5*(Z[k][i,i] + Z[k][j,j])) <= Z[k][i,j] <=  (0.5*(Z[k][i,i] + Z[k][j,j]))

        for i = 1:dim_Z
            for j = 1:dim_Z
                if i != j
                    @constraint(
                        BnB_PEP_model,
                        Z[k][i, j] <= (0.5 * (Z[k][i, i] + Z[k][j, j]))
                    )
                    @constraint(
                        BnB_PEP_model,
                        -(0.5 * (Z[k][i, i] + Z[k][j, j])) <= Z[k][i, j]
                    )
                end
            end
        end


        # add cholesky related constraints
        # --------------------------------

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off


            # Two constraints to define the matrix L_cholesky[k] to be a lower triangular matrix
            # -------------------------------------------------

            # upper off-diagonal terms of L_cholesky are zero

            for i = 1:dim_Z
                for j = 1:dim_Z
                    if i < j
                        # @constraint(BnB_PEP_model, L_cholesky[k][i,j] .== 0)
                        fix((L_cholesky[k])[i, j], 0; force = true)
                    end
                end
            end

            # diagonal components of L_cholesky are non-negative

            for i = 1:dim_Z
                @constraint(BnB_PEP_model, L_cholesky[k][i, i] >= 0)
            end

        end

        # time to implement Z[k] = L*L^T constraint
        # --------------------------------------

        if cholesky_modeling == :definition &&
           find_global_lower_bound_via_cholesky_lazy_constraint == :off

           # set_optimizer_attribute(BnB_PEP_model, "Heuristics", 0)
           #
           # set_optimizer_attribute(BnB_PEP_model, "RINS", 0)

            if quadratic_equality_modeling == :exact

                # direct modeling through definition and vectorization
                # ---------------------------------------------------
                @constraint(
                    BnB_PEP_model,
                    vectorize(
                        Z[k] - (L_cholesky[k] * (L_cholesky[k])'),
                        SymmetricMatrixShape(dim_Z),
                    ) .== 0
                )

            elseif quadratic_equality_modeling == :through_ϵ

                # definition modeling through vectorization and ϵ_tol_feas

                # part 1: models Z[k]-L_cholesky[k]*L_cholesky[k] <= ϵ_tol_feas*ones(dim_Z,dim_Z)
                @constraint(
                    BnB_PEP_model,
                    vectorize(
                        Z[k] - (L_cholesky[k] * (L_cholesky[k])') -
                        ϵ_tol_feas * ones(dim_Z, dim_Z),
                        SymmetricMatrixShape(dim_Z),
                    ) .<= 0
                )

                # part 2: models Z[k]-L_cholesky[k]*L_cholesky[k] >= -ϵ_tol_feas*ones(dim_Z,dim_Z)

                @constraint(
                    BnB_PEP_model,
                    vectorize(
                        Z[k] - (L_cholesky[k] * (L_cholesky[k])') +
                        ϵ_tol_feas * ones(dim_Z, dim_Z),
                        SymmetricMatrixShape(dim_Z),
                    ) .>= 0
                )

            else

                @error "something is not right in Cholesky modeling"

                return

            end


        elseif cholesky_modeling == :formula &&
               find_global_lower_bound_via_cholesky_lazy_constraint == :off

            # Cholesky constraint 1
            # (∀ j ∈ dim_Z) L_cholesky[k][j,j]^2 + ∑_{ℓ∈[1:j-1]} L_cholesky[k][j,ℓ]^2 == Z[k][j,j]

            for j = 1:dim_Z
                if j == 1
                    @constraint(BnB_PEP_model, (L_cholesky[k][j, j])^2 == Z[k][j, j])
                elseif j > 1
                    @constraint(
                        BnB_PEP_model,
                        (L_cholesky[k][j, j])^2 +
                        sum((L_cholesky[k][j, ℓ])^2 for ℓ = 1:j-1) == Z[k][j, j]
                    )
                end
            end

            # Cholesky constraint 2
            # (∀ i,j ∈ dim_Z: i > j) L_cholesky[k][i,j] L_cholesky[k][j,j] + ∑_{ℓ∈[1:j-1]} L_cholesky[k][i,ℓ] L_cholesky[k][j,ℓ] == Z[k][i,j]

            for i = 1:dim_Z
                for j = 1:dim_Z
                    if i > j
                        if j == 1
                            @constraint(
                                BnB_PEP_model,
                                L_cholesky[k][i, j] * L_cholesky[k][j, j] == Z[k][i, j]
                            )
                        elseif j > 1
                            @constraint(
                                BnB_PEP_model,
                                L_cholesky[k][i, j] * L_cholesky[k][j, j] + sum(
                                    L_cholesky[k][i, ℓ] * L_cholesky[k][j, ℓ] for ℓ = 1:j-1
                                ) == Z[k][i, j]
                            )
                        end
                    end
                end
            end

        elseif find_global_lower_bound_via_cholesky_lazy_constraint == :on

            # set_optimizer_attribute(BnB_PEP_model, "FuncPieces", -2) # FuncPieces = -2: Bounds the relative error of the approximation; the error bound is provided in the FuncPieceError attribute. See https://www.gurobi.com/documentation/9.1/refman/funcpieces.html#attr:FuncPieces

            # set_optimizer_attribute(BnB_PEP_model, "FuncPieceError", 0.1) # relative error

            set_optimizer_attribute(BnB_PEP_model, "MIPFocus", 1) # focus on finding good quality feasible solution

            # set_optimizer_attribute(BnB_PEP_model, "FeasibilityTol", 1e-2) # feasibility tolerance

            # add initial cuts
            num_cutting_planes_init = 2 * dim_Z^2
            cutting_plane_array = randn(dim_Z, num_cutting_planes_init)
            num_cuts_array_rows, num_cuts = size(cutting_plane_array)
            for i = 1:num_cuts
                d_cut = cutting_plane_array[:, i]
                d_cut = d_cut / norm(d_cut, 2) # normalize the cutting plane vector
                @constraint(BnB_PEP_model, tr(Z[k] * (d_cut * d_cut')) >= 0)
            end

            cutCount = 0
            # maxCutCount=1e3

            # add the lazy callback function
            # ------------------------------
            function add_lazy_callback(cb_data)
                if cutCount <= maxCutCount
                    Z0 = zeros(dim_Z, dim_Z)
                    for i = 1:dim_Z
                        for j = 1:dim_Z
                            Z0[i, j] = callback_value(cb_data, Z[k][i, j])
                        end
                    end
                    if eigvals(Z0)[1] <= -0.01
                        u_t = eigvecs(Z0)[:, 1]
                        u_t = u_t / norm(u_t, 2)
                        con3 = @build_constraint(tr(Z[k] * u_t * u_t') >= 0.0)
                        MOI.submit(BnB_PEP_model, MOI.LazyConstraint(cb_data), con3)
                        # noPSDCuts+=1
                    end
                    cutCount += 1
                end
            end

            # submit the lazy constraint
            # --------------------------
            MOI.set(BnB_PEP_model, MOI.LazyConstraintCallback(), add_lazy_callback)


        end


        # impose the effective index set of L_cholesky[k] if reduce_index_set_for_L_cholesky  == :on and we are not computing a global lower bound
        # ------------------------------------------

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off &&
           reduce_index_set_for_L_cholesky == :on
            zis_Lc =
                zero_index_set_finder_L_cholesky(L_cholesky_ws[k]; ϵ_tol = ϵ_tol_Cholesky)
            for ℓ = 1:length(zis_Lc)
                # @constraint(BnB_PEP_model, L_cholesky[k][CartesianIndex(zis_Lc[ℓ])] == 0)
                fix(L_cholesky[k][CartesianIndex(zis_Lc[ℓ])], 0; force = true)
            end
        end


    end # for k in 0:N (constraints in loop is added)

    # impose bound 💀 this is outside k ∈ [0:N] loop
    # -----------------------------------------------

    if bound_impose == :on
        @info "[🌃 ] finding bound on the variables"

        # store the values

        λ_lb = 0
        λ_ub = M_λ
        τ_lb = 0
        τ_ub = M_τ
        Z_lb = -M_Z
        Z_ub = M_Z
        L_cholesky_lb = -M_L_cholesky
        L_cholesky_ub = M_L_cholesky
        h_lb = -M_h
        h_ub = M_h
        Θ_lb = -M_Θ
        Θ_ub = M_Θ
        b_lb = 0
        b_ub = M_b
        c_lb = 0
        c_ub = M_c

        # set bound for λ
        # ---------------
        # set_lower_bound.(λ, λ_lb): done in definition
        set_upper_bound.(λ, λ_ub)

        # set bound for τ
        # ---------------
        # set_lower_bound.(τ, τ_lb): done in definition
        set_upper_bound.(τ, τ_ub)

        # set bound for Z
        # ---------------

        for k = 0:N
            set_lower_bound.(Z[k], Z_lb)
            set_upper_bound.(Z[k], Z_ub)
        end


        # set bound for L_cholesky
        # ------------------------

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off
            # need only upper bound for the diagonal compoments, as the lower bound is zero from the model
            for k = 0:N
                for i = 1:dim_Z
                    set_upper_bound(L_cholesky[k][i, i], L_cholesky_ub)
                end
            end
            # need to bound only components, L_cholesky[k][i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
            for k = 0:N
                for i = 1:dim_Z
                    for j = 1:dim_Z
                        if i > j
                            set_lower_bound(L_cholesky[k][i, j], L_cholesky_lb)
                            set_upper_bound(L_cholesky[k][i, j], L_cholesky_ub)
                        end
                    end
                end
            end
        end

        # set bound for Θ
        # ---------------

        for i_j_k_λ in idx_set_λ
            set_lower_bound.(Θ[i_j_k_λ], Θ_lb)
            set_upper_bound.(Θ[i_j_k_λ], Θ_ub)
        end

        # set bound for h
        # ---------------
        set_lower_bound.(h, h_lb)
        set_upper_bound.(h, h_ub)

        # set bound for b
        # ---------------
        set_lower_bound.(b, b_lb)
        set_upper_bound.(b, b_ub)

        # set bound for c
        # ---------------
        set_lower_bound.(c, c_lb)
        set_upper_bound.(c, c_ub)

    end

    ## impose pattern
    # ---------------

    if impose_pattern == :on

        # 🍏 impose the zero entries of λ and τ


        idx_set_zero_λ, idx_set_zero_τ = index_set_zero_entries_dual_variables(N, idx_set_λ, idx_set_τ)

        for i_j_k_λ in idx_set_zero_λ
            fix(λ[i_j_k_λ], 0.0; force = true)
        end

        for i_k_τ in idx_set_zero_τ
            fix(τ[i_k_τ], 0.0; force = true)
        end

        # 🍎 impose the zero entries of Z

        for k in 0:N
            fix(Z[k][1,1], 0.0; force = true)
            fix(Z[k][2,1], 0.0; force = true)
            fix(Z[k][3,1], 0.0; force = true)
            fix(Z[k][4,1], 0.0; force = true)
            fix(Z[k][5,1], 0.0; force = true)
            fix(Z[k][3,2], 0.0; force = true)
            fix(Z[k][3,3], 0.0; force = true)
            fix(Z[k][4,3], 0.0; force = true)
            fix(Z[k][5,3], 0.0; force = true)
        end

        # impose the constraints that hold with very high probability
        # Z[k][4,2] == -Z[k][5,2], Z[k][4,4] == Z[k][5,5], Z[k][5,4]=-Z[k][4,4]
        # λ[k][0,2] == λ[k][2,0] and λ[k][0,3] == 0

        for k in 0:N
            @constraint(BnB_PEP_model, Z[k][4,2] == -Z[k][5,2])
            @constraint(BnB_PEP_model, Z[k][4,4] == Z[k][5,5])
            @constraint(BnB_PEP_model, Z[k][5,4] == -Z[k][4,4])
            @constraint(BnB_PEP_model, λ[i_j_k_idx(0,2,k)] == λ[i_j_k_idx(2,0,k)])
            @constraint(BnB_PEP_model, λ[i_j_k_idx(0,3,k)] == 0.0)
        end

        # last entry of b and c are zero
        fix(b[N+1], 0.0; force = true)
        fix(c[N], 0.0; force = true)


    end # end for imposte pattern condition

    ## time to optimize
    # ----------------

    @info "[🙌 	🙏 ] model building done, starting the optimization process"

    if show_output == :off
        set_silent(BnB_PEP_model)
    end

    # time to optimize!

    optimize!(BnB_PEP_model)

    @info "BnB_PEP_model has termination status = " termination_status(BnB_PEP_model)

    ## Store and return

    if (
        solution_type == :find_locally_optimal &&
        termination_status(BnB_PEP_model) == MOI.LOCALLY_SOLVED
    ) || (
        solution_type == :find_globally_optimal &&
        termination_status(BnB_PEP_model) == MOI.OPTIMAL
    )

        # store the solutions and return
        # ------------------------------

        @info "[😻 ] optimal solution found done, store the solution"

        # store optimal value

        obj_val = objective_value(BnB_PEP_model)

        # store λ_opt

        λ_opt = value.(λ)

        # store τ_opt

        τ_opt = value.(τ)

        # store Z_opt

        # store Z_opt

        Z_opt = Dict{Int64,Array{Float64,2}}(key => value.(Z[key]) for key = 0:N)

        # store b_opt

        b_opt = value.(b)

        # store c_opt

        c_opt = value.(c)

        # store h_opt

        h_opt = value.(h)

        # store Θ_opt

        Θ_opt = Dict{i_j_k_idx,Array{Float64,2}}()

        for i_j_k_λ in idx_set_λ
            k = i_j_k_λ.k
            i = i_j_k_λ.i
            j = i_j_k_λ.j
            𝐰_ws, 𝐠_ws, 𝐟_ws =
                data_generator_potential_pep(h_opt, k; input_type = :stepsize_constant)
            Θ_opt[i_j_k_λ] = ⊙(𝐰_ws[:, i] - 𝐰_ws[:, j], 𝐰_ws[:, i] - 𝐰_ws[:, j])
        end

        # store L_cholesky

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off

            # compute array of L_cholesky's

            L_cholesky_opt =
                Dict{Int64,Array{Float64,2}}(key => value.(L_cholesky[key]) for key = 0:N)

        elseif find_global_lower_bound_via_cholesky_lazy_constraint == :on

            # compute array of L_cholesky's

            # L_cholesky_opt = Dict{Int64,Array{Float64,2}}(
            #     key => compute_pivoted_cholesky_L_mat(Z_opt[key]) for key = 0:N
            # )

            L_cholesky_opt = Dict{Int64,Array{Float64,2}}(
                    key =>  (cholesky(Z_opt[key]; check = false).L)
                    for key in 0:N
                )

            for k = 0:N
                cholesky_error =
                    norm(Z_opt[k] - (L_cholesky_opt[k]) * (L_cholesky_opt[k])', Inf)

                # if cholesky_error > 1e-5
                #     @info "checking the norm bound"
                #     @warn "||Z - L*L^T|| = $(cholesky_error)"
                # end

            end

        end

    else

        @warn "[🙀 ] could not find an optimal solution, returning the warm-start point"

        obj_val,
        λ_opt,
        τ_opt,
        Z_opt,
        L_cholesky_opt,
        b_opt,
        c_opt,
        Θ_opt,
        h_opt,
        idx_set_λ_opt_effective,
        idx_set_τ_opt_effective = d_star_ws,
        λ_ws,
        τ_ws,
        Z_ws,
        L_cholesky_ws,
        b_ws,
        c_ws,
        Θ_ws,
        h_ws,
        idx_set_λ_ws_effective,
        idx_set_τ_ws_effective

    end

    # 💀:  polish_solution == :on through semidefinite programming causes a slight constraint violation in this case, so if it is turned on reduce the sensibility of feasibility tolerance in the solver used. For now recommendation is keep this option off.


    if polish_solution == :on &&
       find_global_lower_bound_via_cholesky_lazy_constraint == :off # note that if we are finding a global lower bound, then polishing the solution would not make sense

        @info "[🎣 ] polishing and sparsifying the solution"

        obj_val,
        ℓ_1_norm_λ_dummy,
        ℓ_1_norm_τ_dummy,
        tr_Z_sum_dummy,
        λ_opt,
        τ_opt,
        Z_opt,
        L_cholesky_opt,
        b_opt,
        c_opt,
        idx_set_λ_effective_dummy,
        idx_set_τ_effective_dummy = solve_dual_with_known_stepsizes(
            N,
            M,
            h_opt;
            show_output = :off,
            ϵ_tol_feas = 1e-6,
            objective_type = :default,
            obj_val_upper_bound = 1.0001 * obj_val,
        )

        obj_val_sparse,
        ℓ_1_norm_λ_dummy,
        ℓ_1_norm_τ_dummy,
        tr_Z_sum_dummy,
        λ_opt,
        τ_opt,
        Z_opt,
        L_cholesky_opt,
        b_opt,
        c_opt,
        idx_set_λ_effective_dummy,
        idx_set_τ_effective_dummy = solve_dual_with_known_stepsizes(
            N,
            M,
            h_opt;
            show_output = :off,
            ϵ_tol_feas = 1e-6,
            objective_type = :find_sparse_sol,
            obj_val_upper_bound = (1 + (1e-6)) * obj_val,
        )

    end

    # find the effective index set of the found λ, τ, η

    idx_set_λ_opt_effective, idx_set_τ_opt_effective  =
        effective_index_set_finder(λ_opt, τ_opt; ϵ_tol = 0.0005)

    @info "[🚧 ] for λ, only $(length(idx_set_λ_opt_effective)) components out of $(length(idx_set_λ)) are non-zero for the optimal solution"

    @info "[🚧 ] for τ, only $(length(idx_set_τ_opt_effective)) components out of $(length(idx_set_τ)) are non-zero for the optimal solution"


    @info "[💹 ] warm-start objective value = $d_star_ws, and objective value of found solution = $obj_val"

    # verify if any of the imposed bounds are violated

    if bound_impose == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :off
        bound_satisfaction_flag = bound_violation_checker_BnB_PEP(
            N,
            obj_val,
            λ_opt,
            τ_opt,
            Z_opt,
            L_cholesky_opt,
            b_opt,
            c_opt,
            Θ_opt,
            h_opt,
            λ_lb,
            λ_ub,
            τ_lb,
            τ_ub,
            Z_lb,
            Z_ub,
            L_cholesky_lb,
            L_cholesky_ub,
            Θ_lb,
            Θ_ub,
            h_lb,
            h_ub,
            b_lb,
            b_ub,
            c_lb,
            c_ub,
            idx_set_λ;
            show_output = :on,
            computing_global_lower_bound = :off,
        )
    elseif bound_impose == :on &&
           find_global_lower_bound_via_cholesky_lazy_constraint == :on
        bound_satisfaction_flag = bound_violation_checker_BnB_PEP(
            N,
            obj_val,
            λ_opt,
            τ_opt,
            Z_opt,
            L_cholesky_opt,
            b_opt,
            c_opt,
            Θ_opt,
            h_opt,
            λ_lb,
            λ_ub,
            τ_lb,
            τ_ub,
            Z_lb,
            Z_ub,
            L_cholesky_lb,
            L_cholesky_ub,
            Θ_lb,
            Θ_ub,
            h_lb,
            h_ub,
            b_lb,
            b_ub,
            c_lb,
            c_ub,
            idx_set_λ;
            show_output = :on,
            computing_global_lower_bound = :on,
        )
    end

    # time to return all the stored values
    # ------------------------------------

    return obj_val,
    λ_opt,
    τ_opt,
    Z_opt,
    L_cholesky_opt,
    b_opt,
    c_opt,
    Θ_opt,
    h_opt,
    idx_set_λ_opt_effective,
    idx_set_τ_opt_effective

end # function end
