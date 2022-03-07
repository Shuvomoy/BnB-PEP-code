
# using Weave
# cd("directory that contains the .jmd file") # directory that contains the .jmd file
# tangle("file_name.jmd", informat = "markdown")


## Comment about the notation in the code
# ---------------------------------------
# Every notation in the paper are kept the same in the code, except:
# (i) the subscript ⋆ (e.g., x_⋆) in the paper corresponds to index `-1` (e.g, `x[-1]`) in the code
# (ii) the stepsize ̄h $\bar{h}$ in the paper corresponds to $\alpha$ in the code

## Load the packages:
# ------------------
using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, KNITRO, OrderedCollections, BenchmarkTools

## Load the pivoted Cholesky finder
# ---------------------------------
include("code_to_compute_pivoted_cholesky.jl")

## If using Pardiso solver for solving linear system, please uncomment the following code:
# If using Pardiso,

# using Libdl

# Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)

# Libdl.dlopen("/home/gridsan/sdgupta/sdgupta_lib/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)

# Please use appropriate directory location for your version liblapack.so.3 and libomp.so.5


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


## Step size conversion functions
# -------------------------------

function compute_α_from_h(h, N, L)
    α = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for ℓ in 1:N
        for i in 0:ℓ-1
            if i==ℓ-1
                α[ℓ,i] = h[ℓ,ℓ-1]
            elseif i <= ℓ-2
                α[ℓ,i] = α[ℓ-1,i] + h[ℓ,i]
            end
        end
    end
    return α
end

function compute_h_from_α(α, N, L)
    h_new = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    for l in N:-1:1
        h_new[l,l-1] = α[l,l-1]
        for i in l-2:-1:0
            h_new[l,i] = α[l,i] - α[l-1,i]
        end
    end
    return h_new
end


# Options for these function are
# step_size_type = :Default => will create a last step of 1/(L) rest will be zero
# step_size_type = :Random => will create a random stepsize

function feasible_h_α_generator(N, L; step_size_type = :Default)

    # construct h
    # -----------
    h = OffsetArray(zeros(N, N), 1:N, 0:N-1)
    if step_size_type == :Default
        for i in 1:N
            h[i, i-1] =  1 # because we have defined h[i,i-1]/L in the algorithm, so declaring 1 will make the stepsizes equal to 1/L
            # this is the optimal stepsize in (0,1/L] due to https://arxiv.org/pdf/2104.05468.pdf
        end
    elseif step_size_type == :Random
        for i in 1:N
            h[i,i-1] = Uniform(0, 1)
        end
    end

    # find α from h
    # -------------

    α = compute_α_from_h(h, N, L)

    return h, α

end


## Data generator function
# ------------------------

# Option for this function:
# input_type == :stepsize_constant means we know the stepsize
# input_type == :stepsize_variable means the stepsize is a decision variable

function data_generator_function(N, α, L; input_type = :stepsize_constant)

    dim_𝐱 = N+2
    dim_𝐠 = N+2
    dim_𝐟 = N+1
    N_pts = N+2 # number of points corresponding to [x_⋆=x_{-1} x_0 ... x_N]

    𝐱_0 = e_i(dim_𝐱, 1)

    𝐱_star = zeros(dim_𝐱, 1)

    # initialize 𝐠 and 𝐟 vectors

    # 𝐠 = [𝐠_{-1}=𝐠_⋆ ∣ 𝐠_0 ∣ 𝐠_1 ∣... ∣ 𝐠_N]

    𝐠 =  OffsetArray(zeros(dim_𝐠, N_pts), 1:dim_𝐠, -1:N)

    # 𝐟  = [𝐟_{-1}=𝐟_⋆ ∣ 𝐟_0 ∣ 𝐟_1 ∣... ∣ 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_𝐟, N_pts), 1:dim_𝐟, -1:N)

    # construct 𝐠 vectors, note that 𝐠_⋆  is already constructed zero

    for k in 0:N
        𝐠[:,k] = e_i(dim_𝐠, k+2)
    end

    # construct 𝐟 vectors, note that 𝐟_⋆ is already constructed zero

    for k in 0:N
        𝐟[:,k] = e_i(dim_𝐟, k+1)
    end

    # time to define the 𝐱 vectors, which requires more care

    if input_type == :stepsize_constant

        # 𝐱 = [𝐱_{⋆} = 𝐱{-1} ∣ 𝐱_0 ∣ 𝐱_1 ∣ ... ∣ 𝐱_N]

        𝐱 = OffsetArray(zeros(dim_𝐱, N_pts), 1:dim_𝐱, -1:N)

        # define 𝐱_0 which corresponds to x_0

        𝐱[:,0] = 𝐱_0

        # construct part of 𝐱 corresponding to the x iterates: x_1, ..., x_N

        for i in 1:N
            𝐱[:,i] = 𝐱_0  - ( (1/L)*sum( α[i,j] * 𝐠[:,j] for j in 0:i-1) )
        end

    elseif input_type == :stepsize_variable

        # caution 💀: keep in mind that this matrix 𝐱 is not 0 indexed yet, so while constructing its elements, ensure to use the full formula for 𝐱_i

        𝐱 = [𝐱_star 𝐱_0]

        # construct part of 𝐱 corresponding to the x iterates: x_1, ..., x_N

        for i in 1:N
            𝐱_i = 𝐱_0  - ( (1/L)*sum( α[i,j] * 𝐠[:,j] for j in 0:i-1) )
            𝐱 = [𝐱 𝐱_i]
        end

        # make 𝐱 an offset array to make our life comfortable

        𝐱 = OffsetArray(𝐱, 1:dim_𝐱, -1:N)
    end

    # time to return

    return 𝐱, 𝐠, 𝐟

end


# Index set creator function for the dual variables λ, τ, η

struct i_j_idx # correspond to (i,j) pair, where i,j ∈ I_N_⋆
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
end

struct i_idx # correspond i in some set
    i::Int64
end

# We have 3 dual variables that operate over mulitple index sets
# λ_ij where i,j ∈ I_N_star,
# η_i  where i ∈ [0:N],
# τ_i where i  ∈ I_N_star,
# so we write a function to construct these index sets

function index_set_constructor_for_dual_vars_full(N)

    I_N_star = -1:N

    # construct the index set for λ
    idx_set_λ = i_j_idx[]
    for i in I_N_star
        for j in I_N_star
            if i!=j
                push!(idx_set_λ, i_j_idx(i,j))
            end
        end
    end

    # construct the index set for η and τ, both of which would have the same index set for the full case

    idx_set_τ = i_idx[]
    for i in 0:N #I_N_star
        push!(idx_set_τ, i_idx(i))
    end

    idx_set_η = i_idx[]
    for i in 0:N
        push!(idx_set_η, i_idx(i))
    end

    return idx_set_λ, idx_set_τ, idx_set_η

end

function effective_index_set_finder(λ, τ, η; ϵ_tol = 0.0005)

    # the variables λ, τ, η are of the type DenseAxisArray whose index set can be accessed using _.axes and data via _.data syntax

    idx_set_λ_current = (λ.axes)[1]

    idx_set_τ_current = (τ.axes)[1]

    idx_set_η_current = (η.axes)[1]

    idx_set_λ_effective = i_j_idx[]

    idx_set_τ_effective = i_idx[]

    idx_set_η_effective = i_idx[]

    # construct idx_set_λ_effective

    for i_j_λ in idx_set_λ_current
        if abs(λ[i_j_λ]) >= ϵ_tol # if λ[i,j] >= ϵ, where ϵ is our cut off for accepting nonzero
            push!(idx_set_λ_effective, i_j_λ)
        end
    end

    # construct idx_set_τ_effective

    for i_τ in idx_set_τ_current
        if abs(τ[i_τ]) >= ϵ_tol
            push!(idx_set_τ_effective, i_τ)
        end
    end

    # construct idx_set_η_effective

    for i_η in idx_set_η_current
        if abs(η[i_η]) >= ϵ_tol
            push!(idx_set_η_effective, i_η)
        end
    end

    return idx_set_λ_effective, idx_set_τ_effective, idx_set_η_effective

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


function index_set_zero_entries_dual_variables(N, idx_set_λ, idx_set_τ, idx_set_η)

    idx_set_zero_η = []

    # zero index set of λ: by solving the BnB-PEPs for N=1,…,5, we have found the pattern that λ[*,k]=0 and λ[k,*]=0

    idx_set_zero_λ_init = []

    for k in 0:N
        idx_set_zero_λ_init = [idx_set_zero_λ_init; i_j_idx(-1,k)]
    end

    for k in 0:N
        idx_set_zero_λ_init = [idx_set_zero_λ_init; i_j_idx(k, -1)]
    end

    idx_set_zero_λ = intersect(idx_set_zero_λ_init, idx_set_λ)


    # we have found the pattern τ[0]=…=τ[N-1] to be true by solving the BnB-PEPs for N=1,…,5

    idx_set_nz_τ = []
    idx_set_nz_τ = [idx_set_nz_τ ; i_idx(N)]

    idx_set_zero_τ = setdiff(idx_set_τ, idx_set_nz_τ)

    return idx_set_zero_λ, idx_set_zero_τ, idx_set_zero_η

end


A_tilde_mat(i,j,α,𝐠,𝐱) = ⊙(𝐠[:,i]+𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat(i,j,α,𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat(i,j,𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
a_vec(i,j,𝐟) = 𝐟[:, j] - 𝐟[:, i]


function solve_primal_with_known_stepsizes(N, R, L, α; show_output = :off)

    # generate the bold vectors
    # -------------------------

    𝐱, 𝐠, 𝐟 = data_generator_function(N, α, L; input_type = :stepsize_constant)

     # declare the model
    # -----------------
    model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # dimension of the decision variables G and Ft
    # --------------------------------------------
    I_N_star = -1:N
    dim_G = N+2
    dim_Ft = N+1

    # add the variables
    # -----------------
    @variable(model_primal_PEP_with_known_stepsizes, G[1:dim_G, 1:dim_G], PSD)

    @variable(model_primal_PEP_with_known_stepsizes, Ft[1:dim_Ft]) # For modeling advantage we have defined a column vector Ft = F^T which is transpose of the row matrix F in our model

    @variable(model_primal_PEP_with_known_stepsizes, t)

    # define objective
    # ----------------

    @objective(model_primal_PEP_with_known_stepsizes, Max, t)

     # define the constraints
    # ----------------------

    # hypograph constraint

    for i in 0:N
        @constraint(model_primal_PEP_with_known_stepsizes, t <= tr(G*C_mat(i,-1,𝐠)) )
    end

    # interpolation constraint

    for i in I_N_star
        for j in I_N_star
            if i != j
                @constraint(model_primal_PEP_with_known_stepsizes, Ft'*a_vec(i,j,𝐟) - ((L/4)*tr(G*B_mat(i,j,α,𝐱))) + (0.5*tr(G*A_tilde_mat(i,j,α,𝐠,𝐱))) + ( (1/(4*L))*tr(G*C_mat(i,j,𝐠)) ) <= 0 )
            end
        end
    end

    # lower bound condition

    for i in I_N_star
        @constraint(model_primal_PEP_with_known_stepsizes, Ft'*a_vec(i,-1,𝐟) + ((1/(2*L))*tr(G*C_mat(i,-1,𝐠))) <= 0)
    end


    # initial condition

    @constraint(model_primal_PEP_with_known_stepsizes, Ft'*a_vec(-1,0,𝐟)  <= R^2)

    # @constraint(model_primal_PEP_with_known_stepsizes, Ft'*a_vec(N,0,𝐟) <= R^2)

    # time to optimize
    # ----------------
    set_silent(model_primal_PEP_with_known_stepsizes)

    optimize!(model_primal_PEP_with_known_stepsizes)

    if termination_status(model_primal_PEP_with_known_stepsizes) != MOI.OPTIMAL
        @warn "primal PEP with known stepsizes did not reach optimality; termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
    end

    p_star = objective_value(model_primal_PEP_with_known_stepsizes)

    # @show "p_star = $p_star"

    G_star = value.(G)

    Ft_star = value.(Ft)

    return p_star, G_star, Ft_star

end


# N = 3
# R = 1
# L = 1
# h_feas, α_feas = feasible_h_α_generator(N, L; step_size_type = :Default)
# p_feas, G_feas, Ft_feas = solve_primal_with_known_stepsizes(N, R, L, α_feas; show_output = :off)


# In this function, the most important option is objective type:
# 0) :default will minimize ν*R^2 (this is the dual of the primal pep for a given stepsize)
# other options are
# 1) :find_sparse_sol, this will find a sparse solution given a particular stepsize and objective value upper bound
# 2) :find_M_λ , find the upper bound for the λ variables by maximizing ||λ||_1 for a given stepsize and particular objective value upper bound
# 3) :find_M_τ, find the upper bound for the τ variables by maximizing || τ ||_1
# 4) :find_M_η, find the upper bound for the η variables by maximizing ||η||_1
# 5) :find_M_Z, find the upper bound for the entries of the slack matrix Z, by maximizing tr(Z) for for a given stepsize and particular objective value upper bound

function solve_dual_PEP_with_known_stepsizes(N, R, L, α;
    show_output = :off,
    ϵ_tol_feas = 1e-6,
    objective_type = :default,
    obj_val_upper_bound = default_obj_val_upper_bound)

    # generate bold vectors
    # ---------------------

     𝐱, 𝐠, 𝐟 = data_generator_function(N, α, L; input_type = :stepsize_constant)

    # index set of points
    # -------------------

    I_N_star = -1:N
    dim_Z = N+2
    dim_𝐱 = N+2

    # define the model
    # ---------------
    # model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-10))

    model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

    # define the index set for the dual variables
    # -------------------------------------------
    idx_set_λ, idx_set_τ, idx_set_η = index_set_constructor_for_dual_vars_full(N)

    # define λ
    # --------
    @variable(model_dual_PEP_with_known_stepsizes, λ[idx_set_λ] >= 0)

    # define τ (the variable corresponding to lower bound)
    # -------
    @variable(model_dual_PEP_with_known_stepsizes, τ[idx_set_τ] >= 0)

    # define η
    # --------
    @variable(model_dual_PEP_with_known_stepsizes, η[idx_set_η] >= 0)

    # define ν
    # --------
    @variable(model_dual_PEP_with_known_stepsizes, ν >= 0)

    # define Z ⪰ 0
    # ------------

    @variable(model_dual_PEP_with_known_stepsizes, Z[1:dim_Z, 1:dim_Z], PSD)

    # add objective
    # -------------
    if objective_type == :default

        @info "🐒 Minimizing the usual performance measure"

        @objective(model_dual_PEP_with_known_stepsizes, Min,  ν*R^2)

    elseif objective_type == :find_sparse_sol

        @info "🐮 Finding a sparse dual solution given the objective value upper bound"

        # @objective(model_dual_PEP_with_known_stepsizes, Min, sum(η[i] for i in idx_set_η) + sum(τ[i] for i in idx_set_τ) + sum(λ[i_j] for i_j in idx_set_λ) )

        @objective(model_dual_PEP_with_known_stepsizes, Min, sum(η[i] for i in idx_set_η) + sum(τ[i] for i in idx_set_τ) + sum(λ[i_j] for i_j in idx_set_λ) )

        @constraint(model_dual_PEP_with_known_stepsizes, ν*R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_λ

        @info "[🐷 ] Finding upper bound on the entries of λ for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(λ[i_j] for i_j in idx_set_λ))

        @constraint(model_dual_PEP_with_known_stepsizes,  ν*R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_τ

        @info "[🐷 ] Finding upper bound on the entries of τ for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(τ[i] for i in idx_set_τ))

        @constraint(model_dual_PEP_with_known_stepsizes,  ν*R^2 <= obj_val_upper_bound)


    elseif objective_type == :find_M_η

        @info "[🐷 ] Finding upper bound on the entries of η for BnB-PEP"

        @objective(model_dual_PEP_with_known_stepsizes, Max, sum(η[i] for i in idx_set_η))

        @constraint(model_dual_PEP_with_known_stepsizes,  ν*R^2 <= obj_val_upper_bound)

    elseif objective_type == :find_M_Z

        @objective(model_dual_PEP_with_known_stepsizes, Max, tr(Z))

        @constraint(model_dual_PEP_with_known_stepsizes,  ν*R^2 <= obj_val_upper_bound)

    else

        @error "something is not right in objective type option setting"

    end

    # add linear constraint
    # ---------------------
    @constraint(model_dual_PEP_with_known_stepsizes,
    ( sum(λ[i_j_λ] * a_vec(i_j_λ.i, i_j_λ.j, 𝐟) for i_j_λ in idx_set_λ) +
    sum(τ[i_τ] * a_vec(i_τ.i,-1,𝐟) for i_τ in idx_set_τ) +
    ν*a_vec(-1,0,𝐟)
    )
    .== 0
    )

    # add LMI constraint
    # ----------------------------------------------

    @constraint(model_dual_PEP_with_known_stepsizes,
    ( -sum(η[i_η]*C_mat(i_η.i,-1,𝐠) for i_η in idx_set_η) )+
    ( (1/(2*L))*sum(τ[i_τ]*C_mat(i_τ.i,-1,𝐠) for i_τ in idx_set_τ) ) +
    (
    sum(λ[i_j_λ]*(
    ( (-L/4)*B_mat(i_j_λ.i, i_j_λ.j, α, 𝐱) ) +
    ( (0.5)*A_tilde_mat(i_j_λ.i, i_j_λ.j, α, 𝐠, 𝐱) ) +
    ( (1/(4*L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠) )
    ) for i_j_λ in idx_set_λ)
    )
    .==
    Z
    )


    # add sum(η) == 1 constraint

    @constraint(model_dual_PEP_with_known_stepsizes, sum(η[i_η] for i_η in idx_set_η) .== 1)

    # time to optimize
    # ----------------

    if show_output == :off
        set_silent(model_dual_PEP_with_known_stepsizes)
    end

    optimize!(model_dual_PEP_with_known_stepsizes)

    if termination_status(model_dual_PEP_with_known_stepsizes) != MOI.OPTIMAL
        @info "model_dual_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_dual_PEP_with_known_stepsizes)
    end

    # store the solutions and return
    # ------------------------------

    # store λ_opt

    λ_opt = value.(λ)

    # store η_opt

    η_opt = value.(η)

    # store τ_opt

    τ_opt = value.(τ)

    # store ν_opt

    ν_opt = value(ν)

    # store Z_opt

    Z_opt = value.(Z)

    # compute cholesky

    L_cholesky_opt =  compute_pivoted_cholesky_L_mat(Z_opt)

    if norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 1e-6
        @info "checking the norm bound"
        @warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt*L_cholesky_opt', Inf))"
    end

    # compute {Θ[i,j]}_{i,j} where i,j∈I_N_⋆
    Θ_opt = zeros(dim_𝐱, dim_𝐱, length(idx_set_λ))

    for ℓ in 1:length(idx_set_λ)
        i_j_λ = idx_set_λ[ℓ]
        Θ_opt[:,:,ℓ] = ⊙(𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j], 𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j])
        if norm(Θ_opt[:,:,ℓ] - B_mat(i_j_λ.i, i_j_λ.j, α, 𝐱), Inf) > 1e-6
            @error "something is not right in Θ"
            return
        end
    end

    # store objective

    # obj_val = objective_value(model_dual_PEP_with_known_stepsizes)

    # effective index sets for the dual variables λ, τ, η

    idx_set_λ_effective, idx_set_τ_effective, idx_set_η_effective = effective_index_set_finder(λ_opt, τ_opt, η_opt; ϵ_tol = 0.0005)

    # return all the stored values

    # store objective and other goodies

    ℓ_1_norm_λ = sum(λ_opt)

    ℓ_1_norm_η = sum(η_opt)

    ℓ_1_norm_τ = sum(τ_opt)

    tr_Z = tr(Z_opt)

    original_performance_measure = ν_opt*R^2

    return original_performance_measure, ℓ_1_norm_λ, ℓ_1_norm_τ, ℓ_1_norm_η, tr_Z, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α, idx_set_λ_effective, idx_set_τ_effective, idx_set_η_effective

end


# # will test the primal at the same time
# N = 5
# R = 1
# L = 1
# default_obj_val_upper_bound = 1e6
# h_feas, α_feas = feasible_h_α_generator(N, L; step_size_type = :Default)
#
# p_feas, G_feas, Ft_feas = solve_primal_with_known_stepsizes(N, R, L, α_feas; show_output = :off)
#
# original_performance_measure_feas, ℓ_1_norm_λ_feas, ℓ_1_norm_τ_feas, ℓ_1_norm_η_feas, tr_Z_feas, λ_feas, τ_feas, η_feas, ν_feas, Z_feas, L_cholesky_feas, Θ_feas, α_feas, idx_set_λ_feas_effective, idx_set_τ_feas_effective, idx_set_η_feas_effective = solve_dual_PEP_with_known_stepsizes(N, R, L, α_feas;
#     show_output = :off,
#     ϵ_tol_feas = 1e-6,
#     objective_type = :default,
#     obj_val_upper_bound = default_obj_val_upper_bound)
#
# @info "performance measure =$(p_feas)"
#
# @info "primal-dual gap = $(p_feas-original_performance_measure_feas)"



# We also provide a function to check if in a particular feasible solution, these bounds are violated

function bound_violation_checker_BnB_PEP(
    # input point
    # -----------
    d_star_sol, λ_sol, τ_sol, η_sol, ν_sol, Z_sol, L_cholesky_sol, Θ_sol,  α_sol,
    # input bounds
    # ------------
    λ_lb, λ_ub, τ_lb, τ_ub, η_lb, η_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, Θ_lb, Θ_ub, α_lb, α_ub;
    # options
    # -------
    show_output = :on,
    computing_global_lower_bound = :off
    )

    if show_output == :on
        @show [minimum(λ_sol)  maximum(λ_sol)  λ_ub]
        @show [minimum(τ_sol)  maximum(τ_sol)  τ_ub]
        @show [minimum(η_sol)  maximum(η_sol)  η_ub]
        @show [ν_lb ν_sol ν_ub]
        @show [Z_lb minimum(Z_sol)   maximum(Z_sol)  Z_ub]
        @show [L_cholesky_lb  minimum(L_cholesky_sol)  maximum(L_cholesky_sol) L_cholesky_ub]
        @show [α_lb minimum(α_sol) maximum(α_sol) α_ub]
        @show [Θ_lb minimum(Θ_sol) maximum(Θ_sol) Θ_ub]
    end

    # bound satisfaction flag

    bound_satisfaction_flag = 1

    # verify bound for λ
    if !(maximum(λ_sol) < λ_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found λ is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for τ
    if !(maximum(τ_sol) < τ_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found τ is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for η
    if !(maximum(η_sol) < η_ub + 1e-8) # lower bound is already encoded in the problem constraint
        @error "found η is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for ν: this is not necessary because this will be ensured due to our objective function being ν R^2
    # if !(maximum(ν_sol) <= ν_ub) # lower bound is already encoded in the problem constraint
    #     @error "found ν is violating the input bound"
    #     bound_satisfaction_flag = 0
    # end

    # verify bound for Z
    if !(Z_lb -  1e-8 < minimum(Z_sol) && maximum(Z_sol) < Z_ub + 1e-8)
        @error "found Z is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for Θ
    if !(Θ_lb -  1e-8 < minimum(Θ_sol) && maximum(Θ_sol) < Θ_ub + 1e-8)
        @error "found Z is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for L_cholesky
    if computing_global_lower_bound == :off
        if !(L_cholesky_lb -  1e-8 < minimum(L_cholesky_sol) && maximum(L_cholesky_sol) < L_cholesky_ub +  1e-8)
            @error "found L_cholesky is violating the input bound"
            bound_satisfaction_flag = 0
        end
    elseif computing_global_lower_bound == :on
        @info "no need to check bound on L_cholesky"
    end

    # verify bound for α
    if !(α_lb -  1e-8 < minimum(α_sol) && maximum(α_sol) < α_ub + 1e-8)
        @error "found α is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # # verify bound for objective value: this is not necessary again, this is already done in BnB_PEP_solver
    # if abs(obj_val_sol-BnB_PEP_cost_lb) <= ϵ_tol_sol
    #     @error "found objective value is violating the input bound"
    #     bound_satisfaction_flag = 0
    # end

    if bound_satisfaction_flag == 0
        @error "[💀 ] some bound is violated, increase the bound intervals"
    elseif bound_satisfaction_flag == 1
        @info "[😅 ] all bounds are satisfied by the input point, rejoice"
    end

    return bound_satisfaction_flag

end


function BnB_PEP_solver(
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
    maxCutCount=1e3, # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off, # wheather is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0, # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on, # wheather to polish the solution to get better precision, the other option is :off,
    M_Θ_factor = 100, # factor by which to magnify the internal M_Θ,
    impose_pattern = :off # other option is :on,  if it is turned on then we impose the pattern found by solving BnB-PEP from solving N=1,2,3,…,5
    )

    # Number of points
    # ----------------

    I_N_star = -1:N
    dim_Z = N+2
    dim_𝐱 = N+2
    𝐱_0 = e_i(dim_𝐱, 1)
    𝐱_star = zeros(dim_𝐱, 1)

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
        # set_optimizer_attribute(BnB_PEP_model, "FeasibilityTol", 1e-4)
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
                "feastol" => 1e-7,
                # "infeastol" => 1e-12,
                "opttol" => 1e-7,
                # "maxit" => 100000
                )
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

            # very high precision solution in IPOPT can be obtained via the following code, comment them out if high precision is not required

            set_optimizer_attribute(BnB_PEP_model, "constr_viol_tol", 1e-6)

            set_optimizer_attribute(BnB_PEP_model, "dual_inf_tol", 1e-6)

            set_optimizer_attribute(BnB_PEP_model, "compl_inf_tol", 1e-6)

            set_optimizer_attribute(BnB_PEP_model, "tol", 1e-10)

            set_optimizer_attribute(BnB_PEP_model, "max_iter", 5000)

            # if using the Pardiso for solving linear system, please uncomment the following code:

            # set_optimizer_attribute(BnB_PEP_model, "linear_solver", "pardiso")

        end
    end

    # ************************
    # define all the variables
    # ------------------------
    # ************************

    @info "[🎉 ] defining the variables"

    # define λ, τ, η
    # --------------

    if reduce_index_set_for_dual_variables == :off
        # define λ,  τ, η over the full index set
        idx_set_λ, idx_set_τ, idx_set_η = index_set_constructor_for_dual_vars_full(N)
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
        @variable(BnB_PEP_model, τ[idx_set_τ] >= 0)
        @variable(BnB_PEP_model, η[idx_set_η] >= 0)
    elseif reduce_index_set_for_dual_variables == :on
        # define λ over a reduced index set, idx_set_λ_ws_effective, which is the effective index set of λ_ws
        idx_set_λ = idx_set_λ_ws_effective
        idx_set_τ = idx_set_τ_ws_effective
        idx_set_η = idx_set_η_ws_effective
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
        @variable(BnB_PEP_model, τ[idx_set_τ] >= 0)
        @variable(BnB_PEP_model, η[idx_set_η] >= 0)
    elseif reduce_index_set_for_dual_variables == :for_warm_start_only
        # this :for_warm_start_only option is same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
        idx_set_λ, idx_set_τ, idx_set_η = index_set_constructor_for_dual_vars_full(N)
        idx_set_λ_ws = idx_set_λ_ws_effective
        idx_set_τ_ws = idx_set_τ_ws_effective
        idx_set_η_ws = idx_set_η_ws_effective
        @variable(BnB_PEP_model, λ[idx_set_λ] >= 0)
        @variable(BnB_PEP_model, τ[idx_set_τ] >= 0)
        @variable(BnB_PEP_model, η[idx_set_η] >= 0)
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

    # Define Θ[i,j] matrices such that for i,j ∈ I_N_star, we have Θ[i,j] = B_mat(i, j, α, 𝐱) = ⊙(𝐱[:,i] -𝐱[:,j], 𝐱[:,i] - 𝐱[:,j])

    Θ = BnB_PEP_model[:Θ] = reshape(
    hcat([
    @variable(BnB_PEP_model, [1:dim_𝐱, 1:dim_𝐱], Symmetric, base_name = "Θ[$i_j_λ]")
    for i_j_λ in idx_set_λ]...), dim_𝐱, dim_𝐱, length(idx_set_λ))

    # i.e, internally it will be defined as
    #     Θ[:,:,k] (julia) = Θ[i,j] (math) such that idx_set_λ[k] = (i,j)
    # so in Julia, we for warm-start define these variables as follows
    # i_j_λ = idx_set_λ[ℓ]
    # Θ_ws[:,:,ℓ] = ⊙(𝐱_ws[:,i_j_λ.i]-𝐱_ws[:,i_j_λ.j], 𝐱_ws[:,i_j_λ.i]-𝐱_ws[:,i_j_λ.j])
    # and similarly for the main variables


    # define the stepsize matrix α
    # ----------------------------
    if positive_step_size == :off
        @variable(BnB_PEP_model,  α[i = 1:N, j= 0:i-1])
    elseif positive_step_size == :on
        @variable(BnB_PEP_model, α[i = 1:N, j= 0:i-1] >= 0)
    end

    # [👲 ] insert warm-start values for all the variables
    # ----------------------------------------------------

    @info "[👲 ] warm-start values for all the variables"

    # warm start for λ, τ, η
    # ----------------------
    if reduce_index_set_for_dual_variables == :for_warm_start_only
        # warm start for λ
        for i_j_λ in idx_set_λ_ws
            set_start_value(λ[i_j_λ], λ_ws[i_j_λ])
        end
        for i_j_λ in setdiff(idx_set_λ, idx_set_λ_ws)
            set_start_value(λ[i_j_λ], 0.0)
        end
        # warm start for τ
        for i_j_τ in idx_set_τ_ws
            set_start_value(τ[i_j_τ], τ_ws[i_j_τ])
        end
        for i_j_τ in setdiff(idx_set_τ, idx_set_τ_ws)
            set_start_value(τ[i_j_τ], 0.0)
        end
        # warm start for η
        for i_j_η in idx_set_η_ws
            set_start_value(η[i_j_η], η_ws[i_j_η])
        end
        for i_j_η in setdiff(idx_set_η, idx_set_η_ws)
            set_start_value(η[i_j_η], 0.0)
        end
    else
        # warm start for λ
        for i_j_λ in idx_set_λ
            set_start_value(λ[i_j_λ], λ_ws[i_j_λ])
        end
        # warm start for τ
        for i_j_τ in idx_set_τ
            set_start_value(τ[i_j_τ], τ_ws[i_j_τ])
        end
        # warm start for η
        for i_j_η in idx_set_η
            set_start_value(η[i_j_η], η_ws[i_j_η])
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

    # warm start for Θ
    # ----------------

    # construct 𝐱_ws, 𝐠_ws, 𝐟_ws corresponding to α_ws
    𝐱_ws, 𝐠_ws, 𝐟_ws = data_generator_function(N, α_ws, L; input_type = :stepsize_constant)

    # construct Θ_ws step by step
    Θ_ws = zeros(dim_𝐱, dim_𝐱, length(idx_set_λ))

    for ℓ in 1:length(idx_set_λ)
        i_j_λ = idx_set_λ[ℓ]
        Θ_ws[:,:,ℓ] = ⊙(𝐱_ws[:,i_j_λ.i]-𝐱_ws[:,i_j_λ.j], 𝐱_ws[:,i_j_λ.i]-𝐱_ws[:,i_j_λ.j])
    end


    # setting the warm-start value for Θ_ws

    for ℓ in 1:length(idx_set_λ)
        i_j_λ = idx_set_λ[ℓ]
        set_start_value.(Θ[:,:,ℓ], Θ_ws[:,:,ℓ])
    end


    # compute M_Θ

    M_Θ = M_Θ_factor*max(1,maximum(abs.(Θ_ws)))

    # warm start for α
    # ----------------

    for i in 1:N
        for j in 0:i-1
            set_start_value(α[i,j], α_ws[i,j])
        end
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



    𝐱, 𝐠, 𝐟 = data_generator_function(N, α, L; input_type = :stepsize_variable)

    # *******************
    # add the constraints
    # *******************


    # add the linear constraint
    # -------------------------


    @info "[🎋 ] adding linear constraint"

    @constraint(BnB_PEP_model,
    ( sum(λ[i_j_λ] * a_vec(i_j_λ.i, i_j_λ.j, 𝐟) for i_j_λ in idx_set_λ) +
    sum(τ[i_τ] * a_vec(i_τ.i,-1,𝐟) for i_τ in idx_set_τ) +
    ν*a_vec(-1,0,𝐟)
    )
    .== 0
    )

    # add the constraint related to Θ
    # -------------------------------

    # add the constraints corresponding to Θ: (∀(i,j) ∈ idx_set_λ) Θ[:,:,position_of_(i,j)_in_idx_set_λ] ==  ⊙(𝐱[:,i] -𝐱[:,j], 𝐱[:,i] - 𝐱[:,j])
    # -----------------------------------------------------

    for ℓ in 1:length(idx_set_λ)
        i_j_λ = idx_set_λ[ℓ]
        @constraint(BnB_PEP_model, vectorize(
        Θ[:,:,ℓ] - ⊙(𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j], 𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j]),
        SymmetricMatrixShape(dim_𝐱)) .== 0)
    end

    # Okay, now let us add the LMI constraint:

    # modeling of the LMI constraint through vectorization (works same)
    # ------------------------------------
    # we are constructing term_1 + term_2 + term_3 + term_4 - term_5 == 0

    @info "[🎢 ] adding LMI constraint"

    if quadratic_equality_modeling == :exact

        # # direct modeling of the LMI constraint
        # ---------------------------------------

        @constraint(BnB_PEP_model,
        vectorize(
        # term 1: ∑ η[i] C[i,⋆] for i∈[0:N]
        ( -sum(η[i_η]*C_mat(i_η.i,-1,𝐠) for i_η in idx_set_η) ) +
        # term 2: (1/2L)*∑ τ[i] C[i,⋆] for i∈I_N_star
        ( (1/(2*L))*sum(τ[i_τ]*C_mat(i_τ.i,-1,𝐠) for i_τ in idx_set_τ) ) +
        # term 3: ∑ λ[i,j]*( {0.5* ̃A [i,j]} + {(1/4L) * C[i,j] }) for i,j ∈ I_N_star
        (
        sum(λ[i_j_λ]*(
        ( (0.5)*A_tilde_mat(i_j_λ.i, i_j_λ.j, α, 𝐠, 𝐱) ) +
        ( (1/(4*L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠) )
        ) for i_j_λ in idx_set_λ)
        ) +
        # term 4: (-L/4) * ∑ λ[i,j] Θ[:,:,position_of_(i,j)_in_idx_set_λ]
        ( (-L/4)*sum( λ[idx_set_λ[ℓ]]*Θ[:,:,ℓ] for ℓ in 1:length(idx_set_λ)) ) -
        # term 5: Z
        Z,
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
        # term 1: ∑ η[i] C[i,⋆] for i∈[0:N]
        ( -sum(η[i_η]*C_mat(i_η.i,-1,𝐠) for i_η in idx_set_η) ) +
        # term 2: (1/2L)*∑ τ[i] C[i,⋆] for i∈I_N_star
        ( (1/(2*L))*sum(τ[i_τ]*C_mat(i_τ.i,-1,𝐠) for i_τ in idx_set_τ) ) +
        # term 3: ∑ λ[i,j]*( {0.5* ̃A [i,j]} + {(1/4L) * C[i,j] }) for i,j ∈ I_N_star
        (
        sum(λ[i_j_λ]*(
        ( (0.5)*A_tilde_mat(i_j_λ.i, i_j_λ.j, α, 𝐠, 𝐱) ) +
        ( (1/(4*L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠) )
        ) for i_j_λ in idx_set_λ)
        ) +
        # term 4: (-L/4) * ∑ λ[i,j] Θ[:,:,position_of_(i,j)_in_idx_set_λ]
        ( (-L/4)*sum( λ[idx_set_λ[ℓ]]*Θ[:,:,ℓ] for ℓ in 1:length(idx_set_λ)) ) -
        # term 5: Z
        Z - ϵ_tol_feas*ones(dim_Z,dim_Z),
        SymmetricMatrixShape(dim_Z)
        ) .<= 0
        )

        # part 2: models
        # (dual related terms) - Z >= -ϵ_tol_feas*ones(dim_Z,dim_z)
        @constraint(BnB_PEP_model,
        vectorize(
        # term 1: ∑ η[i] C[i,⋆] for i∈[0:N]
        ( -sum(η[i_η]*C_mat(i_η.i,-1,𝐠) for i_η in idx_set_η) ) +
        # term 2: (1/2L)*∑ τ[i] C[i,⋆] for i∈I_N_star
        ( (1/(2*L))*sum(τ[i_τ]*C_mat(i_τ.i,-1,𝐠) for i_τ in idx_set_τ) ) +
        # term 3: ∑ λ[i,j]*( {0.5* ̃A [i,j]} + {(1/4L) * C[i,j] }) for i,j ∈ I_N_star
        (
        sum(λ[i_j_λ]*(
        ( (0.5)*A_tilde_mat(i_j_λ.i, i_j_λ.j, α, 𝐠, 𝐱) ) +
        ( (1/(4*L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠) )
        ) for i_j_λ in idx_set_λ)
        ) +
        # term 4: (-L/4) * ∑ λ[i,j] Θ[:,:,position_of_(i,j)_in_idx_set_λ]
        ( (-L/4)*sum( λ[idx_set_λ[ℓ]]*Θ[:,:,ℓ] for ℓ in 1:length(idx_set_λ)) ) -
        # term 5: Z
        Z + ϵ_tol_feas*ones(dim_Z,dim_Z),
        SymmetricMatrixShape(dim_Z)
        ) .>= 0
        )

    else

        @error "something is not right in LMI modeling"

        return

    end


    # add sum(η) == 1 constraint
    # --------------------------

    @constraint(BnB_PEP_model, sum(η[i_η] for i_η in idx_set_η) .== 1)



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
        τ_lb = 0
        τ_ub = M_τ
        η_lb = 0
        η_ub = M_η
        ν_lb = 0
        ν_ub = ν_ws
        Z_lb = -M_Z
        Z_ub = M_Z
        L_cholesky_lb = -M_L_cholesky
        L_cholesky_ub = M_L_cholesky
        α_lb = -M_α
        α_ub = M_α
        Θ_lb = -M_Θ
        Θ_ub = M_Θ

        # set bound for λ
        # ---------------
        # set_lower_bound.(λ, λ_lb): done in definition
        set_upper_bound.(λ, λ_ub)

        # set bound for τ
        # set_lower_bound.(τ, τ_lb): done in definition
        set_upper_bound.(τ, τ_ub)

        # set bound for η
        #  set_lower_bound.(η, η_lb): done in definition
        set_upper_bound.(η, η_ub)

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

        # set bound for Θ
        # ---------------
        set_lower_bound.(Θ, Θ_lb)
        set_upper_bound.(Θ, Θ_ub)

        # set bound for α
        # ---------------
        set_lower_bound.(α, α_lb)
        set_upper_bound.(α, α_ub)

    end

    # impose the effective index set of L_cholesky if reduce_index_set_for_L_cholesky  == :on and we are not computing a global lower bound
    # ----------------------------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off && reduce_index_set_for_L_cholesky == :on
        zis_Lc = zero_index_set_finder_L_cholesky(L_cholesky_ws; ϵ_tol = ϵ_tol_Cholesky)
        for k in 1:length(zis_Lc)
            fix(L_cholesky[CartesianIndex(zis_Lc[k])], 0; force = true)
        end
    end

    # impose pattern found by solving smaller values of N
    # ---------------------------------------------------

    if impose_pattern == :on

        @info "[🍊 ] imposing pattern"

        # set L_cholesky = 0

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off

            for i in 1:dim_Z
                for j in 1:dim_Z
                    fix(L_cholesky[i, j], 0.0; force = true)
                end
            end

        end

        # set Z = 0

        for i in 1:dim_Z
            for j in 1:dim_Z
                fix(Z[i, j], 0.0; force = true)
            end
        end

        # set τ[0]=τ[1]=…=τ[N-1]=0

        idx_set_zero_λ, idx_set_zero_τ, idx_set_zero_η = index_set_zero_entries_dual_variables(N, idx_set_λ, idx_set_τ, idx_set_η)

        for i_τ in idx_set_zero_τ
            fix(τ[i_τ], 0.0; force = true)
        end

        for i_j_λ in idx_set_zero_λ
            fix(λ[i_j_λ], 0.0; force = true)
        end

    end

    # Time to optimize and store


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

        # store τ_opt

        τ_opt = value.(τ)

        # store η_opt

        η_opt = value.(η)

        # store ν_opt

        ν_opt = value.(ν)

        # store α_opt

        α_opt = value.(α)

        # store Θ_opt

        Θ_opt = value.(Θ)

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

        obj_val, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α_opt, idx_set_λ_opt_effective, idx_set_τ_opt_effective, idx_set_η_opt_effective = d_star_ws, λ_ws, τ_ws, η_ws, ν_ws, Z_ws, L_cholesky_ws, Θ_ws, α_ws, idx_set_λ_ws_effective, idx_set_τ_ws_effective, idx_set_η_ws_effective

    end


    if polish_solution == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :off # note that if we are finding a global lower bound, then polishing the solution would not make sense

        @info "[🎣 ] polishing and sparsifying the solution"


        obj_val, ℓ_1_norm_λ_dummy, ℓ_1_norm_τ_dummy, ℓ_1_norm_η_dummy, tr_Z_dummy, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α_opt, idx_set_λ_effective_dummy, idx_set_τ_effective_dummy, idx_set_η_effective_dummy = solve_dual_PEP_with_known_stepsizes(N, R, L, α_opt;  show_output = :off,
        ϵ_tol_feas = 1e-6, objective_type = :default, obj_val_upper_bound = 1.0001*obj_val)

        obj_val_sparse, ℓ_1_norm_λ_dummy, ℓ_1_norm_τ_dummy, ℓ_1_norm_η_dummy, tr_Z_dummy, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α_opt, idx_set_λ_effective_dummy, idx_set_τ_effective_dummy, idx_set_η_effective_dummy = solve_dual_PEP_with_known_stepsizes(N, R, L, α_opt;  show_output = :off,
        ϵ_tol_feas = 1e-6, objective_type = :find_sparse_sol, obj_val_upper_bound = (1+(1e-6))*obj_val)

    end

    # find the effective index set of the found λ, τ, η

    idx_set_λ_opt_effective, idx_set_τ_opt_effective, idx_set_η_opt_effective = effective_index_set_finder(λ_opt, τ_opt, η_opt; ϵ_tol = 0.0005)

    @info "[🚧 ] for λ, only $(length(idx_set_λ_opt_effective)) components out of $(length(idx_set_λ)) are non-zero for the optimal solution"

    @info "[🚧 ] for τ, only $(length(idx_set_τ_opt_effective)) components out of $(length(idx_set_τ)) are non-zero for the optimal solution"

    @info "[🚧 ] for η, only $(length(idx_set_η_opt_effective)) components out of $(length(idx_set_η)) are non-zero for the optimal solution"

    @info "[💹 ] warm-start objective value = $d_star_ws, and objective value of found solution = $obj_val"

    # verify if any of the imposed bounds are violated

    if bound_impose == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :off
        bound_satisfaction_flag = bound_violation_checker_BnB_PEP(obj_val, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α_opt,
        λ_lb, λ_ub, τ_lb, τ_ub, η_lb, η_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, Θ_lb, Θ_ub, α_lb, α_ub;
        show_output = :on,
        computing_global_lower_bound = :off)
    elseif bound_impose == :on && find_global_lower_bound_via_cholesky_lazy_constraint == :on
        bound_satisfaction_flag = bound_violation_checker_BnB_PEP(obj_val, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α_opt,
        λ_lb, λ_ub, τ_lb, τ_ub, η_lb, η_ub, ν_lb, ν_ub, Z_lb, Z_ub, L_cholesky_lb, L_cholesky_ub, Θ_lb, Θ_ub, α_lb, α_ub;
        show_output = :on,
        computing_global_lower_bound = :on)
    end

    # time to return all the stored values

    return obj_val, λ_opt, τ_opt, η_opt, ν_opt, Z_opt, L_cholesky_opt, Θ_opt, α_opt, idx_set_λ_opt_effective, idx_set_τ_opt_effective, idx_set_η_opt_effective

end

