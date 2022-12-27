# Automated SDP relaxation of QCQPs in JuMP+Julia

# Here, we show how to construct automatic SDP relaxation of a nonconvex QCQP in `Julia+JuMP` 
# The code was written based on the tips from Benoît Legat (https://blegat.github.io/)

## Example QCQP

# Consider the following QCQP as an example:
# $$
# p^{\star}=\left(\begin{array}{ll}
# \textrm{minimize} & c_{1}^{\top}x+c_{2}^{\top}y+c_{3}\mathbf{tr}(Z)\\
# \textrm{subject to} & a_{1}xy^{\top}+A_{1}\left(\begin{bmatrix}x\\
# y
# \end{bmatrix}\right)=Z,\\
#  & xy^{\top}+A_{2}\left(\begin{bmatrix}x\\
# y
# \end{bmatrix}\right)\leq0,\\
#  & Z=PP^{\top},
# \end{array}\right)\quad(1)
# $$
# where the decision variables are: $x,y\in\mathbf{R}^{2},\; P,Z\in\mathbf{S}^{2}$. We have the bound information that the optimal value lies between $[-400,-200]$. Here $A_1, A_2$ are linear operators from $\mathbf{R}^4$ to $\mathbf{S}^2$. First, let us solve the problem to global optimality using `Gurobi+JuMP`.


# Load the packages
using JuMP, Gurobi, MosekTools, Mosek, LinearAlgebra, DiffOpt

# Problem data:
c_1 = [-3; -5]
c_2 = [2; -5]
c_3 = -1
a_1 = 4
A_1_op(x,y) = [3*x[1]+2*y[2] -x[2]+y[1]; -x[2]+y[1] -4*x[2]+y[2]]
A_2_op(x,y) = [2*x[1]-5*y[2] -3*x[2]+2*y[1]; -3*x[2]+2*y[1] 4*x[2]-7*y[2]]
x_lb = 0
x_ub = 5
y_lb = 0
y_ub = 10

## Solving the QCQP to global optimality

function QCQP_solver(c_1, c_2, c_3, a_1, x_lb, x_ub, y_lb, y_ub)

    nonlinear_model = Model(Gurobi.Optimizer)

    set_optimizer_attribute(nonlinear_model, "NonConvex", 2)

    @variable(nonlinear_model, x_lb <= x[1:2] <= x_ub)

    @variable(nonlinear_model, y_lb <= y[1:2] <= y_ub)

    @variable(nonlinear_model, Z[1:2, 1:2], Symmetric)

    @variable(nonlinear_model, P[1:2, 1:2])

    @objective( nonlinear_model, Min, c_1'*x + c_2'*y + c_3*tr(Z) )

    con_shape = SymmetricMatrixShape(2)

    QuadCon1 = @constraint(nonlinear_model, vectorize(a_1*x*transpose(y) + A_1_op(x,y) - Z, con_shape) .== 0 )

    QuadCon2 = @constraint(nonlinear_model, vectorize(x*transpose(y) + A_2_op(x,y), con_shape) .<= 0 )

    SDPCon3 = @constraint(nonlinear_model, vectorize(Z - P*P', con_shape) .== 0 )

    @constraint(nonlinear_model, c_1'*x + c_2'*y + c_3*tr(Z) >= -400 )

    @constraint(nonlinear_model, c_1'*x + c_2'*y + c_3*tr(Z) <= -200 )

    optimize!(nonlinear_model)

    p_star = objective_value(nonlinear_model)

    x_star = value.(x)

    y_star = value.(y)

    Z_star = value.(Z)

    @show x_star

    @show y_star

    @show Z_star

    @show p_star

    return x_star, y_star, Z_star, p_star

end



# Let us run the function.


# x_star, y_star, Z_star, p_star = QCQP_solver(c_1, c_2, c_3, a_1, x_lb, x_ub, y_lb, y_ub)


## Extracting the data to construct SDP relaxation

# We have three quadratic constraint arrays in (1). The third one is just $Z\succeq0$, so we can just model that. For the first and second, we want to convert them into standard form quadratic constraints.

# Defining $w=\textrm{vec}(x,y)$ we want to write the first quadratic constraint
# $$
# a_{1}xy^{\top}+A_{1}\left(\begin{bmatrix}x\\
# y
# \end{bmatrix}\right)=Z
# $$
# as
# $$
# w^{\top}P_{1}^{(k,\ell)}w+q_{1}^{(k,\ell)\top}w+r_{1}^{(k,\ell)}=Z_{k,\ell}\quad k\in[1:2],\;\ell\in[1:k],
# $$

#  and we want to write the second quadratic constraint
# $$
# xy^{\top}+A_{2}\left(\begin{bmatrix}x\\
# y
# \end{bmatrix}\right) \leq 0
# $$
# as
# $$
# w^{\top}P_{2}^{(k,\ell)}w+q_{2}^{(k,\ell)\top}w+r_{2}^{(k,\ell)} \leq 0, \quad k\in[1:2],\;\ell\in[1:k],
# $$
# where $P_{i}^{(k,\ell)},q_{i}^{(k,\ell)},r_{i}^{(k,\ell)}$ are problemdata that we are going to extract from the original problem.

# Let us create a `struct` that will help in providing the quadratic form data $P,q,r$.


using DiffOpt

struct Form
    P
    q
    r
end

using LinearAlgebra

LinearAlgebra.symmetric_type(::Type{Form}) = Form
LinearAlgebra.symmetric(f::Form, ::Symbol) = f
LinearAlgebra.transpose(f::Form) = f


# The following function creates the data to construct the SDP relaxation.


function standard_form_data_constructor(c_1, c_2, c_3, a_1, x_lb, x_ub, y_lb, y_ub)

    data_model = Model()

    @variable(data_model, x_lb <= x[1:2] <= x_ub)

    @variable(data_model, y_lb <= y[1:2] <= y_ub)

    @variable(data_model, Z[1:2, 1:2], Symmetric)

    w = Vector{VariableRef}[]

    push!(w, x)

    push!(w, y)

    # create the vectorized w

    w_vec = reduce(vcat, w)

    # Add the qcqp constraints that we want to take to standard form

    con_shape = SymmetricMatrixShape(2)

    QuadCon1 = @constraint(data_model, vectorize(a_1*x*transpose(y) + A_1_op(x,y) - Z, con_shape) .== 0 )

    QuadCon2 = @constraint(data_model, vectorize(x*transpose(y) + A_2_op(x,y), con_shape) .<= 0 )

    # create index map that will contain the indices of the original decision variables

    index_map = MOI.Utilities.IndexMap()

    for (i, var_ref) in enumerate(w_vec)
        index_map[JuMP.index(var_ref)] = MOI.VariableIndex(i)
    end

    n = length(index_map.var_map)

    remove = JuMP.index.(JuMP.vectorize(Z, con_shape)) # we have to ensure that we are not considering Z in w

    # The function `standard_form(con_ref::JuMP.ConstraintRef)` will take a `JuMP` quadratic function and convert it into a standard form quadratic constraint of the form $w^\top P w + q^\top w + r$ form.
    function standard_form_data(con_ref::JuMP.ConstraintRef)
        object = JuMP.constraint_object(con_ref)
        quad_func = JuMP.moi_function(object)
        quad_func = MOI.Utilities.substitute_variables(quad_func) do var
            F = MOI.ScalarAffineFunction{Float64}
            if var in remove
                return zero(F)
            else
                return convert(F, var)
            end
        end
        matrix = DiffOpt.sparse_array_representation(quad_func, n, index_map)
        r = matrix.constant - MOI.constant(JuMP.moi_set(object))
        P, q = Matrix(0.5*matrix.quadratic_terms), Vector(matrix.affine_terms)
        # P, q = 0.5*matrix.quadratic_terms, matrix.affine_terms
        return Form(P, q, r)
    end

    QuadCon1StdTerms = JuMP.reshape_vector(standard_form_data.(QuadCon1), con_shape)
    # For example,
    # QuadCon1StdTerms[1,1].P will give us corresponding P
    # QuadCon1StdTerms[1,1].q will give us corresponding q
    # QuadCon1StdTerms[1,1].r will give us corresponding r

    # QuadCon2StdTerms = map(QuadCon2) do con_refs
    #     JuMP.reshape_vector(standard_form_data.(QuadCon2), con_shape)
    # end

    QuadCon2StdTerms = JuMP.reshape_vector(standard_form_data.(QuadCon2), con_shape)

    return QuadCon1StdTerms, QuadCon2StdTerms

end



# Test the function.


QuadCon1StdTerms, QuadCon2StdTerms = standard_form_data_constructor(c_1, c_2, c_3, a_1, x_lb, x_ub, y_lb, y_ub)

# For example,
# QuadCon1StdTerms[1,1].P will give us corresponding P
# QuadCon1StdTerms[1,1].q will give us corresponding q
# QuadCon1StdTerms[1,1].r will give us corresponding r


## SDP relaxation model

# So, we have the following SDP relaxation of (2):
# $$
# p_{\textrm{SDP}}^{\star}=\left(\begin{array}{ll}
# \textrm{minimize} & c_{1}^{\top}x+c_{2}^{\top}y+c_{3}\mathbf{tr}(Z)\\
# \textrm{subject to} & \mathbf{tr}(P_{1}^{(k,\ell)}W)+q_{1}^{(k,\ell)\top}w+r_{1}^{(k,\ell)}=Z_{k,\ell},\quad k\in[1:2],\;\ell\in[1:k],\\
#  & \mathbf{tr}(P_{2}^{(k,\ell)}W)+q_{2}^{(k,\ell)\top}w+r_{2}^{(k,\ell)}\leq0,\quad k\in[1:2],\;\ell\in[1:k],\\
#  & Z\succeq0,\\
#  & \begin{bmatrix}W & w\\
# w^{\top} & 1
# \end{bmatrix}\succeq0,\\
#  & w=\textrm{vec}(x,y),\\
#  & W-l_{w}w^{\top}-wl_{w}^{\top}\geq-l_{w}l_{w}^{\top},\\
#  & W-l_{w}w{}^{\top}-wl_{w}^{\top}\geq-u_{w}u_{w}^{\top},\\
#  & W-l_{w}w^{\top}-wu_{w}^{\top}\geq-l_{w}u_{w}^{\top},\\
#  & l_{w}\leq w\leq u_{w},
# \end{array}\right)
# $$

# where the decision variables are $x,y\in\mathbf{R}^{2},\; Z\in\mathbf{S}^{2},\; W\in\mathbf{S}^{4}$. The last four constraints are called RLT cuts that are valid inequalities for $W=w w^\top$. For more details about the RLT cut and how the SDP relaxation is constructed in general, see

# > Anstreicher, Kurt M. "Semidefinite programming versus the reformulation-linearization technique for nonconvex quadratically constrained quadratic programming." *Journal of Global Optimization* 43.2 (2009): 471-484.   
# >
# > (Link: [http://www.optimization-online.org/DB_FILE/2007/05/1655.pdf](http://www.optimization-online.org/DB_FILE/2007/05/1655.pdf))

# Let us solve the SDP model step by step.


using Mosek, MosekTools

function SDP_relaxation_solver(c_1, c_2, c_3, a_1, x_lb, x_ub, y_lb, y_ub, QuadCon1StdTerms, QuadCon2StdTerms; big_M = 100, RLT_cut = :on)

    # Define the SDP model

    SDP_model = Model(optimizer_with_attributes(Mosek.Optimizer))

    @variable(SDP_model, x_lb <= x[1:2] <= x_ub)

    @variable(SDP_model, y_lb <= y[1:2] <= y_ub)

    @variable(SDP_model, Z[1:2, 1:2], PSD)

    w = Vector{VariableRef}[]

    push!(w, x)

    push!(w, y)

    # create the vectorized w

    w_vec = reduce(vcat, w)

    len_w = length(w_vec)


    @variable(SDP_model, W[1:len_w, 1:len_w], Symmetric)

    @objective(SDP_model, Min, c_1'*x + c_2'*y + c_3*tr(Z) )

    dim_Z = 2

    # SDP relaxation of qudratic constraint 1
    # ---------------------------------------

    for k in 1:dim_Z
        for ℓ in 1:k
            @constraint(SDP_model, tr(QuadCon1StdTerms[k,ℓ].P * W) + (QuadCon1StdTerms[k,ℓ].q)'*w_vec + (QuadCon1StdTerms[k,ℓ].r) == Z[k,ℓ] )
        end
    end

    # SDP relaxation of qudratic constraint 2
    # ---------------------------------------

    for k in 1:dim_Z
        for ℓ in 1:k
            @constraint(SDP_model, tr(QuadCon2StdTerms[k,ℓ].P * W) + (QuadCon2StdTerms[k,ℓ].q)'*w_vec + (QuadCon2StdTerms[k,ℓ].r) <= 0 )
        end
    end

    # Schur complement constraint
    # ---------------------------

    @constraint(SDP_model, schurCon, [W w_vec; w_vec' 1] in PSDCone())

    @constraint(SDP_model, c_1'*x + c_2'*y + c_3*tr(Z) >= -400 )

    @constraint(SDP_model, c_1'*x + c_2'*y + c_3*tr(Z) <= -200 )

    # Add RLT cuts
    # ------------

    if RLT_cut == :on

        # construct lower and upper bound vector for w_vec

        l_w = zeros(len_w)
        u_w = zeros(len_w)

        for i in 1:len_w

            if has_upper_bound(w_vec[i]) == true
                u_w[i] = upper_bound(w_vec[i])
            else
                u_w[i] = big_M
            end

            if has_lower_bound(w_vec[i]) == true
                l_w[i] = lower_bound(w_vec[i])
            else
                l_w[i] = -big_M
            end

        end

        # Add RLT cuts
        # ------------

        con_shape = SymmetricMatrixShape(2)

        @info "[🎠 ] Adding RLT cuts"

        @constraint(SDP_model, RLT_cut_1, vectorize( W - l_w*w_vec' - w_vec*l_w' + l_w*l_w', con_shape) .>= 0)

        @constraint(SDP_model, RLT_cut_2, vectorize(W - l_w*w_vec' - w_vec*l_w' + u_w*u_w', con_shape) .>= 0)

        @constraint(SDP_model, RLT_cut_3, vectorize(W - l_w*w_vec' - w_vec*u_w' + l_w*u_w', con_shape) .>= 0)

        @constraint(SDP_model, RLT_cut_4, l_w .<= w_vec)

        @constraint(SDP_model, RLT_cut_5, w_vec .<= u_w )

    end

    # Solve the optimization problem

    optimize!(SDP_model)

    objective_value(SDP_model)

    if termination_status(SDP_model) != MOI.OPTIMAL
        @info "[💀]"
        @error "model_dual_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(SDP_model)
    end

    x_star = value.(x)

    y_star = value.(y)

    Z_star = value.(Z)

    W_star = value.(W)

    p_star = objective_value(SDP_model)

    return x_star, y_star, Z_star, W_star, p_star

end

# Test the function.


# x_star_SDP, y_star_SDP, Z_star_SDP, W_star_SDP, p_star_SDP = SDP_relaxation_solver(c_1, c_2, c_3, a_1, x_lb, x_ub, y_lb, y_ub, QuadCon1StdTerms, QuadCon2StdTerms; big_M = 100, RLT_cut = :on)

