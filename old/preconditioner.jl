include("./partial_cholesky.jl")

function get_preconditioner(A::Matrix, max_col::Int)
    L_11, D_1, perm = LDL_w_pivoting(A, allow_pivot=true, max_col=max_col)
    A = A[perm, perm]
    L_21 = A[max_col+1:end, 1:max_col] * inv(L_11') * inv(D_1)
    # TODO: only calculate the diagonal of Schur complement
    schur_comp = A[max_col+1:end, max_col+1:end] - A[max_col+1:end, 1:max_col] * inv(A[1:max_col, 1:max_col]) * A[max_col+1:end, 1:max_col]'

    # Checking decomposition
    decomp_mat_1 = [L_11 zeros(max_col, n - max_col); L_21 I(n - max_col)]
    decomp_mat_2 = [D_1 zeros(max_col, n - max_col); zeros(n - max_col, max_col) schur_comp]
    decomp_mat_3 = [L_11' L_21'; zeros(n - max_col, max_col) I(n - max_col)]
    A_recovered = decomp_mat_1 * decomp_mat_2 * decomp_mat_3
    println("Decomposition error: ", norm(A - A_recovered))

    # calculate preconditioner
    D_2 = Diagonal(diag(A[max_col+1:end, max_col+1:end])) - Diagonal(diag(L_21 * D_1 * L_21'))
    precon_mat_1 = [L_11 zeros(max_col, n - max_col); L_21 I(n - max_col)]
    precon_mat_2 = [D_1 zeros(max_col, n - max_col); zeros(n - max_col, max_col) D_2]
    precon_mat_3 = [L_11' L_21'; zeros(n - max_col, max_col) I(n - max_col)]
    P = precon_mat_1 * precon_mat_2 * precon_mat_3
    P_new = Hermitian(P)

    println("\nComparing preconditioner with permuted matrix:", norm(P - A))
    println("Preconditioner error: ", norm(P - P_new))
    P = P_new
    @assert isposdef(P)
    return P, perm
end

function generate_ill_conditioned_spd(n, cond_num)
    Q, _ = qr(randn(n, n))  # Random orthogonal matrix
    #=λ_max = 1.0=#
    #=λ_min = λ_max / cond_num=#
    λ_max = cond_num
    λ_min = 1.0
    Λ = Diagonal(range(λ_max, λ_min, length=n))  # Spread-out eigenvalues
    A = Q * Λ * Q'
    return A
end

n = 100 # Matrix size
max_col = 90 # Maximum column size
A = generate_ill_conditioned_spd(n, 1e5)

#=A = rand(n, n)=#
#=A = A' * A + I=#

P, perm = get_preconditioner(A, max_col)

println("\nComparing inverse of original matrix with preconditioner:")
println(norm(inv(A) - inv(P)))

println("\nComparing condition number of original matrix with preconditioner:")
println("  Original matrix, κ(A): ", cond(A))
println("  Preconditioned permuted matrix, κ(P⁻¹ΠᵀAΠ): ", cond(inv(P) * A[perm, perm]))
