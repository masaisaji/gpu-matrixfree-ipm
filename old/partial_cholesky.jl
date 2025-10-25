using LinearAlgebra, SparseArrays

function cholesky_no_pivoting(
    A::Matrix;
    max_col::Union{Int,Nothing}=nothing
)::Matrix
    """
    Partial or full Cholesky Decomposition without pivoting.
    If no max_col is provided, full Cholesky decomposition is performed.

    Reference:
      https://github.com/CristianCosci/Cholesky_Decomposition_python/blob/master/cholesky_factorization/cholensky.py#L109
    """
    n = size(A, 1)
    L = zeros(n, n)
    if max_col === nothing
        max_col = n
    end

    for col in 1:max_col
        L[col, col] = sqrt(A[col, col] - sum(L[col, 1:col-1] .^ 2))
        for row in col+1:n
            L[row, col] = (A[row, col] - sum(L[row, 1:col-1] .* L[col, 1:col-1])) / L[col, col]
        end
    end
    return L
end

function cholesky_w_pivoting(
    A::Matrix;
    allow_pivot::Bool=true,
    max_col::Union{Int,Nothing}=nothing
)::Tuple{Matrix,Vector{Int}}
    """
    Partial or full Cholesky Decomposition with complete column pivoting.
    If no max_col is provided, full Cholesky decomposition is performed.
    Given symetric positive definite matrix A, returns L such that 
    Πᵀ A Π= L Lᵀ where Π is the permutation matrix created by pivoting.
    Permutation matrix is returned as a vector instead of a full matrix.

    Reference:
      https://github.com/pyscf/pyscf/blob/master/pyscf/lib/scipy_helper.py
      No other method works
    """
    n = size(A, 1)
    L = zeros(n, n)
    D = diag(A)
    perm = collect(1:n)  # permutation tracking
    piv_val = diag(A)

    if !allow_pivot
        L = cholesky_no_pivoting(A)
        return L, perm
    end

    if max_col === nothing
        max_col = n
    end

    for col in 1:max_col
        # NOTE: only looking at the diagonal, needs proof?
        piv_idx = argmax(abs.(D[col:end])) + col - 1
        perm[col], perm[piv_idx] = perm[piv_idx], perm[col]
        piv_val[col], piv_val[piv_idx] = piv_val[piv_idx], piv_val[col]
        D[col], D[piv_idx] = D[piv_idx], D[col]
        L[[col, piv_idx], :] .= L[[piv_idx, col], :]
        L[col, col] = sqrt(D[col])
        L[col+1:end, col] = (A[perm[col+1:end], perm[col]] - L[col+1:end, 1:col] * vec(L[col, 1:col]')) ./ L[col, col]
        D[col+1:end] -= abs.(L[col+1:end, col]) .^ 2
    end
    return L, perm
end

function LDL_w_pivoting(
    A::Matrix;
    allow_pivot::Bool=true,
    max_col::Union{Int,Nothing}=nothing
)::Tuple{Matrix,Diagonal,Vector{Int}}
    """
    Partial or full LDL decomposition with complete column pivoting.
    If no max_col is provided, full LDL decomposition is performed.
    First calls Cholesky with complete pivoting, then convert it to LDL.
    Given pivoted Cholesky Πᵀ A Π = L_c  L_cᵀ, the conversion is
        L = L_c diag(L_c)⁻¹
        D = diag(L_c)²
    And the resulting LDL decomposition is
        Πᵀ A Π = L D Lᵀ.

    Retruns L, D, and the permutation vector.
    """
    n = size(A, 1)

    if max_col === nothing
        max_col = n
    end

    if !allow_pivot
        L_chol, perm = cholesky_w_pivoting(A, allow_pivot=allow_pivot)
    else
        L_chol, perm = cholesky_w_pivoting(A, allow_pivot=allow_pivot, max_col=max_col)
    end

    L_chol_diag = Diagonal(diag(L_chol)[1:max_col])
    L = L_chol[1:max_col, 1:max_col] * inv(L_chol_diag)
    D = L_chol_diag^2
    return L, D, perm
end

function test_partial_cholesky(n, max_col)
    println("Testing Partial Cholesky with n=$n, max_col=$max_col")

    # Generate a random sparse SPD matrix
    A = rand(n, n)
    A = A' * A + I

    println("\nComparing full Cholesky decomposition with no pivoting:")
    L_chol_ref = cholesky(A).L
    L_chol, _ = cholesky_w_pivoting(A, allow_pivot=false)
    println(norm(L_chol - L_chol_ref))

    println("\nComparing partial Cholesky decomposition with no pivoting:")
    L_chol_ref_partial = L_chol_ref[1:max_col, 1:max_col]
    L_chol_partial, _ = cholesky_w_pivoting(A, allow_pivot=false, max_col=max_col)
    println(norm(L_chol_partial[1:max_col, 1:max_col] - L_chol_ref_partial))

    println("\nComparing full Cholesky decomposition with pivoting:")
    piv_chol_obj = cholesky(A, RowMaximum())
    L_pivot_ref = piv_chol_obj.L
    L_pivot, perm_vec = cholesky_w_pivoting(A, allow_pivot=true)
    println(norm(L_pivot_ref - L_pivot))

    println("\nComparing partial Cholesky decomposition with pivoting:")
    L_pivot_ref_partial = L_pivot_ref[1:max_col, 1:max_col]
    L_pivot_partial, perm_vec_partial = cholesky_w_pivoting(A, allow_pivot=true, max_col=max_col)
    L_pivot_partial = L_pivot_partial[1:max_col, 1:max_col]
    println(norm(L_pivot_partial - L_pivot_ref_partial))
    #=println(norm(A[perm_vec_partial, perm_vec_partial][1:max_col, 1:max_col] - L_pivot_partial[1:max_col, 1:max_col] * L_pivot_partial[1:max_col, 1:max_col]'))=#

    println("\nComparing full LDL decomposition with no pivoting:")
    L, D, perm_vec = LDL_w_pivoting(A, allow_pivot=false)
    println(norm(L * D * L' - A))

    println("\nComparing partial LDL decomposition with no pivoting:")
    L, D, perm_vec = LDL_w_pivoting(A, allow_pivot=false, max_col=max_col)
    println(norm(L * D * L' - A[1:max_col, 1:max_col]))

    println("\nComparing full LDL decomposition with pivoting:")
    L, D, perm_vec = LDL_w_pivoting(A, allow_pivot=true)
    println(norm(L * D * L' - A[perm_vec, perm_vec]))

    println("\nComparing partial LDL decomposition with pivoting:")
    L, D, perm_vec = LDL_w_pivoting(A, allow_pivot=true, max_col=max_col)
    println(norm(L * D * L' - A[perm_vec, perm_vec][1:max_col, 1:max_col]))

    println("\nComparing matrix, before and after permutation:")
    L_pivot_permed = L_pivot[:, perm_vec]
    println(norm(A - L_pivot_permed * L_pivot_permed'))

    println("\nComparing inverse of matrix, before and after permutation:")
    println(norm(inv(A) - inv(L_pivot_permed * L_pivot_permed')))
end

function main()
    n = 50  # Matrix size
    max_col = 30  # Number of columns to compute
    test_partial_cholesky(n, max_col)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
