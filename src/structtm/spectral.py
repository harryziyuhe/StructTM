import numpy as np
import random
from scipy.sparse import csr_matrix, coo_matrix

def gram(mat: csr_matrix) -> np.ndarray:
    """
    Compute the Gram matrix from a sparse document-term matrix.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        Sparse document-term matrix

    Returns
    -------
    np.ndarray
        Gram matrix
    """
    # Compute row sums
    nd = mat.sum(axis=1).A1

    # Remove rows with fewer than 2 tokens
    valid_docs = nd >= 2
    mat = mat[valid_docs]
    nd = nd[valid_docs]

    # Compute divisor for normalization
    divisor = nd * (nd - 1)

    # Calculate the Gram matrix
    Htilde = mat.multiply(1 / np.sqrt(divisor[:, None]))
    Hhat = Htilde.sum(axis=0).A1
    Q = Htilde.T @ Htilde - np.diag(Hhat)

    return Q

def gram_rp(
        mat: csr_matrix,
        s: float = 0.05,
        p: int = 3000,
        d_group_size: int = 2000,
        verbose: bool = True
) -> np.ndarray:
    """
    Compute the Gram matrix using random projections.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        Sparse document-term matrix
    s : float, optional
        Sparsity level of the projection matrix, by default 0.05.
    p : int, optional
        Number of random projections, by default 3000.
    d_group_size : int, optional
        Size of document groups for processing, by default 2000.
    verbose: bool, optional
        If True, prints progress.

    Returns
    -------
    np.ndarray
        Gram matrix with random projections.
    """
    D, V = mat.shape
    n_items = int(s * p)

    # Construct projection matrix
    proj_rows, proj_cols, proj_values = [], [], []
    for i in range(V):
        indices = random.sample(range(p), n_items)
        values = np.random.choice([-1, 1], size = n_items)
        proj_rows.extend([i] * n_items)
        proj_cols.extend(indices)
        proj_values.extend(values)
    proj = csr_matrix((proj_values, (proj_rows, proj_cols)), shape=(V,p))

    # Compute the Gram matrix in blocks
    groups = int(np.ceil(D / d_group_size))
    Q = np.zeros((p, p))
    Qnorm = np.zeros(p)

    for i in range(groups):
        start = i * d_group_size
        end = min((i + 1) * d_group_size, D)
        sub_mat = mat[start:end]

        # Normalize by document length
        rsums = sub_mat.sum(axis=1).A1
        divisor = rsums * (rsums - 1)
        sub_matd = sub_mat.multiply(1 / divisor[:, None])

        Htilde = sub_matd.sum(axis=0).A1 * (rsums - 1 - sub_matd.sum(axis=0).A1)
        Qnorm += Htilde

        sub_mat = sub_mat.multiply(1 / np.sqrt(divisor[:, None]))

        # Update Q
        if i == 0:
            Q = sub_mat.T @ (sub_mat @ proj) - Htilde
        else:
            Q += sub_mat.T @ (sub_mat @ proj) - Htilde
        
        if verbose:
            print(".", end = "")
    
    if verbose:
        print()
    
    Q = Q / Qnorm
    return Q

def fast_anchor(
        Qbar: np.ndarray,
        K: int,
        verbose: bool = True
) -> list:
    """
    Identify anchor words using the Gram-Schmidt process.

    Parameters
    ----------
    Qbar : np.ndarray
        Row-normalized Gram matrix.
    K : int
        Number of anchor words to find.
    verbose : bool, optional
        If True, prints progress.

    Returns
    -------
    list
        Indices of the anchor words.
    """
    basis = []
    row_squared_sums = np.sum(Qbar**2, axis=1)

    for i in range(K):
        # Find the row with the maximum squared sum
        anchor = np.argmax(row_squared_sums)
        basis.append(anchor)

        max_val = row_squared_sums[anchor]
        normalizer = 1 / np.sqrt(max_val)

        # Normalize the anchor row
        Qbar[anchor, :] *= normalizer

        # Project other rows onto the anchor and subtract
        inner_products = Qbar @ Qbar[anchor, :]
        projection = np.outer(inner_products, Qbar[anchor, :])
        projection[basis, :] = 0
        Qbar -= projection

        # Update row sqaured sums, excluding anchors
        row_squared_sums = np.sum(Qbar**2, axis=1)
        row_squared_sums[basis] = 0

        if verbose:
            print(".", end="")
    if verbose:
        print()
    
    return basis

def recover_l2(
        Qbar: np.ndarray,
        anchor: list,
        wprob: np.ndarray,
        verbose: bool = True
) -> np.ndarray:
    """
    Recover topic-word distributions using the RecoverL2 procedure.
    
    Parameters
    ----------
    Qbar : np.ndarray
        Row-normalized Gram matrix.
    anchor : list
        Indices of the anchor words.
    wprob : np.ndarray
        Empirical word probabilities.
    verbose : bool, optional
        If True, prints progress

    Returns
    -------
    np.ndarray
        Recovered topic-word distributions.
    """
    K = len(anchor)
    X = Qbar[anchor, :]
    XtX = X @ X.T

    # Initialize results
    weights = np.zeros((Qbar.shape[0], K))

    for i in range(Qbar.shape[0]):
        if i in anchor:
            weights[i, anchor.index(i)] = 1
        else:
            y = Qbar[i, :]
            solution = np.linalg.solve(XtX, X @ y)
            solution[solution < 0] = 0
            weights[i] = solution / solution.sum()
        
        if verbose and i % 100 == 0:
            print(".", end = "")
    if verbose:
        print()

    # Compute beta
    A = weights * wprob[:, None]
    A /= A.sum(axis=0)
    return A