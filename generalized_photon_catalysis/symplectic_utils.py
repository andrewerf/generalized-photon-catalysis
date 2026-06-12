import numpy as np
from numpy.typing import NDArray

def get_symplectic_form(M: int) -> NDArray[np.float64]:
    return np.block([
        [np.zeros((M, M)), np.eye(M)],
        [-np.eye(M), np.zeros((M, M))]
    ])

def get_dist_to_symplectic(S: NDArray[np.float64]) -> np.float64:
    O = get_symplectic_form(S.shape[0] // 2)
    return np.linalg.norm(S.T @ O @ S - O)

def symplectic_gram_schmidt(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Given a and b s.t. <a, b> = 1, <a, a> = <a, b> = 0 (with <a, b> = a^T Omega b -- symplectic product),
    reconstruct the whole real symplectic matrix S, which has a and b as 1-st and (M+1)-th columns respectively
    """
    rng = np.random.default_rng(67)
    M = a.shape[0] // 2
    O = get_symplectic_form(M)
    def inner(x: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
        return np.float64(x.T @ O @ y)

    p = inner(a, b)
    a = a / np.sqrt(p)
    b = b / np.sqrt(p)

    alphas = [a]
    betas = [b]

    for i in range(1, M):
        at = rng.random(2*M)
        bt = rng.random(2*M)
        a = at.copy()
        b = bt.copy()
        for k in range(i):
            a += inner(at, alphas[k])*betas[k] - inner(at, betas[k])*alphas[k]
            b += inner(betas[k], bt)*alphas[k] - inner(alphas[k], bt)*betas[k]

        p = inner(a, b)
        a = a
        b = b / p

        alphas.append(a)
        betas.append(b)
    
    S = np.column_stack(tuple(alphas + betas))
    return S


def symplectic_completion(r: NDArray[np.complex128]) -> tuple[NDArray[np.float64], np.float64]:
    """
    Given complex vector r, finds symplectic matrix S s.t. S @ r0 = r,
    where r0 is either creation (r0 = [1, 0, ..., 0, -j, 0, ..., 0] / 2) or annihilation (r0 = [1, 0, ..., 0, j, 0, ..., 0] / 2) operator,
    for sign (-1) and 1 correspondingly
    """
    M = r.shape[0] // 2
    a = np.real(r)
    b = np.imag(r)
    O = get_symplectic_form(M)
    s = np.float64(a.T @ O @ b)
    if s < 0:
        b = -b

    return symplectic_gram_schmidt(a, b), np.sign(s)

def get_z_to_r(M: int) -> NDArray[np.complex128]:
    """
    Returns 2M matrix, that transforms vector from the space of creation and annihilation ops to the space of quadratures
    """
    T = np.block([
        [np.eye(M, dtype=np.complex128), -1j*np.eye(M)],
        [np.eye(M, dtype=np.complex128), 1j*np.eye(M)]
    ]) / 2
    return T.T

def normalize_symplectic_z(z: NDArray[np.complex128]) -> NDArray[np.complex128]:
    M = z.shape[0] // 2
    T = get_z_to_r(M)
    r = T @ z
    O = get_symplectic_form(M)
    s = abs(np.float64(np.real(r).T @ O @ np.imag(r)))
    r = (np.real(r) + np.imag(r)*1j) / np.sqrt(s)
    return np.linalg.inv(T) @ r
