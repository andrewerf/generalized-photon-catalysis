from enum import Enum
from numpy.typing import NDArray
import numpy as np
import sympy as sp

import mrmustard.lab as mr
import mrmustard.physics as mrph
import mrmustard.math as mrmath

from generalized_photon_catalysis.symplectic_utils import get_z_to_r, symplectic_completion, get_dist_to_symplectic



class NonGaussOp(Enum):
    Subtraction = 1
    Addition = 2

def get_single_form_preparation_circuit(v: NDArray[np.complex128]) -> tuple[mr.CircuitComponent, NonGaussOp]:
    """
    Returns (G, op) -- a gaussian transformation that prepares given linear form in creation and annihilation operators,
    when conjugated with either photon addition or photon subtraction, i.e. G^dagger p G, where p is a_0 or a^dagger_0.
    """
    M = v.shape[0] // 2
    v = v.copy()

    # this is either problem in my code, or in the mrMustard simulation, but empirically I figured out that the sign should be flipped here...
    # v[M:] = -v[M:]

    T = get_z_to_r(M)
    r = T @ v
    # print(r)
    S, s = symplectic_completion(r)

    # symplectic completion gives the action on the coefficients of the vector of quadratures
    # we need to get the symplectic transformation of the vector itself
    S = S.T
    print(f'Dist to symplectic: {get_dist_to_symplectic(S)}, op: {s}')
    
    return mr.Ggate(tuple(i for i in range(M)), S), NonGaussOp.Subtraction if s > 0 else NonGaussOp.Addition


def verify_single_form_preparation_circuit(P: mr.CircuitComponent, nG: NonGaussOp, v: NDArray[np.complex64], r: float = 0.001, theta: float = 0.001) -> float:
    M = v.shape[0] // 2
    cutoff = 5
    n = 1
    P_inv = P.inverse()
    ancilla = M # this is photon-addition (subtraction) ancillary mode
    target = 0

    s = mr.Number(ancilla, 0, cutoff)
    for k in range(M):
        s = s >> mr.Number(k, n, cutoff)

    if nG == NonGaussOp.Addition:
        nGop = mr.S2gate(modes=(target, ancilla), r=r)
    else:
        nGop = mr.BSgate(modes=(target, ancilla), theta=theta)
    G = P >> nGop >> P_inv

    l = s >> G
    l = l >> mr.Number(ancilla, 1, cutoff=cutoff).dual
    l = l.normalize()

    t = np.zeros((3, )*M, dtype=np.complex64)
    for k in range(M):
        idx = [1]*M
        idx[k] += 1
        t[tuple(idx)] = v[k]*np.sqrt(2)
    for k in range(M):
        idx = [1]*M
        idx[k] -= 1
        t[tuple(idx)] = v[M + k]
    t = mr.Ket.from_fock(tuple(i for i in range(M)), t).normalize()
    # print(state_to_string(state_array_to_dict(l.fock_array())))
    return float(np.real(t.fidelity(l)))


def get_product_form_preparation_circuit(
        V, r: float = 0.001, theta: float = 0.001,
        calc_ancillas=True
    ) -> tuple[mr.CircuitComponent, list[int]] | list[tuple[NonGaussOp | None, mr.CircuitComponent]]:
    """
    Returns (G, [anc]) -- a gaussian transformation and a set of ancillary modes, 
        that prepare a product of linear forms (rows of V) of creation and annihilation operators.
    The ancillary modes should be heralded on having 1 photon each, which is required by photon additions and subtractions in the circuit
    If calc_ancillas is false, just return the series of Gaussian operations, s.t. when interleaved with corresponding NG-op, the product is prepared.
        Note that the first element of the sequence doesn't have a non-gaussian component.
    """
    N = V.shape[0]
    M = V.shape[1] // 2

    Tz2r = get_z_to_r(M)
    Tr2z = np.linalg.inv(Tz2r)
    ops = []
    Gs = []
    Z_inv = np.eye(2*M)
    # correction still doesn't work!
    # Z_inv[0, 0] = np.cosh(r)
    # Z_inv[M, M] = 1 / np.cosh(r)

    for k in range(N):
        Gk, op = get_single_form_preparation_circuit(V[k])
        ops.append(op)
        Gs.append(Gk)

        SkT = Gk.symplectic.T
        SkT = Tr2z @ SkT @ Tz2r
        for i in range(k+1, N):
            V[i, :] = V[i, :] @ np.linalg.inv(SkT.T) @ Z_inv
    
    res = mr.Identity(tuple(range(M)))
    for Gk in Gs:
        res = res >> Gk

    Gs = Gs[::-1]
    ops = ops[::-1]

    if calc_ancillas:
        ancillas = list(range(M, M + N))
        for i in range(N):
            if ops[i] == NonGaussOp.Addition:
                R = mr.S2gate(modes=(0, ancillas[i]), r=r)
            else:
                R = mr.BSgate(modes=(0, ancillas[i]), theta=theta)

            res = res >> R >> Gs[i].inverse()

        return res, ancillas
    else:
        return [(None, res)] + list(map(lambda t: (t[0], t[1].inverse()), zip(ops, Gs)))


def mra_dag(mode: int):
    """Creation operator on `mode`, exact Bargmann form: z_out * exp(z_out w_in)."""
    A = np.array([[0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 0]], dtype=complex)   # vars: (z_out, w_in, y)
    b = np.zeros(3, dtype=complex)
    c = np.array([0, 1], dtype=complex)        # picks the 1st-order term in y
    return mr.Operation.from_bargmann([mode], [mode], (A, b, c), name=f"a_dag_{mode}")

def mra(mode: int):
    """Annihilation operator on `mode`: w_in * exp(z_out w_in)."""
    A = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=complex)   # y coupled to w_in instead
    b = np.zeros(3, dtype=complex)
    c = np.array([0, 1], dtype=complex)
    return mr.Operation.from_bargmann([mode], [mode], (A, b, c), name=f"a_{mode}")
