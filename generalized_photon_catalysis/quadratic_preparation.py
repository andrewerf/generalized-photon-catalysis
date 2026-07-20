import numpy as np
from numpy.typing import NDArray
from photon_catalysis.quadratic_preparation import get_e2_mat, qpoly2mat, mat_sqrt_np
import torch
import torch.nn as nn
import sympy as sp


class ComplexOrthogonal(nn.Module):
    """
    Parametrizes a complex orthogonal matrix via the Cayley map:
        O = (I - A)(I + A)^{-1}
    where A is a skew-symmetric complex matrix (A^T = -A).
    Covers the connected component reachable from I (det = +1 generically).
    """
    def __init__(self, n: int):
        super().__init__()
        # A = S - S^T will be skew-symmetric; store upper triangle as free params
        # Two real tensors for real/imag parts of the upper triangle + diagonal=0
        self.S_re = nn.Parameter(torch.randn(n, n) * 0.01)
        self.S_im = nn.Parameter(torch.randn(n, n) * 0.01)

    def forward(self) -> torch.Tensor:
        S = torch.complex(self.S_re, self.S_im)
        A = S - S.T                          # skew-symmetric: A^T = -A
        n = A.shape[0]
        I = torch.eye(n, dtype=A.dtype, device=A.device)
        O = torch.linalg.solve(I + A, I - A) # (I+A)^{-1}(I-A)  [left Cayley]
        return O


def minimize_trace_complex_orthogonal(
    B: torch.Tensor,
    K: torch.Tensor,
    n_steps: int = 5000,
    lr: float = 1e-3,
) -> torch.Tensor:
    """
    Minimizes tr(O B O^T K) over complex orthogonal matrices (O^T O = I).
    B, K: complex square tensors of shape (n, n).
    """
    model = ComplexOrthogonal(n=B.shape[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    for step in range(n_steps):
        optimizer.zero_grad()
        O = model()
        loss = torch.abs(torch.trace(O @ B @ O.T @ K))
        # Loss is complex in general; minimize the real part
        # (imaginary part of a trace of a product of complex matrices
        #  need not be zero, but the minimum of the real part is well-defined)
        loss.real.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            with torch.no_grad():
                orth_err = torch.linalg.norm(O.T @ O - torch.eye(B.shape[0], dtype=O.dtype)).item()
            print(f"step {step:4d}  loss = {loss.real.item():.6f}  |O^T O - I| = {orth_err:.2e}")

    return model().detach()



def generalized_e2_preparation(
        Qp: NDArray[np.float64],
        n_steps: int = 5000,
        lr: float = 1e-3,
    ) -> NDArray[np.float64]:
    """
    Generates preparation scheme for the target degree 2 polynomial in creation and annihilation operators, homogeneous in the normal order.

    :param Qp: Quadratic-form matrix
    """
    assert Qp.shape[0] == Qp.shape[1]
    assert Qp.shape[0] % 2 == 0

    M = Qp.shape[0] // 2
    R = np.block([
        [np.zeros((M, M)), np.zeros((M, M))],
        [np.eye(M, M), np.zeros((M, M))]
    ])
    Ep = np.asarray(get_e2_mat(2*M), dtype=np.float64)
    E = np.triu(Ep)

    sqrt_Qp = mat_sqrt_np(Qp)
    sqrt_Ep = mat_sqrt_np(Ep)
    inv_sqrt_Ep = np.linalg.inv(sqrt_Ep)

    G = sqrt_Qp @ R @ sqrt_Qp
    K = inv_sqrt_Ep @ E.T @ inv_sqrt_Ep

    O = minimize_trace_complex_orthogonal(
        torch.tensor(G, dtype=torch.complex64),
        torch.tensor(K, dtype=torch.complex64),
        n_steps,
        lr
        ).detach().numpy()
    
    return inv_sqrt_Ep @ O @ sqrt_Qp



def verify_generalized_e2_preparation(V: NDArray[np.float64]) -> sp.Expr:
    a1, a1d = sp.symbols('a_1 a_1^\\dagger', commutative=False)
    a2, a2d = sp.symbols('a_2 a_2^\\dagger', commutative=False)
    a3, a3d = sp.symbols('a_3 a_3^\\dagger', commutative=False)
    all_ad = [a1d, a2d, a3d]
    all_a = [a1, a2, a3]

    def to_nf(expr): 
        return expr.subs({
            a2*a1: a1*a2,
            a2*a1d: a1d*a2,
            a2d*a1: a1*a2d,
            a2d*a1d: a1d*a2d,

            a3*a1: a1*a3,
            a3*a1d: a1d*a3,
            a3d*a1: a1*a3d,
            a3d*a1d: a1d*a3d,

            a3*a2: a2*a3,
            a3*a2d: a2d*a3,
            a3d*a2: a2*a3d,
            a3d*a2d: a2d*a3d,

            a1*a1d: (1 + a1d*a1),
            a2*a2d: (1 + a2d*a2),
            a3*a3d: (1 + a3d*a3),
        })
    def esp2(xs):
        r = 0
        for i in range(len(xs) - 1):
            for j in range(i + 1, len(xs)):
                r += xs[i]*xs[j] 
        return r

    M = V.shape[1] // 2
    z = sp.Matrix([all_ad[i] for i in range(M)] + [all_a[i] for i in range(M)])

    res = to_nf(esp2(V @ z).expand()).simplify()
    return res 

