"""Custom Eigendecomposition Function with Autograd Support."""

from typing import Any, Tuple  # For type hints

import torch
import torch.autograd

# Type hints are added for clarity. Docstrings are improved.


class CustomEig(torch.autograd.Function):
    """Custom torch.autograd.Function for eigendecomposition with gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        a_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform eigenvalue decomposition using torch.linalg.eig.

        Saves the input matrix, eigenvalues, and eigenvectors for the backward pass.

        Args:
            ctx: An object that can be used to stash information for backward
                 computation.
            a_matrix (torch.Tensor): A square matrix for which to compute eigenvalues
                                     and eigenvectors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - eigenvalues (torch.Tensor): The eigenvalues of `a_matrix`.
            - eigenvectors (torch.Tensor): The eigenvectors of `a_matrix`.
        """
        eigenvalues, eigenvectors = torch.linalg.eig(a_matrix)
        ctx.save_for_backward(a_matrix, eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(
        ctx: Any,
        grad_eigenvalues: torch.Tensor,
        grad_eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dL/dA* (gradient of Loss L w.r.t. a_matrix.conj()).

        Formula used: grad_A_star = V @ (diag(dL/dlambda^*) + S_H) @ V_inv
        where:
            - V is the eigenvector matrix.
            - V_inv is the inverse of V.
            - dL/dlambda^* = grad_eigenvalues.conj() (as autograd provides dL/dlambda).
            - S_H = ((V_inv @ dL/dV^*) * F).conj().T
            - dL/dV^* = grad_eigenvectors.conj() (as autograd provides dL/dV).
            - F_ij = 1/(lambda_j - lambda_i) for i != j, else 0. lambda are eigenvalues.

        WARNING: This gradient implementation currently FAILS `gradcheck` for
        most eigenvector cases and all tested complex eigenvalue cases. The computed
        gradients are known to be incorrect. This function is experimental and
        not suitable for use until these gradient issues are resolved.

        Args:
            ctx: Context object with saved tensors.
            grad_eigenvalues (torch.Tensor): Gradient dL/dlambda from autograd.
            grad_eigenvectors (torch.Tensor): Gradient dL/dV from autograd.

        Returns
        -------
        torch.Tensor
            Gradient dL/dA* or dL/dA for complex or real `a_matrix`.
        """
        a_matrix, eigenvalues, eigenvectors = ctx.saved_tensors

        v_mat = eigenvectors
        try:
            v_inv_mat = torch.linalg.inv(v_mat)
        except torch.linalg.LinAlgError as e:
            # T201: Print is intentional for critical runtime error
            print(f"Error inverting eigenvector matrix in CustomEig.backward: {e}.")  # noqa: T201
            # Returning zero gradients as a fallback. This might mask issues
            # if the matrix a_matrix was ill-conditioned for eigendecomposition.
            return torch.zeros_like(a_matrix, dtype=a_matrix.dtype)

        # Calculate F matrix
        eigenvalues_col = eigenvalues.unsqueeze(1)
        eigenvalues_row = eigenvalues.unsqueeze(0)
        eigenvalue_diffs = eigenvalues_row - eigenvalues_col

        eps_val = torch.finfo(eigenvalues.real.dtype).eps
        f_denom_nonzero = torch.where(
            eigenvalue_diffs == 0,
            eps_val * torch.ones_like(eigenvalue_diffs),
            eigenvalue_diffs,
        )
        f_mat = 1.0 / f_denom_nonzero
        f_mat.diagonal(dim1=-2, dim2=-1).fill_(0.0)

        # S = (V_inv @ dL/dV^*) * F
        s_matrix = (v_inv_mat @ grad_eigenvectors.conj()) * f_mat
        s_h_matrix = s_matrix.conj().T

        # term_in_paren = torch.diag(dL/dlambda^*) + S_H
        diag_term = torch.diag(grad_eigenvalues.conj())
        term_in_paren = diag_term + s_h_matrix

        # grad_A_star = V @ term_in_paren @ V_inv (This is dL/dA*)
        grad_a_star = v_mat @ term_in_paren @ v_inv_mat

        if not a_matrix.is_complex():
            if grad_a_star.imag.abs().max() > 1e-6:  # Heuristic threshold
                msg = (
                    "Warning: Input A was real, but grad_A (dL/dA*) has "
                    f"significant imaginary part: {grad_a_star.imag.abs().max().item()}"
                )
                # T201: Print is intentional warning for user
                print(msg)  # noqa: T201
            grad_a = grad_a_star.real
        else:
            grad_a = grad_a_star

        return grad_a.to(a_matrix.dtype)


def eig(a_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the eigenvalue decomposition of a square matrix A.

    WARNING: The gradient computation for this function is currently
    experimental and known to be incorrect for many cases (especially
    eigenvectors and complex matrices). Use with extreme caution and
    do not rely on the computed gradients.

    This function wraps a custom `torch.autograd.Function` (`CustomEig`)
    to enable gradient computation for the eigendecomposition.

    Args:
        a_matrix (torch.Tensor): A square matrix (real or complex).
                                 The matrix should be diagonalizable for meaningful
                                 eigenvector gradients.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - eigenvalues (torch.Tensor): The eigenvalues of `a_matrix`. For real
                                      non-symmetric `a_matrix`, eigenvalues can be
                                      complex. Sorted in no particular order.
        - eigenvectors (torch.Tensor): The corresponding eigenvectors, where
                                       the k-th column `eigenvectors[:, k]` is
                                       the eigenvector for `eigenvalues[k]`.
                                       Normalized to unit length.

    Gradient Computation:
        The backward pass (gradient computation) is implemented in
        `CustomEig.backward`.
        - For real `a_matrix`, it computes `dL/dA`.
        - For complex `a_matrix`, it computes `dL/dA*` (gradient with respect to the
          conjugate of `a_matrix`), following PyTorch's convention for complex autograd.
        The gradient formula is designed for matrices with distinct eigenvalues. While
        numerical stabilization (epsilon) is used for near-repeated eigenvalues,
        caution is advised for matrices with exactly repeated eigenvalues, as
        eigenvector gradients are not uniquely defined in such cases. The implemented
        formula attempts to follow standard approaches for complex variables.
    """
    return CustomEig.apply(a_matrix)
