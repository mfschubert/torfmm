import torch
import torch.autograd


class CustomEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        """
        Performs eigenvalue decomposition using torch.linalg.eig.

        Saves the input matrix A, eigenvalues, and eigenvectors for the backward pass.

        Args:
            ctx: An object that can be used to stash information for backward
                 computation.
            A (torch.Tensor): A square matrix for which to compute eigenvalues and
                              eigenvectors.

        Returns:
            tuple: (eigenvalues, eigenvectors)
                   - eigenvalues (torch.Tensor): The eigenvalues of A.
                   - eigenvectors (torch.Tensor): The eigenvectors of A.
        """
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        ctx.save_for_backward(A, eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors):
        """
        Computes dL/dA* (gradient of Loss L w.r.t. A.conj()).
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

        Returns:
            torch.Tensor: dL/dA* (if A is complex) or dL/dA (if A is real).
        """
        A, eigenvalues, eigenvectors = ctx.saved_tensors

        V = eigenvectors
        try:
            V_inv = torch.linalg.inv(V)
        except torch.linalg.LinAlgError as e:
            print(f"Error inverting eigenvector matrix in CustomEig.backward: {e}.")
            # Returning zero gradients as a fallback. This might mask issues
            # if the matrix A was ill-conditioned for eigendecomposition.
            return torch.zeros_like(A, dtype=A.dtype)

        # Calculate F matrix
        eigenvalues_col = eigenvalues.unsqueeze(1)
        eigenvalues_row = eigenvalues.unsqueeze(0)
        eigenvalue_diffs = eigenvalues_row - eigenvalues_col

        eps_val = torch.finfo(eigenvalues.real.dtype).eps
        F_denom_nonzero = torch.where(
            eigenvalue_diffs == 0,
            eps_val * torch.ones_like(eigenvalue_diffs),
            eigenvalue_diffs,
        )
        F = 1.0 / F_denom_nonzero
        F.diagonal(dim1=-2, dim2=-1).fill_(0.0)

        # S = (V_inv @ dL/dV^*) * F
        S = (V_inv @ grad_eigenvectors.conj()) * F
        S_H = S.conj().T

        # term_in_paren = torch.diag(dL/dlambda^*) + S_H
        diag_term = torch.diag(grad_eigenvalues.conj())
        term_in_paren = diag_term + S_H

        # grad_A_star = V @ term_in_paren @ V_inv (This is dL/dA*)
        grad_A_star = V @ term_in_paren @ V_inv

        if not A.is_complex():
            if grad_A_star.imag.abs().max() > 1e-6:  # Heuristic threshold
                msg = (
                    "Warning: Input A was real, but grad_A (dL/dA*) has "
                    f"significant imaginary part: {grad_A_star.imag.abs().max().item()}"
                )
                print(msg)
            grad_A = grad_A_star.real
        else:
            grad_A = grad_A_star

        return grad_A.to(A.dtype)


def eig(A):
    """
    WARNING: The gradient computation for this function is currently
    experimental and known to be incorrect for many cases (especially
    eigenvectors and complex matrices). Use with extreme caution and
    do not rely on the computed gradients.

    Computes the eigenvalue decomposition of a square matrix A,
    providing support for automatic differentiation.

    This function wraps a custom `torch.autograd.Function` (`CustomEig`)
    to enable gradient computation for the eigendecomposition.

    Args:
        A (torch.Tensor): A square matrix (real or complex).
                          The matrix should be diagonalizable for meaningful
                          eigenvector gradients.

    Returns:
        tuple: (eigenvalues, eigenvectors)
               - eigenvalues (torch.Tensor): The eigenvalues of A. For real
                                           non-symmetric A, eigenvalues can be
                                           complex. Sorted in no particular order.
               - eigenvectors (torch.Tensor): The corresponding eigenvectors, where
                                            the k-th column eigenvectors[:, k] is
                                            the eigenvector for eigenvalues[k].
                                            Normalized to unit length.

    Gradient Computation:
    The backward pass (gradient computation) is implemented in `CustomEig.backward`.
    - For real `A`, it computes `dL/dA`.
    - For complex `A`, it computes `dL/dA*` (gradient with respect to the
      conjugate of `A`), following PyTorch's convention for complex autograd.
    The gradient formula is designed for matrices with distinct eigenvalues. While
    numerical stabilization (epsilon) is used for near-repeated eigenvalues,
    caution is advised for matrices with exactly repeated eigenvalues, as
    eigenvector gradients are not uniquely defined in such cases. The implemented
    formula attempts to follow standard approaches for complex variables.
    """
    return CustomEig.apply(A)
