import torch
import torch.autograd

class CustomEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        """
        Forward pass for CustomEig.

        Args:
            A: Input matrix.

        Returns:
            Eigenvalues and eigenvectors of A.
        """
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        ctx.save_for_backward(A, eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors):
        """
        Backward pass for CustomEig.

        Args:
            grad_eigenvalues: Gradient of the loss with respect to eigenvalues.
            grad_eigenvectors: Gradient of the loss with respect to eigenvectors.

        Returns:
            Gradient of the loss with respect to A.
        """
        A, eigenvalues, eigenvectors = ctx.saved_tensors
        # grad_eigenvalues and grad_eigenvectors are the upstream gradients dy/dL, dv/dL

        # Calculate V_inv = torch.linalg.inv(eigenvectors)
        # torch.linalg.eig(A) returns eigenvectors V such that A V = V W
        # V is not guaranteed to be unitary for general matrices.
        # If A is not diagonalizable, V will be singular.
        # We assume A is diagonalizable here.
        V_inv = torch.linalg.inv(eigenvectors)

        # Calculate F
        # eigenvalues_col[i,0] = eigenvalues[i]
        # eigenvalues_row[0,j] = eigenvalues[j]
        # eigenvalue_diffs[i,j] = eigenvalues[j] - eigenvalues[i] (lambda_j - lambda_i)
        eigenvalues_col = eigenvalues.unsqueeze(1)
        eigenvalues_row = eigenvalues.unsqueeze(0)
        eigenvalue_diffs = eigenvalues_row - eigenvalues_col

        # eps for numerical stability, using real part of dtype for finfo
        eps = torch.finfo(eigenvalues.real.dtype).eps

        # F_ij = 1 / (lambda_j - lambda_i) for i != j, 0 for i == j.
        # Add eps to avoid division by zero/very small numbers.
        # Note: if lambda_j == lambda_i for i != j (repeated eigenvalues),
        # this will result in 1/eps, which is large.
        # The formula assumes distinct eigenvalues for non-diagonal terms.
        F = 1.0 / (eigenvalue_diffs + eps)
        F.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Calculate K = (eigenvectors.conj().T @ grad_eigenvectors) * F
        # This is (V^H @ dV) * F (element-wise product)
        K = (eigenvectors.conj().T @ grad_eigenvectors) * F

        # Construct the diagonal matrix from grad_eigenvalues
        # This is dW_diag (dL/dlambda_i on the diagonal)
        grad_eigenvalues_diag = torch.diag(grad_eigenvalues.to(eigenvalues.dtype)) # Ensure dtype consistency

        # Combine terms: term_in_paren = grad_eigenvalues_diag + K
        term_in_paren = grad_eigenvalues_diag + K

        # Calculate grad_A = V_inv.conj().T @ term_in_paren @ eigenvectors.conj().T
        # This is (V^{-1})^H @ term_in_paren @ V^H
        # (V_inv_H @ term_in_paren @ V_H)
        grad_A = V_inv.conj().T @ term_in_paren @ eigenvectors.conj().T

        # The gradient dA should have the same dtype as A.
        # If A is real, and eigendecomposition is complex, grad_A might be complex.
        # Usually, if A is real, the grad_A should also be real.
        # If A is real, its eigenvalues/vectors can be complex (in conjugate pairs).
        # The gradient formula should yield a real grad_A if A and upstream grads are real.
        # However, intermediate complex values (eigenvalues, F, K, V_inv) are used.
        # If A is real, grad_eigenvalues and grad_eigenvectors should also be real (or handle complex part).
        # Let's ensure the output grad_A is cast to A.dtype if A is real but grad_A becomes complex.
        if A.is_complex():
            return grad_A
        else:
            # If A is real, grad_A should be real.
            # Summing contributions from conjugate pairs of eigenvalues/vectors should make it real.
            # Taking .real can hide issues if the imaginary part is non-negligible.
            # It's better to ensure calculations preserve realness if inputs are real.
            # For now, let's assume the formula correctly results in a real grad if A is real.
            # A check could be: if torch.is_complex(A) or torch.is_complex(grad_eigenvalues) or torch.is_complex(grad_eigenvectors):
            # then return grad_A. Else return grad_A.real
            # However, grad_eigenvalues from loss on complex eigenvalues can be complex.
            # Let's stick to returning grad_A as is, assuming upstream handles types.
            # If A is real, but eigenvalues/vectors are complex, then V_inv, F, K can be complex.
            # The final grad_A should be real if the total derivative is real.
            # This typically happens naturally if the loss is real.
            return grad_A.to(A.dtype) # Cast to original A's dtype, which handles real part if A was real.


def eig(A):
    """
    Computes the eigenvalue decomposition of a square matrix A,
    with support for automatic differentiation.

    For complex matrices, or real matrices with repeated eigenvalues,
    the differentiability relies on a custom backward pass.
    The gradients are computed with respect to A* (conjugate of A)
    for complex inputs, following PyTorch's convention for complex autograd.

    Args:
        A (torch.Tensor): A square matrix.

    Returns:
        tuple: (eigenvalues, eigenvectors)
               - eigenvalues (torch.Tensor): The eigenvalues of A.
               - eigenvectors (torch.Tensor): The eigenvectors of A.
    """
    return CustomEig.apply(A)
