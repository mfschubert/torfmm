"""Unit tests for the custom eigendecomposition function."""

import unittest
from typing import Callable, Optional  # For type hints

import torch
import torch.autograd

from fmm_torch import eig


def get_real_symmetric_matrix(n: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a real symmetric matrix."""
    if seed is not None:
        torch.manual_seed(seed)
    a_matrix = torch.randn(n, n)
    return (a_matrix + a_matrix.T) / 2


def get_real_nonsymmetric_matrix(
    n: int, seed: Optional[int] = None, ensure_distinct_eigenvalues: bool = True,
) -> torch.Tensor:
    """Generate a real non-symmetric matrix."""
    if seed is not None:
        torch.manual_seed(seed)
    a_matrix = torch.randn(n, n)
    if ensure_distinct_eigenvalues:
        # A simple way to increase chances of distinct eigenvalues for small n
        # is to make it asymmetric and add some perturbation if needed.
        # For robust distinct eigenvalues, one might need to check explicitly
        # and regenerate, but for typical tests this often suffices.
        a_matrix = a_matrix - a_matrix.T  # Make it skew-symmetric first
        a_matrix = a_matrix + torch.diag(torch.randn(n) * n)  # Add random diagonal
        a_matrix = a_matrix + torch.randn(n, n) * 0.1  # Add small random perturbation
        # Check if eigenvalues are distinct enough
        vals = torch.linalg.eigvals(a_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.isclose(vals[i], vals[j], atol=1e-4):  # Heuristic
                    # Regenerate with a different seed component
                    return get_real_nonsymmetric_matrix(
                        n,
                        seed=(seed if seed else 0) + i + j + 1,
                        ensure_distinct_eigenvalues=True,
                    )
    return a_matrix


def get_complex_hermitian_matrix(
    n: int, seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a complex Hermitian matrix."""
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn(n, n)
    imag_part = torch.randn(n, n)
    a_matrix = torch.complex(real_part, imag_part)
    return (a_matrix + a_matrix.conj().T) / 2


def get_complex_symmetric_matrix(
    n: int, seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a complex symmetric (but not necessarily Hermitian) matrix."""
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn(n, n)
    imag_part = torch.randn(n, n)
    a_complex = torch.complex(real_part, imag_part)
    # Ensure symmetry
    a_symmetric = (a_complex + a_complex.T) / 2
    # Ensure it's not Hermitian by chance for testing purposes (unless n=1)
    if n > 1 and torch.allclose(a_symmetric, a_symmetric.conj().T):
        a_symmetric[0, 1] = a_symmetric[0, 1] + torch.complex(
            torch.tensor(0.5), torch.tensor(0.5),
        )
        # Ensure symmetry is maintained after the change
        a_symmetric[1, 0] = a_symmetric[0, 1]
    return a_symmetric


def get_complex_nonsymmetric_matrix(
    n: int, seed: Optional[int] = None, ensure_distinct_eigenvalues: bool = True,
) -> torch.Tensor:
    """Generate a complex non-symmetric matrix."""
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn(n, n)
    imag_part = torch.randn(n, n)
    a_matrix = torch.complex(real_part, imag_part)

    if ensure_distinct_eigenvalues:
        # Similar heuristic as for real non-symmetric
        a_matrix = a_matrix - a_matrix.T  # Make it skew-symmetric first (complex)
        a_matrix = a_matrix + torch.diag(
            torch.complex(torch.randn(n) * n, torch.randn(n) * n),
        )  # Add random complex diagonal
        a_matrix = (
            a_matrix + torch.complex(torch.randn(n, n), torch.randn(n, n)) * 0.1
        )  # Add small random perturbation
        # Check if eigenvalues are distinct enough
        vals = torch.linalg.eigvals(a_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.isclose(vals[i], vals[j], atol=1e-4):
                    return get_complex_nonsymmetric_matrix(
                        n,
                        seed=(seed if seed else 0) + i + j + 1,
                        ensure_distinct_eigenvalues=True,
                    )
    return a_matrix


class TestRealMatrixGradients(unittest.TestCase):
    """Tests for real matrix gradient computations."""

    def _test_grad(
        self,
        a_gen_func: Callable,
        matrix_type_str: str,
        n: int = 3,
        seed: Optional[int] = 0,
        ensure_distinct: bool = True,
        use_eigh_for_ref: bool = False,
    ) -> None:
        """Test gradients for a given real matrix type."""
        torch.manual_seed(seed)
        if (
            matrix_type_str == "real symmetric"
        ):  # get_real_symmetric_matrix doesn't take ensure_distinct
            a_matrix = a_gen_func(n, seed=seed)
        else:
            a_matrix = a_gen_func(
                n, seed=seed, ensure_distinct_eigenvalues=ensure_distinct,
            )

        a_matrix = a_matrix.to(torch.float64)  # For gradcheck precision
        a_matrix.requires_grad_(True)

        def func_for_eigenvalues(x_mat: torch.Tensor) -> torch.Tensor:
            # Loss that sums real and imaginary parts of eigenvalues
            eigenvalues, _ = eig(x_mat)
            return torch.sum(eigenvalues.real) + torch.sum(eigenvalues.imag)

        def func_for_eigenvectors(x_mat: torch.Tensor) -> torch.Tensor:
            # Loss that sums real and imaginary parts of eigenvectors
            _, eigenvectors = eig(x_mat)
            return torch.sum(eigenvectors.real) + torch.sum(eigenvectors.imag)

        try:
            torch.autograd.gradcheck(
                func_for_eigenvalues, a_matrix, eps=1e-6, atol=1e-4, nondet_tol=1e-5,
            )
        except RuntimeError as e:
            err_msg = f"eigenvalues failed for {matrix_type_str}"
            # T201: print(f"Gradcheck for {err_msg} matrix: {e}")
            self.fail(f"Gradcheck for {err_msg} with error: {e}")

        # `ensure_distinct` helps get unique eigenvectors (up to scale/phase).
        if ensure_distinct or use_eigh_for_ref:
            try:
                # Eigenvectors of real matrices can be complex.
                torch.autograd.gradcheck(
                    func_for_eigenvectors,
                    a_matrix,
                    eps=1e-6,
                    atol=1e-3,  # Higher atol for eigenvectors
                    nondet_tol=1e-4,
                )
            except RuntimeError as e:
                err_msg = f"eigenvectors failed for {matrix_type_str}"
                # T201: print(f"Gradcheck for {err_msg} matrix: {e}")
                self.fail(f"Gradcheck for {err_msg} with error: {e}")
        # else:
        # T201: print(
        #         f"Skipping eigenvector gradcheck for {matrix_type_str} due to "
        #         "potential non-uniqueness or higher sensitivity."
        #     )

    def test_real_symmetric_grads(self) -> None:
        """Test gradients for real symmetric matrices."""
        # T201: print("\nTesting Real Symmetric Matrix Gradients:")
        self._test_grad(
            get_real_symmetric_matrix,
            "real symmetric",
            n=3,
            seed=0,
            ensure_distinct=False,
            use_eigh_for_ref=True,
        )
        self._test_grad(
            get_real_symmetric_matrix,
            "real symmetric",
            n=4,
            seed=1,
            ensure_distinct=False,
            use_eigh_for_ref=True,
        )

        a_rep = torch.diag(torch.tensor([1.0, 1.0, 2.0, 3.0]))
        a_rep = a_rep.to(torch.float64)  #  Use float64 for gradcheck precision
        a_rep.requires_grad_(True)

        def func_for_eigenvalues_rep(x_mat: torch.Tensor) -> torch.Tensor:
            # Eigenvalues of a real symmetric matrix are always real.
            eigenvalues, _ = eig(x_mat)
            return torch.sum(eigenvalues.real)  # .imag will be zero

        try:
            torch.autograd.gradcheck(
                func_for_eigenvalues_rep, a_rep, eps=1e-7, atol=1e-5, nondet_tol=1e-6,
            )
        except RuntimeError as e:
            # T201: print("Gradcheck eigenvalues: real symm. (rep, diag) - FAILED.")
            # T201: print(f"Error: {e}")
            self.fail(
                f"Gradcheck real symm. (repeated, diag) failed. Error: {e}",
            )

        # Eigenvector gradcheck for repeated eigenvalues is often
        # problematic and typically skipped or handled with specific
        # projections. Skipping for this matrix.

    def test_real_nonsymmetric_grads(self) -> None:
        """Test gradients for real non-symmetric matrices."""
        # T201: print("\nTesting Real Non-Symmetric Matrix Gradients:")
        self._test_grad(
            get_real_nonsymmetric_matrix,
            "real non-symmetric (distinct)",
            n=3,
            seed=0,
            ensure_distinct=True,
        )
        self._test_grad(
            get_real_nonsymmetric_matrix,
            "real non-symmetric (distinct)",
            n=4,
            seed=1,
            ensure_distinct=True,
        )


class TestComplexMatrixGradients(unittest.TestCase):
    """Tests for complex matrix gradient computations."""

    def _test_grad_complex(
        self,
        a_gen_func: Callable,
        matrix_type_str: str,
        n: int = 3,
        seed: Optional[int] = 0,
        ensure_distinct: bool = True,
        use_eigh_for_ref: bool = False,
    ) -> None:
        """Test gradients for a given complex matrix type."""
        torch.manual_seed(seed)
        if (
            matrix_type_str == "complex hermitian"
            or matrix_type_str == "complex symmetric"
        ):
            a_matrix = a_gen_func(n, seed=seed)
        else:  # complex non-symmetric
            a_matrix = a_gen_func(
                n, seed=seed, ensure_distinct_eigenvalues=ensure_distinct,
            )

        a_matrix = a_matrix.to(torch.complex128)  # Use complex128 for gradcheck
        a_matrix.requires_grad_(True)

        def func_for_eigenvalues(x_mat: torch.Tensor) -> torch.Tensor:
            eigenvalues, _ = eig(x_mat)
            return torch.sum(eigenvalues.real) + torch.sum(eigenvalues.imag)

        def func_for_eigenvectors(x_mat: torch.Tensor) -> torch.Tensor:
            _, eigenvectors = eig(x_mat)
            return torch.sum(eigenvectors.real) + torch.sum(eigenvectors.imag)

        try:
            torch.autograd.gradcheck(
                func_for_eigenvalues, a_matrix, eps=1e-7, atol=1e-5, nondet_tol=1e-6,
            )
        except RuntimeError as e:
            err_msg = (
                f"eigenvalues FAILED for {matrix_type_str} matrix (complex)"
            )
            # T201: print(f"Gradcheck for {err_msg}: {e}")
            self.fail(f"Gradcheck for {err_msg}: {e}")

        if ensure_distinct or use_eigh_for_ref:
            try:
                torch.autograd.gradcheck(
                    func_for_eigenvectors,
                    a_matrix,
                    eps=1e-7,
                    atol=1e-4,  # Higher atol
                    nondet_tol=1e-5,
                )
            except RuntimeError as e:
                err_msg = (
                    f"eigenvectors FAILED for {matrix_type_str} matrix (complex)"
                )
                # T201: print(f"Gradcheck for {err_msg}: {e}")
                self.fail(f"Gradcheck for {err_msg}: {e}")
        # else:
            # T201: print(
            #     f"Skipping eigenvector gradcheck for {matrix_type_str} (complex) "
            #     "due to potential non-uniqueness."
            # )

    def test_complex_hermitian_grads(self) -> None:
        """Test gradients for complex Hermitian matrices."""
        # T201: print("\nTesting Complex Hermitian Matrix Gradients:")
        self._test_grad_complex(
            get_complex_hermitian_matrix,
            "complex hermitian",
            n=3,
            seed=10,
            ensure_distinct=False,  # Placeholder
            use_eigh_for_ref=True,
        )
        self._test_grad_complex(
            get_complex_hermitian_matrix,
            "complex hermitian",
            n=4,
            seed=11,
            ensure_distinct=False,  # Placeholder
            use_eigh_for_ref=True,
        )

        a_rep_c = torch.diag(
            torch.tensor([1.0, 1.0, 2.0, 3.0], dtype=torch.complex128),
        )
        a_rep_c.requires_grad_(True)

        def func_for_eigenvalues_rep_c(x_mat: torch.Tensor) -> torch.Tensor:
            eigenvalues, _ = eig(x_mat)
            return torch.sum(eigenvalues.real)  # .imag will be zero for Hermitian

        try:
            torch.autograd.gradcheck(
                func_for_eigenvalues_rep_c,
                a_rep_c,
                eps=1e-7,
                atol=1e-5,
                nondet_tol=1e-6,
            )
        except RuntimeError as e: # F841: e is now used in the f-string
            # T201: print(
            #     "Gradcheck for eigenvalues FAILED for complex hermitian "
            #     f"(repeated, diag): {e}"
            # )
            self.fail(
                f"Gradcheck complex hermitian (repeated, diag) failed. Error: {e}",
            )
        # Skipping eigenvector gradcheck for this case.

    def test_complex_symmetric_grads(self) -> None:
        """Test gradients for complex symmetric (non-Hermitian) matrices."""
        # T201: print("\nTesting Complex Symmetric (Non-Hermitian) Matrix Gradients:")
        self._test_grad_complex(
            get_complex_symmetric_matrix,
            "complex symmetric",
            n=3,
            seed=20,
            ensure_distinct=True,
        )
        self._test_grad_complex(
            get_complex_symmetric_matrix,
            "complex symmetric",
            n=4,
            seed=21,
            ensure_distinct=True,
        )

    def test_complex_nonsymmetric_grads(self) -> None:
        """Test gradients for complex non-symmetric matrices."""
        # T201: print("\nTesting Complex Non-Symmetric Matrix Gradients:")
        self._test_grad_complex(
            get_complex_nonsymmetric_matrix,
            "complex non-symmetric (distinct)",
            n=3,
            seed=30,
            ensure_distinct=True,
        )
        self._test_grad_complex(
            get_complex_nonsymmetric_matrix,
            "complex non-symmetric (distinct)",
            n=4,
            seed=31,
            ensure_distinct=True,
        )


if __name__ == "__main__":
    unittest.main()
