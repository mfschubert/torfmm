import torch
import unittest
from fmm_torch import eig # Assuming src/fmm_torch is in PYTHONPATH or installed
import torch.autograd

def get_real_symmetric_matrix(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn(n, n)
    return (A + A.T) / 2

def get_real_nonsymmetric_matrix(n, seed=None, ensure_distinct_eigenvalues=True):
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn(n, n)
    if ensure_distinct_eigenvalues:
        # A simple way to increase chances of distinct eigenvalues for small n
        # is to make it asymmetric and add some perturbation if needed.
        # For robust distinct eigenvalues, one might need to check explicitly
        # and regenerate, but for typical tests this often suffices.
        A = A - A.T # Make it skew-symmetric first
        A = A + torch.diag(torch.randn(n) * n) # Add random diagonal
        A = A + torch.randn(n,n) * 0.1 # Add small random perturbation
        # Check if eigenvalues are distinct enough
        vals = torch.linalg.eigvals(A)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.isclose(vals[i], vals[j], atol=1e-4): # Heuristic
                    # Regenerate with a different seed component
                    return get_real_nonsymmetric_matrix(n, seed=(seed if seed else 0) + i + j + 1, ensure_distinct_eigenvalues=True)
    return A

def get_complex_hermitian_matrix(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn(n, n)
    imag_part = torch.randn(n, n)
    A = torch.complex(real_part, imag_part)
    return (A + A.conj().T) / 2

def get_complex_symmetric_matrix(n, seed=None):
    # Symmetric but not necessarily Hermitian
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn(n, n)
    imag_part = torch.randn(n, n)
    A_complex = torch.complex(real_part, imag_part)
    # Ensure symmetry
    A_symmetric = (A_complex + A_complex.T) / 2
    # Ensure it's not Hermitian by chance for testing purposes (unless n=1)
    if n > 1 and torch.allclose(A_symmetric, A_symmetric.conj().T):
         A_symmetric[0,1] = A_symmetric[0,1] + torch.complex(torch.tensor(0.5), torch.tensor(0.5))
         # Ensure symmetry is maintained after the change
         A_symmetric[1,0] = A_symmetric[0,1]
    return A_symmetric

def get_complex_nonsymmetric_matrix(n, seed=None, ensure_distinct_eigenvalues=True):
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn(n, n)
    imag_part = torch.randn(n, n)
    A = torch.complex(real_part, imag_part)

    if ensure_distinct_eigenvalues:
        # Similar heuristic as for real non-symmetric
        A = A - A.T # Make it skew-symmetric first (complex)
        A = A + torch.diag(torch.complex(torch.randn(n) * n, torch.randn(n) * n)) # Add random complex diagonal
        A = A + torch.complex(torch.randn(n,n), torch.randn(n,n)) * 0.1 # Add small random perturbation
        # Check if eigenvalues are distinct enough
        vals = torch.linalg.eigvals(A)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.isclose(vals[i], vals[j], atol=1e-4):
                     return get_complex_nonsymmetric_matrix(n, seed=(seed if seed else 0) + i + j + 1, ensure_distinct_eigenvalues=True)
    return A

# Add imports for torch.testing if needed for tests later
# import torch.testing as tt

class TestRealMatrixGradients(unittest.TestCase):
    def _test_grad(self, A_gen_func, matrix_type_str, n=3, seed=0, ensure_distinct=True, use_eigh_for_ref=False):
        torch.manual_seed(seed)
        if matrix_type_str == "real symmetric": # get_real_symmetric_matrix doesn't take ensure_distinct
             A = A_gen_func(n, seed=seed)
        else:
             A = A_gen_func(n, seed=seed, ensure_distinct_eigenvalues=ensure_distinct)

        A.requires_grad_(True)

        # Loss that sums real and imaginary parts of eigenvalues
        func_for_eigenvalues = lambda x_mat: torch.sum(eig(x_mat)[0].real) + torch.sum(eig(x_mat)[0].imag)

        # Loss that sums real and imaginary parts of eigenvectors
        func_for_eigenvectors = lambda x_mat: torch.sum(eig(x_mat)[1].real) + torch.sum(eig(x_mat)[1].imag)

        # Gradcheck for eigenvalues
        try:
            # For real symmetric matrices using eigh as reference, eigenvalues are real, so .imag is 0.
            # For real non-symmetric, eigenvalues can be complex.
            torch.autograd.gradcheck(func_for_eigenvalues, A, eps=1e-6, atol=1e-4, nondet_tol=1e-5)
            print(f"Gradcheck for eigenvalues passed for {matrix_type_str} matrix.")
        except Exception as e:
            print(f"Gradcheck for eigenvalues FAILED for {matrix_type_str} matrix: {e}")
            self.fail(f"Gradcheck for eigenvalues failed for {matrix_type_str} with error: {e}")

        # Gradcheck for eigenvectors
        # Eigenvector gradcheck is generally more sensitive.
        # For symmetric matrices (especially with eigh), eigenvectors are orthogonal and well-behaved (if eigenvalues distinct).
        # For non-symmetric, eigenvectors might not be orthogonal, and left/right eigenvectors differ.
        # The provided backward function computes dL/dA (or dL/dA* for complex)
        # `ensure_distinct` helps in getting unique eigenvectors (up to scale and phase).
        if ensure_distinct or use_eigh_for_ref:
            try:
                # Eigenvectors of real matrices can be complex if eigenvalues are complex.
                torch.autograd.gradcheck(func_for_eigenvectors, A, eps=1e-6, atol=1e-3, nondet_tol=1e-4) # Higher atol for eigenvectors
                print(f"Gradcheck for eigenvectors passed for {matrix_type_str} matrix.")
            except Exception as e:
                print(f"Gradcheck for eigenvectors FAILED for {matrix_type_str} matrix: {e}")
                self.fail(f"Gradcheck for eigenvectors failed for {matrix_type_str} with error: {e}")
        else:
            print(f"Skipping eigenvector gradcheck for {matrix_type_str} due to potential non-uniqueness or higher sensitivity.")


    def test_real_symmetric_grads(self):
        print("\nTesting Real Symmetric Matrix Gradients:")
        # For symmetric, ensure_distinct is not an explicit param for the generator,
        # but eigh (implicitly used for reference in theory) handles repeated eigenvalues correctly.
        # Our custom eig uses torch.linalg.eig, which can also handle them.
        self._test_grad(get_real_symmetric_matrix, "real symmetric", n=3, seed=0, ensure_distinct=False, use_eigh_for_ref=True) # ensure_distinct=False as placeholder
        self._test_grad(get_real_symmetric_matrix, "real symmetric", n=4, seed=1, ensure_distinct=False, use_eigh_for_ref=True) # ensure_distinct=False as placeholder

        # Test with explicitly repeated eigenvalues for symmetric matrix
        A_rep = torch.diag(torch.tensor([1.0, 1.0, 2.0, 3.0]))
        A_rep = A_rep.to(torch.float64) # Use float64 for better precision in gradcheck
        A_rep.requires_grad_(True)

        # Eigenvalues of a real symmetric matrix are always real.
        func_for_eigenvalues_rep = lambda x_mat: torch.sum(eig(x_mat)[0].real) # .imag will be zero

        try:
            torch.autograd.gradcheck(func_for_eigenvalues_rep, A_rep, eps=1e-7, atol=1e-5, nondet_tol=1e-6, check_complex=True) # check_complex for safety
            print("Gradcheck for eigenvalues passed for real symmetric with repeated eigenvalues (diag).")
        except Exception as e:
            print(f"Gradcheck for eigenvalues FAILED for real symmetric with repeated eigenvalues (diag): {e}")
            self.fail(f"Gradcheck for eigenvalues failed for real symmetric with repeated eigenvalues (diag) with error: {e}")

        # Eigenvector gradcheck for repeated eigenvalues is often problematic and typically skipped or handled with specific projections.
        # Skipping eigenvector gradcheck for the matrix with repeated eigenvalues here.

    def test_real_nonsymmetric_grads(self):
        print("\nTesting Real Non-Symmetric Matrix Gradients:")
        self._test_grad(get_real_nonsymmetric_matrix, "real non-symmetric (distinct)", n=3, seed=0, ensure_distinct=True)
        self._test_grad(get_real_nonsymmetric_matrix, "real non-symmetric (distinct)", n=4, seed=1, ensure_distinct=True)
        # Test with (potentially, less controlled) non-distinct eigenvalues
        # self._test_grad(get_real_nonsymmetric_matrix, "real non-symmetric (potentially non-distinct)", n=4, seed=2, ensure_distinct=False)


class TestComplexMatrixGradients(unittest.TestCase):
    def _test_grad_complex(self, A_gen_func, matrix_type_str, n=3, seed=0, ensure_distinct=True, use_eigh_for_ref=False):
        torch.manual_seed(seed)
        # Adapt for complex hermitian which doesn't take ensure_distinct in its generator
        if matrix_type_str == "complex hermitian":
            A = A_gen_func(n, seed=seed)
        else:
            A = A_gen_func(n, seed=seed, ensure_distinct_eigenvalues=ensure_distinct)

        A = A.to(torch.complex128) # Use complex128 for better precision in gradcheck
        A.requires_grad_(True)

        # Loss that sums real and imaginary parts of eigenvalues
        func_for_eigenvalues = lambda x_mat: torch.sum(eig(x_mat)[0].real) + torch.sum(eig(x_mat)[0].imag)

        # Loss that sums real and imaginary parts of eigenvectors
        func_for_eigenvectors = lambda x_mat: torch.sum(eig(x_mat)[1].real) + torch.sum(eig(x_mat)[1].imag)

        # Gradcheck for eigenvalues
        try:
            torch.autograd.gradcheck(func_for_eigenvalues, A, eps=1e-7, atol=1e-5, nondet_tol=1e-6)
            print(f"Gradcheck for eigenvalues passed for {matrix_type_str} matrix (complex).")
        except Exception as e:
            print(f"Gradcheck for eigenvalues FAILED for {matrix_type_str} matrix (complex): {e}")
            self.fail(f"Gradcheck for eigenvalues failed for {matrix_type_str} (complex) with error: {e}")

        # Gradcheck for eigenvectors
        if ensure_distinct or use_eigh_for_ref:
            try:
                torch.autograd.gradcheck(func_for_eigenvectors, A, eps=1e-7, atol=1e-4, nondet_tol=1e-5) # Higher atol
                print(f"Gradcheck for eigenvectors passed for {matrix_type_str} matrix (complex).")
            except Exception as e:
                print(f"Gradcheck for eigenvectors FAILED for {matrix_type_str} matrix (complex): {e}")
                self.fail(f"Gradcheck for eigenvectors failed for {matrix_type_str} (complex) with error: {e}")
        else:
            print(f"Skipping eigenvector gradcheck for {matrix_type_str} (complex) due to potential non-uniqueness.")

    def test_complex_hermitian_grads(self):
        print("\nTesting Complex Hermitian Matrix Gradients:")
        # For Hermitian, eigh handles repeated eigenvalues correctly.
        # Our custom eig uses torch.linalg.eig.
        self._test_grad_complex(get_complex_hermitian_matrix, "complex hermitian", n=3, seed=10, ensure_distinct=False, use_eigh_for_ref=True) # ensure_distinct=False placeholder
        self._test_grad_complex(get_complex_hermitian_matrix, "complex hermitian", n=4, seed=11, ensure_distinct=False, use_eigh_for_ref=True) # ensure_distinct=False placeholder

        # Test with explicitly repeated eigenvalues for Hermitian matrix
        A_rep_c = torch.diag(torch.tensor([1.0, 1.0, 2.0, 3.0], dtype=torch.complex128))
        A_rep_c.requires_grad_(True)

        # Eigenvalues of a Hermitian matrix are always real.
        func_for_eigenvalues_rep_c = lambda x_mat: torch.sum(eig(x_mat)[0].real) # .imag will be zero

        try:
            torch.autograd.gradcheck(func_for_eigenvalues_rep_c, A_rep_c, eps=1e-7, atol=1e-5, nondet_tol=1e-6)
            print("Gradcheck for eigenvalues passed for complex hermitian with repeated eigenvalues (diag).")
        except Exception as e:
            print(f"Gradcheck for eigenvalues FAILED for complex hermitian with repeated eigenvalues (diag): {e}")
            self.fail(f"Gradcheck for eigenvalues failed for complex hermitian with repeated eigenvalues (diag) with error: {e}")
        # Skipping eigenvector gradcheck for this case.

    def test_complex_symmetric_grads(self):
        # Symmetric but not necessarily Hermitian
        print("\nTesting Complex Symmetric (Non-Hermitian) Matrix Gradients:")
        self._test_grad_complex(get_complex_symmetric_matrix, "complex symmetric", n=3, seed=20, ensure_distinct=True)
        self._test_grad_complex(get_complex_symmetric_matrix, "complex symmetric", n=4, seed=21, ensure_distinct=True)

    def test_complex_nonsymmetric_grads(self):
        print("\nTesting Complex Non-Symmetric Matrix Gradients:")
        self._test_grad_complex(get_complex_nonsymmetric_matrix, "complex non-symmetric (distinct)", n=3, seed=30, ensure_distinct=True)
        self._test_grad_complex(get_complex_nonsymmetric_matrix, "complex non-symmetric (distinct)", n=4, seed=31, ensure_distinct=True)
        # Optional: Test with ensure_distinct=False if the generator supports it well enough
        # self._test_grad_complex(get_complex_nonsymmetric_matrix, "complex non-symmetric (potentially non-distinct)", n=4, seed=32, ensure_distinct=False)


if __name__ == '__main__':
    unittest.main()
