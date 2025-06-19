[![Python CI](https://github.com/your-username/fourier-modal-method-torch/actions/workflows/python_ci.yml/badge.svg)](https://github.com/your-username/fourier-modal-method-torch/actions/workflows/python_ci.yml)

# Fourier Modal Method (FMM) in PyTorch

A PyTorch-based optical simulation tool implementing the Fourier Modal Method (also known as Rigorous Coupled Wave Analysis - RCWA).

## Project Structure

- `src/fmm_torch/`: Main Python package for the simulation tool.
- `.github/workflows/`: Contains GitHub Actions CI/CD workflows.
- `pyproject.toml`: Defines project metadata, dependencies, and tool configurations.
- `LICENSE`: Project license.
- `README.md`: This file.

## Development Setup

It's recommended to use a virtual environment for development.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fourier-modal-method-torch.git # TODO: Replace with actual URL
    cd fourier-modal-method-torch
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project uses `pyproject.toml` to manage dependencies. Install the project in editable mode with development dependencies:
    ```bash
    pip install -e .[dev]
    ```

### Running Linters and Checkers

This project uses several tools to ensure code quality:

-   **Ruff (Formatter & Linter):**
    ```bash
    # Check formatting
    ruff format --check .
    # Apply formatting
    ruff format .
    # Run linter
    ruff check .
    # Auto-fix linting issues (where possible)
    ruff check . --fix
    ```

-   **MyPy (Static Type Checker):**
    ```bash
    mypy src
    ```

-   **Pydocstyle (Docstring Style Checker):**
    ```bash
    pydocstyle src
    ```

These checks are also performed automatically by GitHub Actions when you push changes or open a pull request.