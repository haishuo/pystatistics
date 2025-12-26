#!/usr/bin/env python3
"""
Generate test fixtures for regression validation.

Creates CSV files with known properties for testing against R.

Test cases:
1. basic_100x3      - Normal, well-conditioned, n=100, p=3
2. tall_skinny      - n >> p (1000 x 5)
3. near_square      - n ≈ p (50 x 45)
4. ill_conditioned  - High condition number
5. collinear_almost - Near-singular (high correlation between predictors)
6. different_scales - Predictors on very different scales
7. no_intercept     - Data centered, no intercept needed
8. large_coeffs     - Large true coefficients (numerical stability)
9. small_noise      - Very small residual variance (R² ≈ 1)
10. high_noise      - High residual variance (R² ≈ 0.5)

Run from /mnt/projects/pystatistics:
    python tests/fixtures/generate_fixtures.py
"""

import numpy as np
import json
from pathlib import Path

# Reproducibility
RNG = np.random.default_rng(20250126)

FIXTURES_DIR = Path(__file__).parent


def save_fixture(name: str, X: np.ndarray, y: np.ndarray, beta_true: np.ndarray, metadata: dict):
    """Save fixture as CSV with metadata JSON."""
    # Combine X and y
    data = np.column_stack([X, y])
    
    # Column names
    p = X.shape[1]
    col_names = [f'x{i}' for i in range(p)] + ['y']
    
    # Save CSV with high precision
    csv_path = FIXTURES_DIR / f"{name}.csv"
    header = ','.join(col_names)
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.18e')
    
    # Save metadata
    meta_path = FIXTURES_DIR / f"{name}_meta.json"
    meta = {
        'name': name,
        'n': int(X.shape[0]),
        'p': int(X.shape[1]),
        'beta_true': beta_true.tolist(),
        'condition_number': float(np.linalg.cond(X)),
        **metadata
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  ✅ {name}: n={meta['n']}, p={meta['p']}, cond={meta['condition_number']:.2e}")


def generate_basic_100x3():
    """Normal, well-conditioned case."""
    n, p = 100, 3
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    beta_true = np.array([1.0, 2.0, -0.5])
    sigma = 0.5
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('basic_100x3', X, y, beta_true, {
        'sigma': sigma,
        'description': 'Basic well-conditioned regression with intercept'
    })


def generate_tall_skinny():
    """n >> p case."""
    n, p = 1000, 5
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal((n, p-1))
    ])
    beta_true = np.array([0.5, 1.0, -1.5, 2.0, -0.25])
    sigma = 1.0
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('tall_skinny', X, y, beta_true, {
        'sigma': sigma,
        'description': 'Tall skinny matrix n >> p'
    })


def generate_near_square():
    """n ≈ p (just barely overdetermined)."""
    n, p = 50, 45
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal((n, p-1))
    ])
    beta_true = RNG.standard_normal(p)
    sigma = 0.1
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('near_square', X, y, beta_true, {
        'sigma': sigma,
        'description': 'Nearly square matrix, low degrees of freedom'
    })


def generate_ill_conditioned():
    """High condition number but still full rank."""
    n, p = 100, 5
    
    # Create X with specified singular values (condition number ~ 1e6)
    U, _ = np.linalg.qr(RNG.standard_normal((n, p)))
    V, _ = np.linalg.qr(RNG.standard_normal((p, p)))
    singular_values = np.array([1e6, 1e4, 1e2, 1e1, 1.0])
    X = U @ np.diag(singular_values) @ V.T
    
    # Add intercept column
    X = np.column_stack([np.ones(n), X])
    
    beta_true = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.05])
    sigma = 0.1
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('ill_conditioned', X, y, beta_true, {
        'sigma': sigma,
        'target_condition': 1e6,
        'description': 'Ill-conditioned matrix with high condition number'
    })


def generate_collinear_almost():
    """Near-singular: two predictors highly correlated."""
    n, p = 100, 4
    
    x1 = RNG.standard_normal(n)
    x2 = RNG.standard_normal(n)
    # x3 is almost x1 + x2 (correlation > 0.999)
    x3 = x1 + x2 + RNG.standard_normal(n) * 0.01
    
    X = np.column_stack([np.ones(n), x1, x2, x3])
    beta_true = np.array([1.0, 2.0, -1.0, 0.5])
    sigma = 0.5
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('collinear_almost', X, y, beta_true, {
        'sigma': sigma,
        'description': 'Near-collinear predictors (x3 ≈ x1 + x2)'
    })


def generate_different_scales():
    """Predictors on very different scales."""
    n, p = 100, 4
    
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n) * 1e-6,    # Very small scale
        RNG.standard_normal(n) * 1.0,      # Normal scale
        RNG.standard_normal(n) * 1e6,      # Very large scale
    ])
    
    beta_true = np.array([1.0, 1e6, 1.0, 1e-6])  # Compensating coefficients
    sigma = 0.5
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('different_scales', X, y, beta_true, {
        'sigma': sigma,
        'description': 'Predictors on vastly different scales'
    })


def generate_no_intercept():
    """Centered data, no intercept in true model."""
    n, p = 100, 3
    
    # Generate centered predictors
    X_raw = RNG.standard_normal((n, p))
    X = X_raw - X_raw.mean(axis=0)  # Center
    
    beta_true = np.array([2.0, -1.0, 0.5])
    sigma = 0.3
    y_raw = X @ beta_true + RNG.standard_normal(n) * sigma
    y = y_raw - y_raw.mean()  # Center response too
    
    save_fixture('no_intercept', X, y, beta_true, {
        'sigma': sigma,
        'centered': True,
        'description': 'Centered data, no intercept term'
    })


def generate_large_coeffs():
    """Large true coefficients."""
    n, p = 100, 3
    
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    
    beta_true = np.array([1e4, 5e3, -2e3])
    sigma = 100.0
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('large_coeffs', X, y, beta_true, {
        'sigma': sigma,
        'description': 'Large coefficient values'
    })


def generate_small_noise():
    """Very high R² (almost perfect fit)."""
    n, p = 100, 3
    
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    
    beta_true = np.array([1.0, 2.0, -0.5])
    sigma = 1e-6  # Tiny noise
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('small_noise', X, y, beta_true, {
        'sigma': sigma,
        'expected_r2': 'very close to 1.0',
        'description': 'Nearly perfect fit'
    })


def generate_high_noise():
    """Low R² (high residual variance)."""
    n, p = 100, 3
    
    X = np.column_stack([
        np.ones(n),
        RNG.standard_normal(n),
        RNG.standard_normal(n)
    ])
    
    beta_true = np.array([1.0, 2.0, -0.5])
    sigma = 5.0  # High noise relative to signal
    y = X @ beta_true + RNG.standard_normal(n) * sigma
    
    save_fixture('high_noise', X, y, beta_true, {
        'sigma': sigma,
        'expected_r2': 'around 0.5 or less',
        'description': 'High residual variance'
    })


def main():
    print("Generating regression test fixtures...\n")
    
    generate_basic_100x3()
    generate_tall_skinny()
    generate_near_square()
    generate_ill_conditioned()
    generate_collinear_almost()
    generate_different_scales()
    generate_no_intercept()
    generate_large_coeffs()
    generate_small_noise()
    generate_high_noise()
    
    print(f"\n✅ Generated 10 fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()