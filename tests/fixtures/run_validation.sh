#!/bin/bash
#
# Full validation pipeline for PyStatistics regression
#
# Run from /mnt/projects/pystatistics:
#   bash tests/fixtures/run_validation.sh
#

set -e

cd /mnt/projects/pystatistics

echo "=============================================="
echo "PyStatistics Regression Validation Pipeline"
echo "=============================================="
echo ""

# Step 1: Generate fixtures
echo "Step 1: Generating test fixtures..."
python tests/fixtures/generate_fixtures.py
echo ""

# Step 2: Run R validation
echo "Step 2: Running R reference analysis..."
Rscript tests/fixtures/run_r_validation.R
echo ""

# Step 3: Compare Python vs R
echo "Step 3: Validating PyStatistics against R..."
python tests/fixtures/validate_against_r.py