#!/bin/bash

set -e

function print_header() {
    echo
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_header "Running Black"
black --check .

# if conflicts, run `black .` to fix them

print_header "Running Flake8"
flake8 .

print_header "Running Bandit"
bandit -r src --exclude src/human_mesh/TokenHMR

echo
echo "All checks completed successfully!"