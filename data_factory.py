import numpy as np
import itertools
import random
import argparse
from sqlite_utils import save_to_sqlite, read_from_sqlite, create_extended_table


def generate_functions(m, n, low=0, high=1):
    """Generate m functions in n dimensions with random coefficients."""
    functions = np.random.randint(low, high + 1, size=(m, n)).tolist()
    return functions


def compute_differences_with_constants(functions, constant_low, constant_high):
    """Compute all unique differences between pairs of functions and match each with a random constant."""
    records = []
    record_id = 1  # Start IDs from 1
    for (f_i, f_j) in itertools.combinations(functions, 2):
        diff = [a - b for a, b in zip(f_i, f_j)]
        constant_a = random.randint(constant_low, constant_high)
        constant_b = random.randint(constant_low, constant_high)

        records.append((record_id, *diff, constant_a - constant_b))  # Include ID in the record
        record_id += 1
    return records

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate and process random functions.")
    parser.add_argument("m", type=int, help="Number of functions")
    parser.add_argument("n", type=int, help="Dimension of functions")
    parser.add_argument("--low", type=int, default = 0, help="Lower bound for coefficients (default: 0)")
    parser.add_argument("--high", type=int, default = 100, help="Upper bound for coefficients (default: 100)")
    parser.add_argument("--constant-low", type=int, default= 0, help="Lower bound for random constants (default: 0)")
    parser.add_argument("--constant-high", type=int, default= 0, help="Upper bound for random constants (default: 100)")
    args = parser.parse_args()

    m = args.m
    n = args.n
    low = args.low
    high = args.high
    constant_low = args.constant_low
    constant_high = args.constant_high

    # create_extended_table(m, n, "test_intersections.db", low, high)

    # Step 1: Generate m functions with n dimensions
    functions = generate_functions(m, n, low, high)

    # Step 2: Compute all unique differences between pairs of functions and match with constants
    records = compute_differences_with_constants(functions, constant_low, constant_high)

    # print(records)
    # Step 3: Save the records to an SQLite table dynamically based on m and n
    save_to_sqlite(records, m, n, "test_intersections.db", constant_low, constant_high)
