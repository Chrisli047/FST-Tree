import argparse


from sqlite_utils import get_all_ids, SQLiteReader
from function_utils import generate_constraints, FunctionProfiler
from tqdm import tqdm  # Import the progress bar library
import time  # Import time for measuring execution
import sqlite3


from vi_tree import VITree
from visualization_utils import plot_linear_equations


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process VI tree with given m and n.")
    parser.add_argument("m", type=int, help="Number of functions (m)")
    parser.add_argument("n", type=int, help="Dimension of functions (n)")
    parser.add_argument("--db", type=str, default="test_intersections.db", help="Database file (default: intersections.db)")
    parser.add_argument("--var_min", type=float, default=0, help="Minimum value for variables (default: 0)")
    parser.add_argument("--var_max", type=float, default=1000, help="Maximum value for variables (default: 10)")
    args = parser.parse_args()

    m = args.m
    n = args.n
    db_name = args.db
    var_min = args.var_min
    var_max = args.var_max

    # Dynamically construct the table name
    table_name = f"intersections_m{m}_n{n}"

    # Open a single connection to the database
    conn = sqlite3.connect(db_name)

    # Get all IDs from the table
    ids = get_all_ids(m, n, db_name=db_name)
    print(f"Found {len(ids)} IDs in table {table_name}.")

    # Generate constraints using n (as dimensionality), var_min, and var_max
    constraints = generate_constraints(n, var_min, var_max)
    print("Generated constraints as tuples:")
    for constraint in constraints:
        print(constraint)

    # Compute vertices for the initial domain
    vertices = FunctionProfiler.compute_vertices(constraints)
    print(f"Computed vertices of the initial domain: {vertices}")

    SQLiteReader.read_all_from_sqlite(m, n)

    # Create a VI Tree
    vi_tree = VITree()

    # Fetch and process records by ID
    print("Processing records:")

    # Set a counter for the number of intersection partitions the domain
    counter = 0


    # Start the timer
    start_time = time.time()

    records_to_draw = []

    # Insert records into the VI Tree with progress tracking
    for record_id in tqdm(ids, desc="Processing Records", unit="sampled_records"):

        record = SQLiteReader.get_record_by_id(record_id)
        # records_to_draw.append(record)

        if FunctionProfiler.check_sign_change(record, vertices):
            counter += 1

            # records_to_draw.append(record)
            # print(f"Record with ID {record_id} satisfies the condition: {record}")
            # Insert the record into the VI Tree
            vi_tree.insert(record_id, constraints, vertices, n)
        # if counter > 1000:
        #     break

    # Stop the timer
    end_time = time.time()

    # Print the number of intersection partitions
    print(f"Number of intersection partitions: {counter}")

    # Print the time taken to insert all records
    print(f"Time taken to insert all records into the VI Tree: {end_time - start_time:.2f} seconds")

    print("\nVI Tree Structure (Layer by Layer with Records):")
    vi_tree.print_tree_by_layer(m, n, db_name, conn)

    # Print the height of the tree
    print(f"Height of the VI Tree: {vi_tree.get_height()}")

    # Print the number of leaf nodes
    print(f"Number of leaf nodes in the VI Tree: {vi_tree.get_leaf_count()}")
    print(f"Number of nodes in the VI Tree: {vi_tree.get_total_node_count()}")

    print("Total time in compute_vertices:", FunctionProfiler.total_time_compute_vertices)
    print("Total time in check_function:", FunctionProfiler.total_time_check_function)
    print("Total time in read_from_sqlite:", FunctionProfiler.total_time_read_from_sqlite)
    print("Total time of compute right vertices:", FunctionProfiler.total_time_get_right_vertices)

    print("Total feasibility checking calls:", FunctionProfiler.total_feasibility_check_calls)
    print("Total compute vertices calls:", FunctionProfiler.total_compute_vertices_calls)
    print("Total compute right vertices calls:", FunctionProfiler.total_get_right_vertices_calls)

    # Close the database connection
    conn.close()
    print("Database connection closed.")
    #
    # plot_linear_equations(records_to_draw)
