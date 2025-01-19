import sqlite3
import time
from copy import deepcopy

import cdd
import numpy as np

from sqlite_utils import SQLiteReader


def generate_constraints(n, var_min, var_max):
    """
    Generate constraints for an n-dimensional space with var_min and var_max.
    Returns a list of tuples in the format (coe1, coe2, ..., coen, constant).
    # Define inequalities in the form A x + b > 0
    """
    constraints = []

    for i in range(n):
        # Lower bound: x_i >= var_min -> x_i - var_min >= 0
        lower_bound = [0] * n
        lower_bound[i] = 1
        constraints.append((*lower_bound, -var_min))

        # Upper bound: x_i <= var_max -> -x_i + var_max >= 0
        upper_bound = [0] * n
        upper_bound[i] = -1
        constraints.append((*upper_bound, var_max))

    return constraints


def merge_constraints(node_constraints, init_constraints):
    """
    Merge node.constraints with init_constraints by fetching records from the database.
    Parameters:
        node_constraints (list): Constraints for the current node.
        init_constraints (list): Global initial constraints.
        m (int): Number of functions.
        n (int): Dimension of functions.
        db_name (str): Database file name.
        conn: SQLite database connection.
    Returns:
        list of tuples: Merged constraints.
    """
    # Deep-copy init_constraints to avoid modifying the original
    merged_constraints = deepcopy(init_constraints)


    for record_id in node_constraints:
        # Fetch record from the database
        if record_id < 0:
            # record = FunctionProfiler.read_from_sqlite(m=m, n=n, db_name=db_name, record_id=-record_id, conn=conn)
            record = SQLiteReader.get_record_by_id(-record_id)
            # Negate coefficients, keep constant unchanged
            record = tuple(-coeff for coeff in record[:-1]) + (-record[-1],)  # Convert to a tuple
        else:
            # record = FunctionProfiler.read_from_sqlite(m=m, n=n, db_name=db_name, record_id=record_id, conn=conn)
            record = SQLiteReader.get_record_by_id(record_id)
            # Keep coefficients, negate constant
            record = tuple(record[:-1]) + (record[-1],)  # Convert to a tuple

        # Append the modified record as a tuple to merged_constraints
        merged_constraints.append(record)

    # print("merged_constraints: ", merged_constraints)
    return merged_constraints


class FunctionProfiler:
    total_time_compute_vertices = 0.0
    total_time_check_function = 0.0
    total_time_read_from_sqlite = 0.0
    total_time_satisfies_all_constraints = 0.0  # New accumulator for satisfies_all_constraints
    total_time_get_right_vertices = 0.0

    total_feasibility_check_calls = 0
    total_compute_vertices_calls = 0
    total_get_right_vertices_calls = 0

    compute_vertex_set = []

    @classmethod
    def compute_vertices(cls, constraints):
        # print("constraints: ", constraints)
        start_time = time.time()
        try:
            rows = []
            for constraint in constraints:
                # Split the tuple into coefficients and the constant
                *coefficients, constant = constraint
                row = [constant] + coefficients  # Include constant as the first element
                rows.append(row)

            # Convert to cdd matrix
            mat = cdd.matrix_from_array(rows, rep_type=cdd.RepType.INEQUALITY)

            # cdd.matrix_redundancy_remove(mat)

            # Create polyhedron from the matrix
            poly = cdd.polyhedron_from_matrix(mat)
            ext = cdd.copy_generators(poly)

            vertices = []
            for row in ext.array:
                if row[0] == 1.0:  # This indicates a vertex
                    vertex = [round(coord) for coord in row[1:]]
                    # vertex = [coord for coord in row[1:]]
                    vertices.append(vertex)


        except Exception as e:
            print(f"Error in compute_vertices: {e}")
            vertices = []

        elapsed_time = time.time() - start_time
        cls.total_time_compute_vertices += elapsed_time
        cls.total_compute_vertices_calls += 1
        return vertices

    @classmethod
    def check_sign_change(cls, func, vertices, cache=None) -> bool:
        """
        Checks if the function evaluates to both >0 and <0 across the vertex set,
        with timer and cache logic included.

        Parameters:
            func (list or tuple): A linear function represented as coefficients followed by a constant term.
            vertices (list or numpy.ndarray): A list or array of vertices, where each vertex is an n-dimensional point.
            cache (dict, optional): A dictionary for caching results.

        Returns:
            bool: True if the function evaluates to both positive and negative values, False otherwise.
        """

        start_time = time.time()

        # Separate coefficients and constant
        coefficients = np.array(func[:-1], dtype=float)
        constant = float(func[-1])

        # print("coefficients: ", coefficients, "constant: ", constant)
        # print("vertices: ", vertices)
        # Convert vertices to a numpy array if it's a list
        vertices = np.asarray(vertices, dtype=float)

        # Vectorized evaluation: values = (vertices @ coefficients) + constant
        values = vertices.dot(coefficients) + constant

        # Check overall min and max
        val_min = values.min()
        val_max = values.max()

        # We have a sign change if val_min < 0 and val_max > 0
        result = (val_min < 0.0) and (val_max > 0.0)

        elapsed_time = time.time() - start_time
        cls.total_time_check_function += elapsed_time
        cls.total_feasibility_check_calls += 1


        return result

    @classmethod
    def read_from_sqlite(cls, m, n, db_name="test_intersections.db", record_id=None, conn=None):
        """
        Read records from a dynamically named SQLite table based on m and n.
        Optionally filter by ID. Use an existing connection if provided.
        Returns a single record (without the index) as a tuple or a list of tuples.
        """
        start_time = time.time()

        close_conn = False
        if conn is None:
            conn = sqlite3.connect(db_name)
            close_conn = True

        cursor = conn.cursor()
        table_name = f"intersections_m{m}_n{n}"

        # Query records
        if record_id is not None:
            cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (record_id,))
            result = cursor.fetchone()  # Fetch a single record
            result = tuple(result[1:]) if result else None  # Skip the index (position 0)
        else:
            cursor.execute(f"SELECT * FROM {table_name}")
            result = [tuple(row[1:]) for row in cursor.fetchall()]  # Skip the index for all rows

        # Close the connection if it was created in this function
        if close_conn:
            conn.close()

        elapsed_time = time.time() - start_time
        cls.total_time_read_from_sqlite += elapsed_time
        return result


    @classmethod
    def satisfies_all_constraints(cls, vertex, inequalities):
        """
        Check if the given vertex satisfies all linear inequalities of the form:
        a1*x1 + a2*x2 + ... + c > 0
        """
        start_time = time.time()

        # Convert vertex to a NumPy array for efficient computation
        vertex = np.array(vertex)

        for constraint in inequalities:
            # Separate coefficients and constant
            coefficients = np.array(constraint[:-1])
            c = constraint[-1]

            # Compute lhs using dot product for efficiency
            lhs = np.dot(coefficients, vertex) + c

            # Check if lhs <= 0
            if lhs <= 0:
                elapsed_time = time.time() - start_time
                cls.total_time_satisfies_all_constraints += elapsed_time
                return False

        elapsed_time = time.time() - start_time
        cls.total_time_satisfies_all_constraints += elapsed_time
        return True

    @classmethod
    def get_right_vertices(cls, left_vertices, current_vertices, new_vertices, func, cache=None):
        """
        Determines the right vertices from the combined set of left + current vertices
        based on the given function and record_id, eliminating duplicates.
        Includes timing and call count.

        Parameters:
            left_vertices (list): Vertices from the left child.
            current_vertices (list): Current vertices.
            func (list or tuple): Linear function represented as coefficients followed by a constant term.
            cache (dict, optional): Cache to store and retrieve precomputed results.

        Returns:
            list: Right vertices.
        """
        start_time = time.time()

        # Determine right vertices
        right_vertices = []

        # Combine left and current vertices and remove duplicates while preserving order
        combined_vertices = []
        seen = set()
        for vertex in new_vertices:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in seen:
                seen.add(vertex_tuple)
                right_vertices.append(vertex)

        for vertex in left_vertices + current_vertices:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in seen:
                seen.add(vertex_tuple)
                combined_vertices.append(vertex)


        # Separate coefficients and constant
        coefficients = np.array(func[:-1])
        constant = func[-1]


        for vertex in combined_vertices:
            value = np.dot(vertex, coefficients) - constant
            if value >= 0:
                right_vertices.append(vertex)


        # Update timing and call count
        elapsed_time = time.time() - start_time
        cls.total_time_get_right_vertices += elapsed_time
        cls.total_get_right_vertices_calls += 1

        return right_vertices
