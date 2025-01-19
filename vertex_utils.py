from decimal import Decimal

import numpy as np

from function_utils import check_smallest_intervals


def create_lookup_table(variable_range_start, variable_range_end, interval, dimensions):
    # Generate discretized values
    discretized_values = np.round(np.arange(variable_range_start, variable_range_end + interval, interval) / interval) * interval
    # Create a lookup table for all dimensions
    lookup_table = {i: set(discretized_values) for i in range(dimensions)}
    return lookup_table


def round_vertex(vertex, interval):
    """
    Round each value in the vertex to the nearest multiple of the interval.
    """
    return [round(value / interval) * interval for value in vertex]


def check_new_vertices(lookup_table, new_vertices, interval):
    """
    Check if any value in the new vertices is unvisited.

    Parameters:
    lookup_table (dict): Current lookup table of unvisited values.
    new_vertices (list of lists): New set of vertices.
    interval (float): Discretization interval.

    Returns:
    bool: True if new values are found, False otherwise.
    """
    for vertex in new_vertices:
        for i, value in enumerate(vertex):
            rounded_value = round(value / interval) * interval
            if rounded_value in lookup_table[i]:
                return True  # New value found
    return False


def update_visited(lookup_table, vertices, interval):
    """
    Update the lookup table by marking rounded values from the vertex set as visited.
    """
    for vertex in vertices:
        for i, value in enumerate(vertex):
            rounded_value = round(value / interval) * interval
            if rounded_value in lookup_table[i]:  # Mark as visited by removing from the table
                lookup_table[i].remove(rounded_value)


def process_new_vertices(lookup_table, new_vertices, interval):
    """
    Check if new vertices introduce any unvisited values, and update the lookup table if so.

    Parameters:
    lookup_table (dict): Current lookup table of unvisited values.
    new_vertices (list of lists): New set of vertices to check and potentially update.
    interval (float): Discretization interval.

    Returns:
    bool: True if new values were introduced (and the table updated), False otherwise.
    """
    if check_new_vertices(lookup_table, new_vertices, interval):
        update_visited(lookup_table, new_vertices, interval)
        return True
    return False


class VertexManager:
    def __init__(self, precision, max_visits=1):
        """
        Initialize the VertexManager with a given precision and max visits threshold.

        Parameters:
        precision (float): The precision to use for rounding and interval checking.
        max_visits (int): The number of times a rounded set can be visited before considering it a violation.
        """
        self.precision = precision
        self.max_visits = max_visits
        self.rounded_sets_list = []  # List to store each encountered rounded_set

    def _round_vertex_set(self, vertices):
        """
        Round all values in the set of vertices down to the nearest multiple of the precision.

        Parameters:
        vertices (list of list): A list of vertices, each vertex being a list of coordinates.

        Returns:
        frozenset: A set of rounded vertices as tuples.
        """
        precision_decimal = Decimal(str(self.precision))

        rounded_vertices = [
            tuple(
                float(Decimal(str(value)) // precision_decimal * precision_decimal)
                for value in vertex
            )
            for vertex in vertices
        ]

        return frozenset(rounded_vertices)

    def process_vertex_set(self, vertices):
        """
        Process a set of vertices based on the given rules.

        Parameters:
        vertices (list of list): A set of vertices to be checked and processed.

        Returns:
        bool: True if the set of vertices violates the rules, False otherwise.
              Rules:
              - If a set violates the interval condition, return True.
              - If a set has already been visited 'max_visits' times, return True.
              - Otherwise, return False.
        """
        # Round the set of vertices
        # rounded_set = self._round_vertex_set(vertices)

        # Append the rounded_set to the list
        self.rounded_sets_list.append(vertices)
        # print(self.rounded_sets_list)

        # Count how many times this rounded_set has appeared
        current_count = self.rounded_sets_list.count(vertices)

        # If already visited max_visits times, return True (violation)
        if current_count > self.max_visits:
            # print(f"Visited {current_count} times: {vertices}")
            return True
        else:
            return False

        #
        # # Check if the set satisfies the smallest interval condition
        # if check_smallest_intervals(vertices, self.precision):
        #     return True  # Valid and recorded
        # else:
        #     # Violates the interval condition
        #     # print(f"Smallest interval violation: {vertices}")
        #     return False