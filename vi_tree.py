from function_utils import (check_function, FunctionProfiler, merge_constraints, get_tight_constraints,
                            check_smallest_intervals)
from sqlite_utils import read_from_sqlite, SQLiteReader
from vertex_utils import create_lookup_table, process_new_vertices

init_constraints = []  # Global variable to store initial constraints

def round_vertices(data):
    return [[round(value, 3) for value in sublist] for sublist in data]

class TreeNode:
    def __init__(self, intersection_id, constraints=None, vertices=None):
        self.intersection_id = intersection_id  # ID of the intersection (record_id)
        self.constraints = constraints if constraints is not None else []  # Constraints for this node, defaults to []
        self.vertices = vertices if vertices is not None else []  # Associated vertices, defaults to []
        self.rounded_vertices = []  # Vertices rounded to the nearest interval
        self.left_children = None  # Left child
        self.right_children = None  # Right child
        self.skip_flag = False  # Flag to indicate if this node should be skipped
        self.not_enough_vertices = False


class VITree:
    def __init__(self):
        self.root = None  # Initialize the tree with no root

    def insert(self, record_id, constraints, vertices=None, n=None):
        """
        Insert a node into the VI tree using a non-recursive method.
        Parameters:
            record_id (int): Intersection ID for the node.
            constraints (list): Constraints for the node.
            vertices (list): Vertices for the node, defaults to an empty list if not provided.
            m (int): Number of functions.
            n (int): Dimension of functions.
            db_name (str): Database file name.
            conn: SQLite database connection.
        """
        global init_constraints

        new_node = TreeNode(record_id, None, vertices)

        if self.root is None:
            # Set the root if the tree is empty
            self.root = new_node

            # Update the global variable and node properties
            init_constraints = constraints
            print(f"Initial constraints for record {record_id}: {init_constraints}")

            # Explicitly set root node properties
            self.root.intersection_id = record_id
            self.root.vertices = vertices if vertices is not None else []

            # Initialize left and right children with constraints
            self.root.left_children = TreeNode(-record_id, [-record_id])
            self.root.right_children = TreeNode(record_id, [record_id])

            left_merged_constraints = merge_constraints(self.root.left_children.constraints, init_constraints)
            # print(f"Left merged constraints: {left_merged_constraints}")
            right_merged_constraints = merge_constraints(self.root.right_children.constraints, init_constraints)
            # print(f"Right merged constraints: {right_merged_constraints}")

            self.root.left_children.vertices = FunctionProfiler.compute_vertices(left_merged_constraints)
            # self.root.left_children.rounded_vertices = round_vertices(self.root.left_children.vertices)
            # print(f"Left children vertices: {self.root.left_children.vertices}")
            self.root.right_children.vertices = FunctionProfiler.compute_vertices(right_merged_constraints)
            # self.root.right_children.rounded_vertices = round_vertices(self.root.right_children.vertices)
            # print(f"Right children vertices: {self.root.right_children.vertices}")

            return

        # Use a stack to manage nodes for non-recursive traversal
        stack = [self.root.left_children, self.root.right_children]
        # print("length of stack", len(stack))
        # stack = [self.root]

        # Set to store previously computed vertices
        cache = {}

        while stack:
            # print(len(cache))
            current = stack.pop()
            # Get the record from the database

            insert_record = SQLiteReader.get_record_by_id(record_id)
            # print(f"Processing record {record_id}: {insert_record}")

            # if not FunctionProfiler.check_sign_change(insert_record, current.rounded_vertices, cache=cache):
            #     continue  # Skip to the next iteration if not satisfied

            if not FunctionProfiler.check_sign_change(insert_record, current.vertices, cache=cache):
                continue  # Skip to the next iteration if not satisfied


            if current.left_children is None and current.right_children is None:
                left_merged_constraints = merge_constraints(current.constraints + [-record_id], init_constraints)
                right_merged_constraints = merge_constraints(current.constraints + [record_id], init_constraints)

                # print(f"Left merged constraints: {left_merged_constraints}")
                # Compute Left children vertices once
                left_children_vertices = FunctionProfiler.compute_vertices(left_merged_constraints)
                # print("current vertices", current.vertices)
                # print(f"Left children vertices: {left_children_vertices}")
                # left_children_vertices_rounded = round_vertices(left_children_vertices)
                # print(f"Left children vertices rounded: {left_children_vertices_rounded}")

                # Compute the new vertex: Left children vertices - current vertices
                # new_vertices = [v for v in left_children_vertices if v not in current.vertices]
                # # new_vertices = [v for v in left_children_vertices_rounded if v not in current.rounded_vertices]
                #
                # if not new_vertices:  # If no new vertex is found, continue
                #     # print("No new vertices found")
                #     continue
                #
                # # Check if the new vertices are valid
                # right_children_vertices = FunctionProfiler.get_right_vertices(left_children_vertices, current.vertices, new_vertices, insert_record, cache=cache)

                right_children_vertices = FunctionProfiler.compute_vertices(right_merged_constraints)
                # print(f"Right children vertices: {right_children_vertices}")
                # right_children_vertices_rounded = round_vertices(right_children_vertices)

                # Use a dictionary to remove duplicates and maintain order
                left_children_vertices = [list(item) for item in {tuple(sublist): None for sublist in left_children_vertices}]
                right_children_vertices = [list(item) for item in {tuple(sublist): None for sublist in right_children_vertices}]


                if len(left_children_vertices) <= n or len(right_children_vertices) <= n:
                    # print("Not enough vertices")
                    continue

                # if [current.vertices].count(left_children_vertices) > 0 or [current.vertices].count(
                #         right_children_vertices) > 0:
                #     continue

                #
                # if len(left_children_vertices_rounded) <= n or len(right_children_vertices_rounded) <= n:
                #     continue
                #
                # if [current.rounded_vertices].count(left_children_vertices_rounded) > 0 or [current.rounded_vertices].count(
                #         right_children_vertices_rounded) > 0:
                #     continue

                if FunctionProfiler.compute_vertex_set.count(left_children_vertices) > 0 or FunctionProfiler.compute_vertex_set.count(right_children_vertices) > 0:
                    continue
                else:
                    FunctionProfiler.compute_vertex_set.append(left_children_vertices)
                    FunctionProfiler.compute_vertex_set.append(right_children_vertices)

                # print("lenght of left children vertices", len(left_children_vertices))
                # print("lenght of right children vertices", len(right_children_vertices))

                current.left_children = TreeNode(
                    -record_id,
                    constraints=[-record_id] + current.constraints
                )

                current.right_children = TreeNode(
                    record_id,
                    constraints=[record_id] + current.constraints
                )

                current.left_children.vertices = left_children_vertices
                current.right_children.vertices = right_children_vertices

                # current.left_children.rounded_vertices = left_children_vertices_rounded
                # current.right_children.rounded_vertices = right_children_vertices_rounded

                continue

            stack.append(current.left_children)
            stack.append(current.right_children)


    def print_tree_by_layer(self, m, n, db_name, conn):
        """
        Print the VI Tree layer by layer, showing each node's ID, vertices, and database record.
        Handles negative IDs by converting them to positive when fetching records.
        Parameters:
            m (int): Number of functions.
            n (int): Dimension of functions.
            db_name (str): Database file name.
            conn: SQLite database connection.
        """
        if self.root is None:
            print("The tree is empty.")
            return

        # Use a queue to implement level-order traversal, along with layer tracking
        queue = [(self.root, 0)]  # Each element is a tuple (node, layer)
        current_layer = 0
        layer_output = []

        while queue:
            current, layer = queue.pop(0)  # Dequeue the front node

            # Check if we've moved to a new layer
            if layer > current_layer:
                # Print all nodes in the previous layer
                print(f"Layer {current_layer}:")
                for node_output in layer_output:
                    print(node_output)
                print()  # Blank line between layers
                layer_output = []  # Reset the layer output
                current_layer = layer

            # Fetch record from the database
            record_id = abs(current.intersection_id)  # Use positive ID for fetching
            record = read_from_sqlite(m, n, db_name=db_name, record_id=record_id, conn=conn)

            # Add the current node's details to the layer output
            layer_output.append(
                f"Node ID: {current.intersection_id}, Vertices: {current.vertices}, Record: {record}"
            )

            # Enqueue the left and right children if they exist, with incremented layer
            if current.left_children:
                queue.append((current.left_children, layer + 1))
            if current.right_children:
                queue.append((current.right_children, layer + 1))

        # Print the last layer
        print(f"Layer {current_layer}:")
        for node_output in layer_output:
            print(node_output)

    def get_height(self):
        if self.root is None:
            return 0
        # Iterative approach using a queue (BFS)
        from collections import deque
        queue = deque([(self.root, 1)])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            if node is not None:
                max_depth = max(max_depth, depth)
                queue.append((node.left_children, depth + 1))
                queue.append((node.right_children, depth + 1))
        return max_depth

    def get_leaf_count(self):
        if self.root is None:
            return 0
        # Iterative approach using a stack (DFS)
        stack = [self.root]
        leaf_count = 0
        while stack:
            node = stack.pop()
            if node is not None:
                # Check if it is a leaf and not flagged as not_enough_vertices
                if node.left_children is None and node.right_children is None and not node.not_enough_vertices:
                    leaf_count += 1
                else:
                    stack.append(node.left_children)
                    stack.append(node.right_children)
        return leaf_count

    def get_total_node_count(self):
        if self.root is None:
            return 0
        # Iterative approach using a stack (DFS)
        stack = [self.root]
        total_count = 0
        while stack:
            node = stack.pop()
            if node is not None:
                # Increment the total node count for every node visited
                total_count += 1
                stack.append(node.left_children)
                stack.append(node.right_children)
        return total_count