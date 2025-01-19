# VI-tree Project

## Prerequisites

- **Python Version**: Python 3.12 is required.
- **Dependencies**: Install all required packages using `requirements.txt`.

## Installation

1. Clone the repository or download the project files.
2. Ensure you have Python 3.12 installed.
3. Install the dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Data Using `data_factory.py`

The `data_factory.py` script generates random intersection data and stores it in an SQLite database.

**Command**:
```bash
python data_factory.py <m> <n> [--low <low>] [--high <high>] [--constant-low <constant_low>] [--constant-high <constant_high>]
```

**Parameters**:
- `<m>`: Number of functions (required).
- `<n>`: Dimension of functions (required).
- `--low`: Lower bound for function coefficients (default: 0).
- `--high`: Upper bound for function coefficients (default: 100).
- `--constant-low`: Lower bound for random constants (default: 0).
- `--constant-high`: Upper bound for random constants (default: 100).

**Example**:
Generate data with 5 functions in 3 dimensions, coefficients between 10 and 50, and constants between 0 and 20:
```bash
python data_factory.py 5 3 --low 10 --high 50 --constant-low 0 --constant-high 20
```

### 2. Build the VI Tree Using `vi_tree_main.py` 

The `vi_tree_main.py` script builds the VI Tree from the generated SQLite data.

**Command**:
```bash
python vi_tree_main.py <m> <n> [--db <db_name>] [--var_min <var_min>] [--var_max <var_max>]
```

**Parameters**:
- `<m>`: Number of functions (required).
- `<n>`: Dimension of functions (required).
- `--db`: SQLite database file name (default: `test_intersections.db`).
- `--var_min`: Minimum value for variables (default: 0).
- `--var_max`: Maximum value for variables (default: 10).

**Example**:
Build a VI Tree from data with 5 functions and 3 dimensions stored in `test_intersections.db`, using the variable range [0, 20]:
```bash
python vi_tree_main.py 5 3 --db test_intersections.db --var_min 0 --var_max 20
```

**Outputs**:
- The VI Tree structure is printed layer by layer, showing:
  - Node ID
  - Vertices
  - Associated record from the database
- Additional tree statistics:
  - Tree height
  - Number of leaf nodes
  - Processing time with a progress bar

### 3. Build the I Tree Using `i_tree_main.py` 

The `i_tree_main.py` script builds the I Tree from the generated SQLite data.

**Command**:
```bash
python i_tree_main.py <m> <n> [--db <db_name>] [--var_min <var_min>] [--var_max <var_max>]
```

**Parameters**:
- `<m>`: Number of functions (required).
- `<n>`: Dimension of functions (required).
- `--db`: SQLite database file name (default: `test_intersections.db`).
- `--var_min`: Minimum value for variables (default: 0).
- `--var_max`: Maximum value for variables (default: 10).

**Example**:
Build a I Tree from data with 5 functions and 3 dimensions stored in `test_intersections.db`, using the variable range [0, 20]:
```bash
python i_tree_main.py 5 3 --db test_intersections.db --var_min 0 --var_max 20
```

## Project Files

- **`data_factory.py`**: Generates random data and stores it in SQLite. Creates tables dynamically based on the number of functions and dimensions.
- **`vi_tree_main.py`**: Builds the VI Tree from data. Computes constraints and vertices for the initial domain and inserts records into the tree based on validation.
- **`requirements.txt`**: Contains all the necessary dependencies for the project.

## Example Workflow

1. Generate data with 100 functions in 2 dimensions:
   ```bash
   python data_factory.py 100 2
   ```

2. Build the VI Tree using the generated data:
   ```bash
   python vi_tree_main.py 100 2
   ```

