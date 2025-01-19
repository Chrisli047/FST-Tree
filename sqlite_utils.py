import sqlite3
import time


def create_extended_table(m, n, db_name="test_intersections.db", new_column_low=0, new_column_high=100):
    """
    Extend the dimension of an existing table by appending one column and creating a new table.
    """
    import sqlite3
    import random

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Define table names
    old_table_name = f"intersections_m{m}_n{n-1}"
    new_table_name = f"intersections_m{m}_n{n}"

    # Check if the old table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (old_table_name,))
    if not cursor.fetchone():
        print(f"Table {old_table_name} does not exist. Cannot create {new_table_name}.")
        conn.close()
        return

    # Fetch records from the old table
    cursor.execute(f"SELECT * FROM {old_table_name}")
    records = cursor.fetchall()
    num_columns = len(records[0])  # Including ID and constant

    # Append a new column to each record
    extended_records = [
        (*record[:-1], random.randint(new_column_low, new_column_high) - random.randint(new_column_low, new_column_high), record[-1])
        for record in records
    ]

    # Define new table schema
    new_columns = ", ".join([f"c{i+1} INTEGER" for i in range(num_columns - 2 + 1)])  # Adding one more column
    cursor.execute(f"""
        CREATE TABLE {new_table_name} (
            id INTEGER PRIMARY KEY,
            {new_columns},
            constant INTEGER
        )
    """)

    # Insert extended records into the new table
    cursor.executemany(
        f"INSERT INTO {new_table_name} VALUES ({', '.join(['?' for _ in range(num_columns + 1)])})",
        extended_records
    )

    # Create index on ID for faster queries
    cursor.execute(f"CREATE INDEX idx_{new_table_name}_id ON {new_table_name}(id)")

    conn.commit()
    conn.close()
    print(f"New table {new_table_name} created successfully.")



def save_to_sqlite(records, m, n, db_name="test_intersections.db", constant_low=-150, constant_high=-50):
    """
    Save records to an SQLite database in a table named dynamically based on m and n.
    If the table exists, update the constants.
    """
    import sqlite3
    import random

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Define the dynamic table name
    table_name = f"intersections_m{m}_n{n}"

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone()

    if table_exists:
        print(f"Table {table_name} exists. Updating constants.")
        # Update constants for existing records
        cursor.execute(f"SELECT id FROM {table_name}")
        record_ids = [row[0] for row in cursor.fetchall()]

        for record_id in record_ids:
            new_constant = random.randint(constant_low, constant_high) - random.randint(constant_low, constant_high)
            cursor.execute(f"UPDATE {table_name} SET constant = ? WHERE id = ?", (new_constant, record_id))

    else:
        print(f"Table {table_name} does not exist. Creating a new table.")
        # Create table
        num_coefficients = len(records[0]) - 2  # Number of coefficients in each record
        columns = ", ".join([f"c{i+1} INTEGER" for i in range(num_coefficients)])
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                {columns},
                constant float
            )
        """)

        # Insert records
        cursor.executemany(
            f"INSERT INTO {table_name} VALUES ({', '.join(['?' for _ in range(num_coefficients + 2)])})",
            records
        )

        # Create index on ID for faster queries
        cursor.execute(f"CREATE INDEX idx_{table_name}_id ON {table_name}(id)")

    conn.commit()
    conn.close()



def read_from_sqlite(m, n, db_name="test_intersections.db", record_id=None, conn=None):
    """
    Read records from a dynamically named SQLite table based on m and n.
    Optionally filter by ID. Use an existing connection if provided.
    Returns a single record (without the index) as a tuple or a list of tuples.
    """
    # Use the provided connection, or create a new one
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

    return result



def get_all_ids(m, n, db_name="test_intersections.db"):
    """
    Fetch all IDs from the specified table.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    table_name = f"intersections_m{m}_n{n}"
    try:
        cursor.execute(f"SELECT id FROM {table_name}")
        ids = [row[0] for row in cursor.fetchall()]
        return ids
    except sqlite3.OperationalError as e:
        print(f"Error fetching IDs from table {table_name}: {e}")
        return []
    finally:
        conn.close()


class SQLiteReader:
    """
    Class to read records from SQLite and store them in a class variable.
    """
    records = []  # Class variable to store all fetched records
    time_get_record_by_id = 0.0

    @classmethod
    def read_all_from_sqlite(cls, m, n, db_name="test_intersections.db", conn=None):
        """
        Fetch all records from the table and save them to the class variable.
        Skips the index column.
        """
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(db_name)
            close_conn = True

        cursor = conn.cursor()
        table_name = f"intersections_m{m}_n{n}"

        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            cls.records = [tuple(row[1:]) for row in cursor.fetchall()]  # Skip index column
            print(f"Records loaded from table {table_name}.")
        except sqlite3.OperationalError as e:
            print(f"Error reading table {table_name}: {e}")
            cls.records = []  # Reset records if there's an error
        finally:
            if close_conn:
                conn.close()

    @classmethod
    def get_records(cls):
        """
        Return the records stored in the class variable.
        """
        return cls.records

    @classmethod
    def get_record_by_id(cls, record_id):
        """
        Retrieve a record by ID (1-based index).
        Returns None if the ID is out of range.
        """
        start_time = time.time()
        if not cls.records:
            print("No records loaded. Call read_all_from_sqlite first.")
            cls.time_get_record_by_id += time.time() - start_time
            return None

        try:
            # SQLite IDs usually start from 1, so subtract 1 for list indexing
            cls.time_get_record_by_id += time.time() - start_time
            return cls.records[record_id - 1]
        except IndexError:
            print(f"Record with ID {record_id} does not exist.")
            return None
