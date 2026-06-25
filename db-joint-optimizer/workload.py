import psycopg2
import multiprocessing
import random
import string
import time

#  Heavy SELECT (CPU load)
HEAVY_SELECT = """
SELECT COUNT(*)
FROM pg_class c
JOIN pg_attribute a ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace;
"""

#  Write-heavy queries (TPS + WAL + IO)
INSERT_QUERY = """
INSERT INTO load_test (data)
VALUES (%s);
"""

UPDATE_QUERY = """
UPDATE load_test
SET data = %s
WHERE id = (
    SELECT id FROM load_test ORDER BY RANDOM() LIMIT 1
);
"""

#  Create table if not exists
INIT_QUERY = """
CREATE TABLE IF NOT EXISTS load_test (
    id SERIAL PRIMARY KEY,
    data TEXT
);
"""


def random_string(length=50):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def worker(worker_id):
    conn = psycopg2.connect(
        dbname="tpch",
        user="postgres",
        password="postgres",
        host="127.0.0.1"
    )
    conn.autocommit = True
    cur = conn.cursor()

    #  Ensure table exists
    cur.execute(INIT_QUERY)

    print(f" Worker {worker_id} started")

    while True:
        try:
            action = random.choice(["select", "insert", "update"])

            if action == "select":
                cur.execute(HEAVY_SELECT)

            elif action == "insert":
                # batch inserts → high TPS
                for _ in range(5):
                    cur.execute(INSERT_QUERY, (random_string(),))

            elif action == "update":
                cur.execute(UPDATE_QUERY, (random_string(),))

        except Exception as e:
            print(f"Worker {worker_id} error:", e)


if __name__ == "__main__":
    processes = 80  #  increase concurrency

    print(f" Starting load generator with {processes} workers...")

    pool = multiprocessing.Pool(processes)
    pool.map(worker, range(processes))