import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        database="baylogic",
        user="root",
        password="root"
    )

def create_table():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS parking_log (
            id                  INT AUTO_INCREMENT PRIMARY KEY,
            bay                 INT,
            direction           VARCHAR(20),
            entry_time          VARCHAR(20),
            exit_time           VARCHAR(20),
            duration_seconds    INT,
            time the vehicle was operated on      VARCHAR(30),
            technician_seconds  INT,
            technician_duration VARCHAR(30)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_record(record):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO parking_log (
            bay, direction, entry_time, exit_time,
            duration_seconds, duration_human,
            technician_seconds, technician_duration
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        record["bay"],
        record["direction"],
        record["entry_time"],
        record["exit_time"],
        record["duration_seconds"],
        record["duration_human"],
        record["technician_seconds"],
        record["technician_duration"]
    ))
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Inserted record for bay {record['bay']}")
