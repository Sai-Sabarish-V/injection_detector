from database.oracle import OracleDatabase
from model import predict_query


def read_query():
    print("SQL> ", end="")

    lines = []

    while True:
        line = input()

        lines.append(line)

        # Semicolon means the SQL statement is complete
        if line.rstrip().endswith(";"):
            break

        print("...> ", end="")

    return "\n".join(lines)

db = OracleDatabase(
    user="system",
    password="a",
    dsn="localhost:1521/XEPDB1"
)


while True:
    query = read_query()

    if query.strip().lower() in ("exit;", "exit"):
        break

    result = predict_query(query)

    print(
        f"Detector: {result['prediction']} "
        f"({result['probability']:.2%})"
    )

    if result["prediction"] == "SQL Injection":
        print("BLOCKED: Potential SQL injection detected.")
        continue

    try:
        rows = db.execute_safe(query)

        for row in rows:
            print(row)

    except Exception as e:
        print("Oracle error:", e)


db.close()