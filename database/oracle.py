import oracledb


class OracleDatabase:

    def __init__(self, user, password, dsn):
        self.connection = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn
        )

    def execute_safe(self, query):
        query = query.strip()

        # Remove SQL*Plus-style trailing semicolon
        if query.endswith(";"):
            query = query[:-1].rstrip()

        cursor = self.connection.cursor()

        try:
            cursor.execute(query)

            # SELECT returns rows; INSERT/UPDATE/DELETE/DDL don't
            if cursor.description is not None:
                rows = cursor.fetchall()
            else:
                rows = []

            self.connection.commit()

            return rows

        finally:
            cursor.close()

    def close(self):
        self.connection.close()