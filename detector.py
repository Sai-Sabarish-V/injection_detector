import re
import joblib
import numpy as np

from scipy.sparse import hstack, csr_matrix

from database.oracle import OracleDatabase


log_reg = joblib.load(r"models pkl files/final_model_fe.pkl")
log_reg_char_vectorizer = joblib.load(r"models pkl files/char_vectorizer_fe.pkl")
log_reg_word_vectorizer = joblib.load(r"models pkl files/word_vectorizer_fe.pkl")
scaler = joblib.load(r"models pkl files/sql_feature_scaler.pkl")

xgb_model = joblib.load(r"models pkl files/xgb_model.pkl")
xgb_char_vectorizer = joblib.load(r"models pkl files/char_vectorizer_fe.pkl")
xgb_word_vectorizer = joblib.load(r"models pkl files/word_vectorizer_fe.pkl")


def extract_sql_features(query):
    q = query.lower()

    features = {
        "length": len(query),
        "word_count": len(query.split()),
        "single_quotes": query.count("'"),
        "double_quotes": query.count('"'),
        "backticks": query.count("`"),
        "semicolons": query.count(";"),
        "equals": query.count("="),
        "parentheses": query.count("(") + query.count(")"),
        "commas": query.count(","),
        "slashes": query.count("/"),
        "or_count": len(re.findall(r"\bor\b", q)),
        "and_count": len(re.findall(r"\band\b", q)),
        "between_count": len(re.findall(r"\bbetween\b", q)),
        "select_count": len(re.findall(r"\bselect\b", q)),
        "from_count": len(re.findall(r"\bfrom\b", q)),
        "where_count": len(re.findall(r"\bwhere\b", q)),
        "union_count": len(re.findall(r"\bunion\b", q)),
        "insert_count": len(re.findall(r"\binsert\b", q)),
        "update_count": len(re.findall(r"\bupdate\b", q)),
        "delete_count": len(re.findall(r"\bdelete\b", q)),
        "drop_count": len(re.findall(r"\bdrop\b", q)),
        "truncate_count": len(re.findall(r"\btruncate\b", q)),
        "create_count": len(re.findall(r"\bcreate\b", q)),
        "alter_count": len(re.findall(r"\balter\b", q)),
        "comment_count": len(re.findall(r"--|/\*|\*/|#", query)),
        "hex_count": len(re.findall(r"0x[0-9a-f]+", q)),
        "sleep_count": len(re.findall(r"\bsleep\b", q)),
        "benchmark_count": len(re.findall(r"\bbenchmark\b", q)),
        "xp_cmdshell_count": len(re.findall(r"xp_cmdshell", q)),
        "statement_count": len([x for x in query.split(";") if x.strip()]),
        "semicolon_after_quote": int(bool(re.search(r"['\"].*;", query, re.DOTALL))),
        "comment_after_semicolon": int(bool(re.search(r";.*(--|#|/\*)", query, re.DOTALL))),
        "quote_before_comment": int(bool(re.search(r"['\"].*(--|#|/\*)", query, re.DOTALL))),
        "tautology_count": len(re.findall(r"\b\d+\s*=\s*\d+\b", q))
    }

    return list(features.values())


def predict_log_reg(query):
    char_features = log_reg_char_vectorizer.transform([query])
    word_features = log_reg_word_vectorizer.transform([query])

    tfidf_features = hstack([
        char_features,
        word_features
    ]).tocsr()

    sql_features = np.array([
        extract_sql_features(query)
    ])

    sql_features = scaler.transform(sql_features)

    final_features = hstack([
        tfidf_features,
        csr_matrix(sql_features)
    ]).tocsr()

    probability = log_reg.predict_proba(
        final_features
    )[0, 1]

    prediction = int(probability >= 0.5)

    return {
        "prediction": (
            "SQL Injection"
            if prediction == 1
            else "Safe"
        ),
        "probability": float(probability)
    }


def predict_xgb(query):
    char_features = xgb_char_vectorizer.transform([query])
    word_features = xgb_word_vectorizer.transform([query])

    tfidf_features = hstack([
        char_features,
        word_features
    ]).tocsr()

    sql_features = np.array([
        extract_sql_features(query)
    ])

    sql_features = scaler.transform(sql_features)

    final_features = hstack([
        tfidf_features,
        csr_matrix(sql_features)
    ]).tocsr()

    probability = xgb_model.predict_proba(
        final_features
    )[0, 1]

    prediction = int(probability >= 0.5)

    return {
        "prediction": (
            "SQL Injection"
            if prediction == 1
            else "Safe"
        ),
        "probability": float(probability)
    }


print("LogReg expects:", log_reg.n_features_in_)
print("XGBoost expects:", xgb_model.n_features_in_)



flag = input(
    "Use XGBoost model? (y/n): "
).strip().lower() == "y"


def read_query():
    print("SQL> ", end="")

    lines = []

    while True:
        line = input()
        lines.append(line)

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

    if flag:
        result = predict_xgb(query)
    else:
        result = predict_log_reg(query)

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