#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#%%

df = pd.read_csv("C:\\Users\\sabbu\\PycharmProjects\\injection_detector\\dataset\\SQLiV3.csv")

df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

df = df[df["Label"].isin([0, 1])]

df = df[["Sentence", "Label"]]

#%%
X = df["Sentence"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#%%
import re
import numpy as np
import pandas as pd


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

        "statement_count": len([
            x for x in query.split(";")
            if x.strip()
        ]),

        "semicolon_after_quote": int(
            bool(re.search(r"['\"].*;", query, re.DOTALL))
        ),

        "comment_after_semicolon": int(
            bool(re.search(r";.*(--|#|/\*)", query, re.DOTALL))
        ),

        "quote_before_comment": int(
            bool(re.search(r"['\"].*(--|#|/\*)", query, re.DOTALL))
        ),

        # Boolean patterns
        "tautology_count": len(
            re.findall(
                r"\b\d+\s*=\s*\d+\b",
                q
            )
        ),
    }

    return list(features.values())


#%%
X_train_sql = np.array([
    extract_sql_features(q)
    for q in X_train
])

X_test_sql = np.array([
    extract_sql_features(q)
    for q in X_test
])

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_sql_scaled = scaler.fit_transform(X_train_sql)
X_test_sql_scaled = scaler.transform(X_test_sql)
#%%
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=100000
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

#%%
from scipy.sparse import hstack

X_train_combined = hstack([
    X_train_char,
    X_train_word
]).tocsr()

X_test_combined = hstack([
    X_test_char,
    X_test_word
]).tocsr()

#%%
from scipy.sparse import csr_matrix, hstack

X_train_final = hstack([
    X_train_combined,
    csr_matrix(X_train_sql_scaled)
]).tocsr()

X_test_final = hstack([
    X_test_combined,
    csr_matrix(X_test_sql_scaled)
]).tocsr()

#%%
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_final, y_train)

y_pred_xgb = xgb_model.predict(X_test_final)


#%%

def predict_query_xgb(query):
    char_features = char_vectorizer.transform([query])
    word_features = word_vectorizer.transform([query])

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
