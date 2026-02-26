import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import os
import joblib
import openml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

MODEL_PATH = "app/model/model.joblib"

def load_data():
    # German Credit on OpenML: "credit-g"
    ds = openml.datasets.get_dataset("credit-g")
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    # y is categorical: 'good'/'bad'. Let's map bad=1 (default risk), good=0
    y = y.map({"good": 0, "bad": 1}).astype(int)
    return X, y

def build_pipeline(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return clf

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline(X_train)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC: {auc:.4f}")

    # Save ROC Curve
    import os
    os.makedirs("reports/figures", exist_ok=True)

    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"ROC Curve (AUC = {auc:.4f})")
    plt.savefig("reports/figures/roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved ROC curve to reports/figures/roc_curve.png")

    preds = (proba >= 0.5).astype(int)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()