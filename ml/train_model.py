"""
Readmission prediction — train & evaluate, capture real numbers + plots.
Outputs metrics JSON + confusion matrix + feature importance PNGs.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, RocCurveDisplay)

RNG = 42
df = pd.read_csv("data/patients.csv")

# ----- features / target -----
TARGET = "readmitted_30d"
num_feats = ["age", "length_of_stay"]
cat_feats = ["gender", "department", "admission_type", "diagnosis"]
X = df[num_feats + cat_feats]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RNG, stratify=y)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ("num", "passthrough", num_feats),
])

results = {}

# ----- Model 1: Logistic Regression (baseline, interpretable) -----
logreg = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
logreg.fit(X_train, y_train)
p_lr = logreg.predict(X_test); pr_lr = logreg.predict_proba(X_test)[:,1]
results["logistic_regression"] = {
    "accuracy": round(accuracy_score(y_test, p_lr), 3),
    "roc_auc": round(roc_auc_score(y_test, pr_lr), 3),
}

# ----- Model 2: Random Forest (stronger, gives feature importance) -----
rf = Pipeline([("pre", pre),
               ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                        class_weight="balanced", random_state=RNG))])
rf.fit(X_train, y_train)
p_rf = rf.predict(X_test); pr_rf = rf.predict_proba(X_test)[:,1]
results["random_forest"] = {
    "accuracy": round(accuracy_score(y_test, p_rf), 3),
    "roc_auc": round(roc_auc_score(y_test, pr_rf), 3),
}

# ----- Confusion matrix (RF) -----
cm = confusion_matrix(y_test, p_rf)
results["confusion_matrix_rf"] = cm.tolist()
results["test_size"] = int(len(y_test))
results["positive_rate"] = round(float(y.mean()), 3)

fig, ax = plt.subplots(figsize=(4.2,3.6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["No","Yes"]); ax.set_yticklabels(["No","Yes"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Random Forest")
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# ----- Feature importance (RF) -----
ohe = rf.named_steps["pre"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cat_feats))
feat_names = cat_names + num_feats
imp = rf.named_steps["clf"].feature_importances_
order = np.argsort(imp)[::-1][:12]
top_names = [feat_names[i] for i in order]; top_imp = imp[order]

fig, ax = plt.subplots(figsize=(7,4.2))
ax.barh(range(len(top_imp))[::-1], top_imp, color="#8C4FFF")
ax.set_yticks(range(len(top_imp))[::-1]); ax.set_yticklabels(top_names, fontsize=8)
ax.set_xlabel("Importance"); ax.set_title("Top factors driving 30-day readmission")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

results["top_features"] = [(top_names[i], round(float(top_imp[i]),3)) for i in range(min(6,len(top_imp)))]

# ROC curve
fig, ax = plt.subplots(figsize=(4.6,4))
RocCurveDisplay.from_predictions(y_test, pr_rf, name=f"RF (AUC={results['random_forest']['roc_auc']})", ax=ax)
RocCurveDisplay.from_predictions(y_test, pr_lr, name=f"LogReg (AUC={results['logistic_regression']['roc_auc']})", ax=ax)
ax.plot([0,1],[0,1],"--",color="gray",lw=1)
ax.set_title("ROC Curve"); plt.tight_layout(); plt.savefig("roc_curve.png", dpi=150); plt.close()

print(json.dumps(results, indent=2))
json.dump(results, open("ai_results.json","w"), indent=2)
