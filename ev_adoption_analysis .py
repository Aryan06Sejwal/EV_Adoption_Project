"""
Electric Vehicle Adoption Intelligence: Data-Driven Analysis and Growth Prediction Across Indian States
Subject: Fundamentals of Data Science | Dhruv Pandey (590028550), Aryan Sejwal (590023034)

HOW THIS FILE FOLLOWS THE COURSE PLAN (one section per stage, in this order)
  1. Problem understanding  (Unit I)   - problem, objective, CRISP-DM, "data not older than 2025"
  2. Data understanding     (Unit III) - data types, source, quality assessment and cleaning
  3. Exploratory analysis   (Unit III) - bar chart, histogram, scatter plot, correlation heatmap, more
  4. Feature engineering    (Unit IV)  - feature selection, feature transformation, PCA
  5. Machine learning       (Unit V)   - supervised: Linear Regression, Decision Tree, Logistic Regression, KNN
                                          unsupervised: K-Means, Hierarchical clustering
  6. Model evaluation       (Unit V)   - MAE/RMSE/R2, accuracy + confusion matrix, silhouette + inertia

1. PROBLEM UNDERSTANDING
   Problem statement : EV registration data for India exists but is rarely compared across states,
                       vehicle categories and time together, so it is hard to see who leads, what grows
                       fastest, what influences adoption and whether adoption can be predicted.
   Business objective: give policymakers, manufacturers and researchers evidence on where EV adoption is
                       strong or weak (RQ1-RQ5 in the report).
   CRISP-DM          : business understanding -> data understanding (section 2) -> data preparation
                       (sections 2 and 4) -> modelling (5) -> evaluation (6) -> deployment (the report).
   Data period       : calendar year 2025 only. Older rows are removed in section 2.

Input : data/ev_registrations.csv  (state, month, vehicle_category, ev_registrations, total_registrations)
Run   : python ev_adoption_analysis.py          (writes outputs/tables and outputs/figures)
"""
import argparse, json, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_predict, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42                                   # fixed so every run gives the same numbers
START = pd.Timestamp("2025-01-01")          # project rule: data not older than 2025
COLS = ["state", "month", "vehicle_category", "ev_registrations", "total_registrations"]
CATEGORIES = ["2W", "3W", "4W", "Bus", "Other"]
TIERS = ["Low", "Medium", "High"]
STATE_ALIASES = {"nct of delhi": "Delhi", "orissa": "Odisha", "pondicherry": "Puducherry",
                 "uttaranchal": "Uttarakhand"}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   saved", path)


# =========================================================================== #
# 2. DATA UNDERSTANDING: data types, quality assessment and cleaning
# =========================================================================== #
# Data types: state, vehicle_category = categorical (nominal); month = temporal;
# ev_registrations, total_registrations = numerical (discrete counts).
def clean_data(raw):
    """Assess and fix data quality. Returns the clean table and a quality log (report Section 3)."""
    df = raw[COLS].copy()
    log = [("Raw rows loaded", len(df), "None")]

    # inconsistencies: spelling variants of state names and category labels
    old = df["state"].astype(str).str.strip()
    key = old.str.replace("&", "and").str.replace(r"\s+", " ", regex=True).str.lower()
    df["state"] = [STATE_ALIASES.get(k, k.title().replace(" And ", " and ")) for k in key]
    log.append(("State names standardised", int((old != df["state"]).sum()),
                "Mapped to one official name each"))
    cat = {c.upper(): c for c in CATEGORIES}
    df["vehicle_category"] = df["vehicle_category"].astype(str).str.strip().str.upper().map(cat).fillna("Other")

    # types
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    for c in ["ev_registrations", "total_registrations"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # period rule
    too_old = df["month"] < START
    log.append(("Months before 2025 removed", int(too_old.sum()), "Removed (project rule: data not older than 2025)"))
    df = df[~too_old]

    # duplicate records
    dup = df.duplicated(subset=["state", "month", "vehicle_category"], keep="first")
    log.append(("Duplicate rows removed", int(dup.sum()), "Kept first record per state, month and category"))
    df = df[~dup]

    # missing values: a missing EV count with a known total means zero EVs; other gaps are dropped
    n_missing = int(df[COLS].isna().any(axis=1).sum())
    zero_fill = df["ev_registrations"].isna() & df["total_registrations"].notna()
    df.loc[zero_fill, "ev_registrations"] = 0
    dropped = int(df[COLS].isna().any(axis=1).sum())
    df = df.dropna(subset=COLS)
    log.append(("Rows with missing values", n_missing,
                f"{int(zero_fill.sum())} EV counts set to 0 (total present); {dropped} rows dropped"))

    # impossible values: negative counts are removed, EV above total is capped at total
    neg = (df["ev_registrations"] < 0) | (df["total_registrations"] < 0)
    df = df[~neg].copy()
    over = df["ev_registrations"] > df["total_registrations"]
    df.loc[over, "ev_registrations"] = df.loc[over, "total_registrations"]
    log.append(("Impossible values fixed", int(neg.sum() + over.sum()),
                f"{int(neg.sum())} negative-count rows removed; {int(over.sum())} rows with EV > total capped at total"))

    # outliers: IQR rule on log values within each vehicle category. Flag only: large states are real.
    x = np.log1p(df["ev_registrations"])
    g = x.groupby(df["vehicle_category"])
    q1, q3 = g.transform(lambda v: v.quantile(0.25)), g.transform(lambda v: v.quantile(0.75))
    df["outlier_flag"] = (x > q3 + 1.5 * (q3 - q1)) | (x < q1 - 1.5 * (q3 - q1))
    log.append(("Outlier rows flagged (kept)", int(df["outlier_flag"].sum()), "None (flag only)"))

    df = df.sort_values(["month", "state", "vehicle_category"]).reset_index(drop=True)
    log.append(("Clean rows used for analysis", len(df), "None"))
    return df, pd.DataFrame(log, columns=["quality_check", "rows_affected", "action_taken"])


def state_table(clean):
    """One row per state: totals, EV share, vehicle mix and half-over-half growth."""
    st = clean.groupby("state").agg(total_ev=("ev_registrations", "sum"),
                                    total_vehicles=("total_registrations", "sum"))
    st["ev_share_pct"] = 100 * st["total_ev"] / st["total_vehicles"]
    mix = clean.pivot_table(index="state", columns="vehicle_category", values="ev_registrations",
                            aggfunc="sum", fill_value=0)
    for c in ["2W", "3W", "4W"]:
        st[f"share_{c.lower()}"] = mix[c] / st["total_ev"]
    st["growth_h2_vs_h1_pct"] = half_growth(clean, "state")
    st["log_total_ev"] = np.log1p(st["total_ev"])          # transformation: tames the right skew
    return st.reset_index()


def half_growth(clean, by):
    """Percent change in EV registrations, second half of the period vs first half."""
    months = sorted(clean["month"].unique())
    h = len(months) // 2
    a = clean[clean["month"].isin(months[:h])].groupby(by)["ev_registrations"].sum()
    b = clean[clean["month"].isin(months[-h:])].groupby(by)["ev_registrations"].sum()
    return 100 * (b - a) / a


# =========================================================================== #
# 3. EXPLORATORY DATA ANALYSIS: six figures, each tied to a research question
# =========================================================================== #
def eda(clean, st, fig_dir):
    # 01 bar chart (RQ1): top states by volume and by share
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, col, color, title in [(ax[0], "total_ev", "#1b9e77", "Top 15 states by EV registrations"),
                                 (ax[1], "ev_share_pct", "#7570b3", "Top 15 states by EV share of all registrations")]:
        top = st.nlargest(15, col).iloc[::-1]
        a.barh(top["state"], top[col], color=color)
        a.set_title(title)
    save(fig, fig_dir / "01_state_bar.png")

    # 02 bar chart (RQ2): vehicle categories, volume and growth
    vol = clean.groupby("vehicle_category")["ev_registrations"].sum().reindex(CATEGORIES)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].bar(vol.index, vol.values, color="#1b9e77")
    ax[0].set_title("EV registrations by vehicle category")
    gr = half_growth(clean, "vehicle_category").reindex(CATEGORIES)
    ax[1].bar(gr.index, gr.values, color="#d95f02")
    ax[1].set_title("Growth: second half vs first half (%)")
    save(fig, fig_dir / "02_category_comparison.png")

    # 03 line chart (RQ3): monthly trend
    mc = clean.pivot_table(index="month", columns="vehicle_category", values="ev_registrations", aggfunc="sum")
    tot = clean.groupby("month")[["ev_registrations", "total_registrations"]].sum()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in CATEGORIES:
        ax[0].plot(mc.index, mc[c], marker="o", label=c)
    ax[0].set_title("National EV registrations by month and category")
    ax[0].legend()
    ax[1].plot(tot.index, 100 * tot["ev_registrations"] / tot["total_registrations"], marker="o", color="#7570b3")
    ax[1].set_title("EV share of all registrations by month (%)")
    for a in ax:
        a.tick_params(axis="x", rotation=45)
    save(fig, fig_dir / "03_monthly_trend.png")

    # 04 histograms: distribution of state values (raw, log, share)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].hist(st["total_ev"], bins=9, color="#1b9e77"); ax[0].set_title("State EV registrations (raw)")
    ax[1].hist(np.log10(st["total_ev"]), bins=9, color="#d95f02"); ax[1].set_title("State EV registrations (log10)")
    ax[2].hist(st["ev_share_pct"], bins=9, color="#7570b3"); ax[2].set_title("State EV share (%)")
    save(fig, fig_dir / "04_histogram_state_ev.png")

    # 05 scatter plot (RQ4): market size vs EV registrations, log-log trend line
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(st["total_vehicles"], st["total_ev"], color="#1b9e77", s=45)
    slope, icept = np.polyfit(np.log10(st["total_vehicles"]), np.log10(st["total_ev"]), 1)
    xs = np.linspace(st["total_vehicles"].min(), st["total_vehicles"].max(), 100)
    ax.plot(xs, 10 ** (icept + slope * np.log10(xs)), color="#d95f02", lw=2, label="log-log trend")
    for _, r in st.nlargest(5, "total_ev").iterrows():
        ax.annotate(r["state"], (r["total_vehicles"], r["total_ev"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="Total vehicle registrations (market size)", ylabel="EV registrations",
           title="Market size vs EV registrations")
    ax.legend()
    save(fig, fig_dir / "05_scatter_charging_vs_ev.png")   # file name kept so the report's Figure 4.5 still matches

    # 06 correlation heatmap (RQ4)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(st.drop(columns=["state", "log_total_ev"]).corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation heatmap of state-level features")
    save(fig, fig_dir / "06_correlation_heatmap.png")


def statistics(clean, st, tab_dir):
    """Descriptive statistics, normality, hypothesis tests T1 and T2, and correlations (report Section 5)."""
    cols = ["total_ev", "ev_share_pct", "growth_h2_vs_h1_pct"]
    d = st[cols].describe().T
    d["median"], d["skewness"] = st[cols].median(), st[cols].skew()
    d.to_csv(tab_dir / "descriptive_stats.csv", index_label="variable")
    st.drop(columns="state").corr().to_csv(tab_dir / "correlation_matrix.csv")

    rows = []
    for c in cols:                                         # Shapiro-Wilk: is the variable normal?
        w, p = stats.shapiro(st[c])
        rows.append({"test": "Shapiro-Wilk (raw)", "variable": c, "statistic": w, "p_value": p, "n": len(st)})
        if (st[c] >= 0).all():
            w, p = stats.shapiro(np.log1p(st[c]))
            rows.append({"test": "Shapiro-Wilk (log1p)", "variable": c, "statistic": w, "p_value": p, "n": len(st)})

    months = sorted(clean["month"].unique())               # T1: first half vs second half, paired by state
    h = len(months) // 2
    a = clean[clean["month"].isin(months[:h])].groupby("state")["ev_registrations"].sum()
    b = clean[clean["month"].isin(months[-h:])].groupby("state")["ev_registrations"].sum()
    w, p = stats.wilcoxon(a, b)
    rows.append({"test": "T1 Wilcoxon signed-rank: first vs second half", "variable": "ev_registrations by state",
                 "statistic": w, "p_value": p, "n": len(a)})

    cs = clean.groupby(["state", "vehicle_category"])[["ev_registrations", "total_registrations"]].sum().reset_index()
    cs["share"] = cs["ev_registrations"] / cs["total_registrations"]      # T2: does EV share differ by category?
    groups = [g["share"].values for _, g in cs.groupby("vehicle_category")]
    h_stat, p = stats.kruskal(*groups)
    rows.append({"test": "T2 Kruskal-Wallis: EV share across categories", "variable": "ev share by category",
                 "statistic": h_stat, "p_value": p, "n": sum(len(g) for g in groups)})
    pd.DataFrame(rows).to_csv(tab_dir / "stat_tests.csv", index=False)

    pairs = []
    for x, y in [("total_vehicles", "total_ev"), ("share_2w", "ev_share_pct")]:
        r, pr = stats.pearsonr(st[x], st[y])
        rho, ps = stats.spearmanr(st[x], st[y])
        pairs.append({"pair": f"{x} vs {y}", "pearson_r": r, "pearson_p": pr,
                      "spearman_rho": rho, "spearman_p": ps, "n": len(st)})
    pd.DataFrame(pairs).to_csv(tab_dir / "correlation_pairs.csv", index=False)


# =========================================================================== #
# 4. FEATURE ENGINEERING: transformation, selection, dimensionality reduction
# =========================================================================== #
def monthly_panel(clean):
    """State x month table for forecasting. Uses LAST month's values as inputs (known at forecast time)."""
    p = clean.groupby(["state", "month"])[["ev_registrations", "total_registrations"]].sum().reset_index()
    months = sorted(p["month"].unique())
    p["month_idx"] = p["month"].map({m: i + 1 for i, m in enumerate(months)})
    prev = p[["state", "month_idx", "ev_registrations", "total_registrations"]].copy()
    prev["month_idx"] += 1                                  # shift by month number, so gaps cannot mislead
    prev.columns = ["state", "month_idx", "lag_1_ev", "lag_1_total"]
    p = p.merge(prev, on=["state", "month_idx"]).copy()     # first month has no previous month and drops out
    p["log_lag_1_total"] = np.log1p(p["lag_1_total"])       # transformation: log of last month's market size
    p["log_lag_1_ev"] = np.log1p(p["lag_1_ev"])             # transformation: log of last month's EVs
    p["target_log_ev"] = np.log1p(p["ev_registrations"])
    return p.sort_values(["month_idx", "state"]).reset_index(drop=True), len(months)


def select_features(X, y, tab_dir):
    """Feature selection: F-test, mutual information and a correlation check (report Section 6.3)."""
    f, pf = f_regression(X, y)
    mi = mutual_info_regression(X, y, random_state=SEED)
    corr = X.corr().abs()
    maxc = [float(corr.loc[c].drop(c).max()) for c in X.columns]
    out = pd.DataFrame({"feature": X.columns, "f_score": f, "f_p_value": pf, "mutual_information": mi,
                        "max_abs_corr_with_other_feature": maxc})
    out["flag"] = np.where(out["max_abs_corr_with_other_feature"] > 0.9, "highly correlated (>0.9) - check", "ok")
    out.to_csv(tab_dir / "feature_scores.csv", index=False)


def drop_correlated(X, threshold=0.9):
    """Feature selection: of any pair with |r| above the threshold, keep the first and drop the second."""
    corr, drop = X.corr().abs(), set()
    cols = list(X.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a not in drop and b not in drop and corr.loc[a, b] > threshold:
                drop.add(b)
    return [c for c in cols if c not in drop], sorted(drop)


def pca_analysis(Z, features, tab_dir, fig_dir):
    """Dimensionality reduction: how many components explain 90% of the variance (report Section 6.4)."""
    pca = PCA(random_state=SEED).fit(Z)
    ev, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    rows = [{"component": f"PC{i + 1}", "variance_explained": ev[i], "cumulative_variance": cum[i],
             "top_features": ", ".join(pd.Series(np.abs(pca.components_[i]), index=features).nlargest(3).index)}
            for i in range(len(ev))]
    pd.DataFrame(rows).to_csv(tab_dir / "pca_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(1, len(ev) + 1), ev, color="#1b9e77", label="per component")
    ax.plot(range(1, len(ev) + 1), cum, marker="o", color="#d95f02", label="cumulative")
    ax.axhline(0.9, ls="--", color="grey")
    ax.set(xlabel="Principal component", ylabel="Variance explained", title=f"PCA scree plot ({n90} components reach 90%)")
    ax.legend()
    save(fig, fig_dir / "07_pca_scree.png")
    return n90


# =========================================================================== #
# 5 + 6. SUPERVISED MODELS AND THEIR EVALUATION
# =========================================================================== #
def regression(panel, n_months, tab_dir, fig_dir):
    """Predict monthly EV registrations. Train on early months, test on the last quarter (no peeking ahead).
    Models: Linear Regression, Decision Tree. Metrics: MAE, RMSE, R2, compared with two naive baselines."""
    feats = ["month_idx", "log_lag_1_total", "log_lag_1_ev"]
    cut = n_months - max(1, n_months // 4)
    train, test = panel[panel["month_idx"] <= cut], panel[panel["month_idx"] > cut]
    print(f"   train months 1-{cut} ({len(train)} rows) | test months {cut + 1}-{n_months} ({len(test)} rows)")
    Xtr, ytr, Xte, actual = train[feats], train["target_log_ev"], test[feats], test["ev_registrations"].values
    select_features(Xtr, ytr, tab_dir)

    back = lambda z: np.clip(np.expm1(z), 0, None)          # undo the log transformation
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
    search = GridSearchCV(DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 3, 5, 10]},
                          cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error").fit(Xtr, ytr)
    tree = search.best_estimator_
    print("   Decision Tree best params:", search.best_params_)
    preds = {"Baseline: same as last month": back(Xte["log_lag_1_ev"].values),
             "Baseline: training mean": np.full(len(test), train["ev_registrations"].mean()),
             "Linear Regression": back(lr.predict(Xte)),
             "Decision Tree Regressor": back(tree.predict(Xte))}
    res = pd.DataFrame([{"model": n, "MAE": mean_absolute_error(actual, p),
                         "RMSE": float(np.sqrt(mean_squared_error(actual, p))), "R2": r2_score(actual, p)}
                        for n, p in preds.items()])
    res.to_csv(tab_dir / "regression_metrics.csv", index=False)
    pd.DataFrame({"feature": feats, "linear_regression_standardised_coef": lr[-1].coef_,
                  "decision_tree_importance": tree.feature_importances_}).to_csv(tab_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))                 # figure 10: actual vs predicted
    ax.scatter(actual + 1, preds["Linear Regression"] + 1, alpha=0.6, color="#1b9e77", label="Linear Regression")
    ax.scatter(actual + 1, preds["Decision Tree Regressor"] + 1, alpha=0.6, color="#d95f02", marker="^", label="Decision Tree")
    top = max(actual.max(), preds["Linear Regression"].max(), preds["Decision Tree Regressor"].max()) + 1
    ax.plot([1, top], [1, top], "k--", lw=1, label="perfect prediction")
    ax.set(xscale="log", yscale="log", xlabel="Actual monthly EV registrations (+1)", ylabel="Predicted (+1)",
           title="Actual vs predicted (test months)")
    ax.legend()
    save(fig, fig_dir / "10_actual_vs_pred.png")
    fig, ax = plt.subplots(figsize=(15, 7))                  # figure 11: the fitted tree
    plot_tree(tree, feature_names=feats, filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Decision Tree Regressor (top 3 levels; target = log(1 + EV registrations))")
    save(fig, fig_dir / "11_decision_tree.png")
    return res


def classification(st, tab_dir, fig_dir):
    """Classify states into Low / Medium / High EV-share tiers.
    Models: Decision Tree, Logistic Regression, KNN. Metrics: accuracy, macro F1, confusion matrix."""
    d = st.copy()
    d["tier"] = pd.qcut(d["ev_share_pct"].rank(method="first"), 3, labels=TIERS)   # 6 states per tier
    feats = ["share_2w", "share_3w", "growth_h2_vs_h1_pct", "log_total_ev"]          # EV share itself is excluded
    X, y = d[feats], d["tier"].astype(str)
    fill = SimpleImputer(strategy="median")                  # fills gaps inside each fold, so nothing leaks
    models = {"Decision Tree Classifier": make_pipeline(fill, DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=SEED)),
              "Logistic Regression": make_pipeline(fill, StandardScaler(), LogisticRegression(max_iter=2000)),
              "KNN": make_pipeline(fill, StandardScaler(), KNeighborsClassifier(n_neighbors=5))}
    folds = 5                                                # repeated 10 times: 18 states are too few for one split
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=10, random_state=SEED)
    rows, pooled = [], {}
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({"model": name, "accuracy_mean": r["test_accuracy"].mean(), "accuracy_std": r["test_accuracy"].std(),
                     "macro_f1_mean": r["test_f1_macro"].mean(), "macro_f1_std": r["test_f1_macro"].std()})
        pooled[name] = cross_val_predict(m, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=SEED))
    res = pd.DataFrame(rows)
    res.to_csv(tab_dir / "classification_metrics.csv", index=False)
    best = res.sort_values("accuracy_mean", ascending=False).iloc[0]["model"]
    print("   NOTE: the best model is picked on the same scores that are reported, so its accuracy is slightly optimistic.")
    cm = confusion_matrix(y, pooled[best], labels=TIERS)
    pd.DataFrame(cm, index=[f"actual_{t}" for t in TIERS], columns=[f"pred_{t}" for t in TIERS]).to_csv(tab_dir / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TIERS, yticklabels=TIERS, ax=ax)
    ax.set(xlabel="Predicted tier", ylabel="Actual tier", title=f"Confusion matrix: {best}\n(cross-validated predictions)")
    save(fig, fig_dir / "12_confusion_matrix.png")
    return res, best


# =========================================================================== #
# 5 + 6. UNSUPERVISED MODELS AND THEIR EVALUATION
# =========================================================================== #
def clustering(st, tab_dir, fig_dir):
    """Group states into adoption profiles. Models: K-Means and hierarchical (Ward).
    Metrics: inertia (K-Means) and silhouette score (both); k is chosen by the best K-Means silhouette."""
    pool = ["ev_share_pct", "log_total_ev", "growth_h2_vs_h1_pct", "share_2w", "share_3w", "share_4w"]
    d = st.copy()
    d[pool] = d[pool].fillna(d[pool].median())
    kept, dropped = drop_correlated(d[pool])                 # feature selection before clustering
    print("   correlation filter (>0.9) dropped:", dropped)
    Z = StandardScaler().fit_transform(d[kept])              # transformation: put all features on one scale
    n90 = pca_analysis(Z, kept, tab_dir, fig_dir)            # dimensionality reduction (figure 07)

    rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_kmeans": silhouette_score(Z, km.labels_),
                     "silhouette_hierarchical": silhouette_score(Z, hc.labels_)})
    sel = pd.DataFrame(rows)
    sel.to_csv(tab_dir / "kmeans_selection.csv", index=False)
    best_k = int(sel.loc[sel["silhouette_kmeans"].idxmax(), "k"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))          # figure 08: elbow and silhouette
    ax[0].plot(sel["k"], sel["inertia"], marker="o", color="#1b9e77")
    ax[0].set(title="Elbow method: inertia vs k", xlabel="k", ylabel="Inertia")
    ax[1].plot(sel["k"], sel["silhouette_kmeans"], marker="o", color="#d95f02", label="K-Means")
    ax[1].plot(sel["k"], sel["silhouette_hierarchical"], marker="s", color="#7570b3", label="Hierarchical")
    ax[1].set(title="Silhouette score vs k", xlabel="k")
    ax[1].legend()
    save(fig, fig_dir / "08_kmeans_selection.png")

    d["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Z).labels_
    d[["state", "cluster"]].to_csv(tab_dir / "cluster_assignments.csv", index=False)
    summary = d.groupby("cluster")[kept + ["total_ev"]].mean().round(3)
    summary.insert(0, "n_states", d.groupby("cluster").size())
    summary.insert(1, "example_states", d.sort_values("total_ev", ascending=False).groupby("cluster")["state"]
                   .apply(lambda s: ", ".join(s.head(4))))
    summary["profile_name"] = ""                             # name the clusters yourself after reading the table
    summary.to_csv(tab_dir / "cluster_summary.csv")

    P = PCA(n_components=2, random_state=SEED).fit_transform(Z)    # figure 09: clusters on the first two components
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(P[:, 0], P[:, 1], c=d["cluster"], cmap="viridis", s=60)
    for i, s in enumerate(d["state"]):
        ax.annotate(s, (P[i, 0], P[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"K-Means clusters (k={best_k}) on the first two principal components")
    ax.legend(*sc.legend_elements(), title="Cluster")
    save(fig, fig_dir / "09_clusters_pca.png")
    return sel, best_k, n90


# =========================================================================== #
# MAIN: run the stages in order
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent          # folders are found next to this file, wherever you run it from
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--out-dir", default=str(here / "outputs"))
    a = ap.parse_args()
    fig_dir, tab_dir = Path(a.out_dir) / "figures", Path(a.out_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("[2] Data understanding: quality assessment and cleaning")
    csv_path = Path(a.data_dir) / "ev_registrations.csv"
    if not csv_path.exists():
        raise SystemExit(f"Cannot find {csv_path}\nPut ev_registrations.csv in a folder named 'data' next to this script.")
    clean, quality = clean_data(pd.read_csv(csv_path))
    quality.to_csv(tab_dir / "data_quality_report.csv", index=False)
    print(quality.to_string(index=False))
    st = state_table(clean)
    st.to_csv(tab_dir / "state_features.csv", index=False)
    print(f"   {len(st)} states, {clean['month'].nunique()} months")

    print("[3] Exploratory data analysis and statistics")
    eda(clean, st, fig_dir)
    statistics(clean, st, tab_dir)

    print("[4-6] Feature engineering, models and evaluation")
    panel, n_months = monthly_panel(clean)
    print("Regression (Linear Regression, Decision Tree)")
    reg = regression(panel, n_months, tab_dir, fig_dir)
    print("Classification (Decision Tree, Logistic Regression, KNN)")
    cls, best_clf = classification(st, tab_dir, fig_dir)
    print("Clustering (K-Means, hierarchical) and PCA")
    sel, best_k, n90 = clustering(st, tab_dir, fig_dir)

    # Model comparison: each task is compared with its own metrics (they cannot be ranked on one scale)
    print("\nMODEL COMPARISON")
    print("Regression (MAE, RMSE, R2):\n", reg.round(3).to_string(index=False))
    print("Classification (accuracy, macro F1; chance = 0.33):\n", cls.round(3).to_string(index=False))
    print(f"Clustering (best k = {best_k}; inertia, silhouette):\n", sel.round(3).to_string(index=False))
    summary = {"states": len(st), "rows_clean": len(clean), "best_classifier": best_clf, "best_k": best_k,
               "pca_components_for_90pct": n90, "regression": reg.round(4).to_dict("records"),
               "classification": cls.round(4).to_dict("records"), "clustering": sel.round(4).to_dict("records")}
    (tab_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Tables: {tab_dir}  Figures: {fig_dir}")


if __name__ == "__main__":
    main()
