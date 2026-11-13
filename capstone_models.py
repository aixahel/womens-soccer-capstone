"""
===============================================================================
 DS 785 Capstone - The Rise of Women's Soccer
 Presentation 4: Model Building and Evaluation
 Author: Aixa Helbringer-Escobar
===============================================================================

Run capstone_pipeline.py and capstone_eda.py first. Then:
  py -m pip install xgboost scikit-learn statsmodels matplotlib
  py capstone_models.py

In this script I build two kinds of models and score both only on the
held-out test seasons (seasons ending 2024-25):

  A. FORECASTING (the Objective 2 deliverable from my proposal)
     What I forecast: league-month average goals and xG per team-match in
             my five leagues.
     Models I compare: seasonal naive (my benchmark), linear regression
             (OLS baseline), ARIMA(p,0,q), Random Forest, XGBoost.
     How I tune: exhaustive grid search, scored by mean RMSE over k
             season-blocked, expanding-window folds.
     How I test: walk-forward, one month at a time; I refit every model on
             all months before the month it predicts.
     How I score: RMSE, MAE, MAPE, MASE, plus a Diebold-Mariano test against
             the seasonal naive (HLN correction, Holm-adjusted alpha = 0.05).

  B. MATCH-WIN CLASSIFIER (which performance profile separates wins)
     What I predict: match win (1/0) for each team-match in the StatsBomb data.
     Models I compare: training win rate (no-skill), logistic regression
             (baseline), Random Forest, XGBoost.
     How I tune: grid / randomized search with 5-fold stratified CV grouped
             by match on the training seasons (<= 2022), scored by log loss.
             I check the result on the 2023 validation season, refit on
             2018-2023 and test on 2024-25.
     How I score: ROC AUC, Brier score, log loss, accuracy and F1 at 0.5,
             plus paired bootstrap 95% CIs (2,000 resamples of matches)
             against logistic regression.

  After that I run the error analysis and feature importance (XGBoost SHAP
  values, permutation importance, logistic odds ratios), save the slide
  figures (figures/p4_*.png) and every number I quote
  (data/processed/model_results.json), and finish with PASS/FAIL checks.
"""

# =============================================================================
# SECTION 0 - SETUP
# =============================================================================
import itertools
import json
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.arima.model import ARIMA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)
T0 = time.time()

PROCESSED = "data/processed"
ALPHA = 0.05
SEED = 785
R = {}            # every number the slides quote ends up in here
CHECKS = []

LEAGUES = ["NWSL", "WSL", "Frauen Bundesliga", "Serie A Women", "Liga F"]

# I keep the same slide palette as Presentation 3 so the figures match my deck
INK, MUTED, GRID, SURFACE = "#16202E", "#6B7684", "#E3E6E4", "#F7F5EF"
ACCENT, SLATE = "#177A52", "#7B98AF"
LEAGUE_COLOR = dict(zip(LEAGUES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.linewidth": 2.2, "axes.titleweight": "bold", "axes.titlesize": 16,
})


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def p_fmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def check(name, ok):
    CHECKS.append((name, bool(ok)))


con = sqlite3.connect("data/capstone.db")


# =============================================================================
# SECTION A - FORECASTING LEAGUE-MONTH GOALS AND xG
# =============================================================================
print("[A] Forecasting league-month goals and xG per team-match...")

lm_raw = pd.read_sql("SELECT * FROM league_month_features", con)
lm_raw["ym"] = pd.PeriodIndex(lm_raw.year_month, freq="M")

# I can't use shuffled k-fold on a time series, because a model would train on
# months that come after the ones it's scored on. So I use season-blocked,
# expanding-window folds. My rule for k: one fold per season, using up to the
# five most recent pre-test seasons (2019-2023), and a season only counts if
# the series has at least two earlier seasons to learn from.
# Goals (data from 2013) -> k = 5. xG (data from 2019) -> k = 3 (2021-2023).
# I never touch the 2024-25 test seasons while tuning.
MAX_FOLDS = 5
TEST_FROM = 2024
FOLD_YEARS = []   # set per target in the loop below
ML_FEATURES = ["lag1", "lag2", "roll3", "prev_season_mean", "hist_mean",
               "month_of_season"] + [f"lg_{l}" for l in LEAGUES[1:]]

RF_GRID = [dict(max_depth=d, min_samples_leaf=m, max_features=f)
           for d in [3, 5, None] for m in [5, 10, 20] for f in [0.5, 1.0]]
XGB_GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=w)
            for d in [2, 3, 4] for lr in [0.03, 0.1] for n in [100, 300] for w in [1, 5]]
ARIMA_GRID = [(p, 0, q) for p in [0, 1, 2] for q in [0, 1, 2]]


def build_series(target):
    """I build one row per league-month where the target exists. Every
    feature I create here only looks at earlier months, so nothing from the
    month I'm predicting leaks in."""
    d = (lm_raw.dropna(subset=[target])
               [["league", "season_end_year", "year_month", "ym", "month_of_season", "fixtures", target]]
               .rename(columns={target: "y"}).sort_values(["league", "ym"]).reset_index(drop=True))
    g = d.groupby("league").y
    d["lag1"] = g.shift(1)
    d["lag2"] = g.shift(2)
    d["roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    d["hist_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    seas = d.groupby(["league", "season_end_year"]).y.mean().rename("season_mean").reset_index()
    seas["prev_season_mean"] = seas.groupby("league").season_mean.shift(1)
    d = d.merge(seas[["league", "season_end_year", "prev_season_mean"]], on=["league", "season_end_year"], how="left")
    # My seasonal naive benchmark: the same month of the previous season, same league
    sn = d[["league", "season_end_year", "month_of_season", "y"]].copy()
    sn["season_end_year"] += 1
    d = d.merge(sn.rename(columns={"y": "snaive"}), on=["league", "season_end_year", "month_of_season"], how="left")
    # When a lag doesn't exist yet, I fall back to the previous-season mean and then
    # the league's running mean, both of which only use past months
    d["snaive"] = d.snaive.fillna(d.prev_season_mean).fillna(d.hist_mean)
    for c in ["lag1", "lag2", "roll3", "prev_season_mean"]:
        d[c] = d[c].fillna(d.hist_mean)
    d = d.dropna(subset=["hist_mean"]).reset_index(drop=True)   # first month of each league
    for l in LEAGUES[1:]:
        d[f"lg_{l}"] = (d.league == l).astype(int)
    return d.sort_values(["ym", "league"]).reset_index(drop=True)


def make_ml(name, params):
    if name == "Linear regression":
        return LinearRegression()
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, **params)
    return xgb.XGBRegressor(subsample=0.8, random_state=SEED, n_jobs=4, **params)


def arima_onestep(hist, future, order, fallback):
    """Inside the CV folds I fit ARIMA once on the history, then make
    one-step-ahead predictions through the fold with the parameters held
    fixed. If a league has fewer than 12 past months, I use its running
    mean instead, the same fallback the other models get."""
    if len(hist) < 12:
        return np.asarray(fallback)
    res = ARIMA(hist, order=order, trend="c").fit()
    res_full = res.apply(np.concatenate([hist, future]))
    return np.asarray(res_full.fittedvalues)[len(hist):]


def arima_next(hist, order, fallback):
    if len(hist) < 12:
        return float(fallback)
    return float(ARIMA(hist, order=order, trend="c").fit().forecast(1)[0])


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def cv_score_ml(d, name, params):
    errs = []
    for y in FOLD_YEARS:
        tr, va = d[d.season_end_year < y], d[d.season_end_year == y]
        m = make_ml(name, params).fit(tr[ML_FEATURES], tr.y)
        errs.append(rmse(va.y - m.predict(va[ML_FEATURES])))
    return float(np.mean(errs)), errs


def cv_score_arima(d, order):
    errs = []
    for y in FOLD_YEARS:
        e = []
        for l in LEAGUES:
            dl = d[d.league == l]
            hist = dl[dl.season_end_year < y].y.values
            fut_rows = dl[dl.season_end_year == y]
            if len(fut_rows) == 0:
                continue
            e.append(fut_rows.y.values - arima_onestep(hist, fut_rows.y.values, order, fut_rows.hist_mean.values))
        errs.append(rmse(np.concatenate(e)))
    return float(np.mean(errs)), errs


def dm_test(e_model, e_bench):
    """I use the Diebold-Mariano test to check whether a model's forecast
    errors are really smaller than the benchmark's: squared-error loss,
    h = 1, with the Harvey-Leybourne-Newbold small-sample correction and a
    two-sided p-value from t(n - 1)."""
    dlt = np.square(e_model) - np.square(e_bench)
    n = len(dlt)
    dm = dlt.mean() / np.sqrt(dlt.var(ddof=0) / n)
    dm *= np.sqrt((n - 1) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


FC = {}
for target, label in [("goals_pg", "goals"), ("xg_pg", "xg")]:
    t1 = time.time()
    d = build_series(target)
    pre = d[d.season_end_year < TEST_FROM]
    test = d[d.season_end_year >= TEST_FROM].copy()
    seasons = sorted(pre.season_end_year.unique())
    FOLD_YEARS[:] = [y for y in seasons if sum(s < y for s in seasons) >= 2][-MAX_FOLDS:]
    out = {"n_rows": int(len(d)), "n_test": int(len(test)), "fold_years": list(map(int, FOLD_YEARS)),
           "test_by_league": test.groupby("league").size().to_dict()}

    # ---- Hyperparameter tuning: I try every grid setting and keep the one with
    #      the lowest mean RMSE across my folds ----
    best = {}
    tune_tbl = {}
    for name, grid in [("Random Forest", RF_GRID), ("XGBoost", XGB_GRID)]:
        scores = [(cv_score_ml(pre, name, p), p) for p in grid]
        (s, folds), p = min(scores, key=lambda z: z[0][0])
        best[name] = p
        tune_tbl[name] = {"candidates": len(grid), "best_params": {k: (None if v is None else v) for k, v in p.items()},
                          "cv_rmse": r2(s), "fold_rmse": [r2(f) for f in folds]}
    scores = [(cv_score_arima(pre, o), o) for o in ARIMA_GRID]
    (s, folds), o = min(scores, key=lambda z: z[0][0])
    best["ARIMA"] = o
    tune_tbl["ARIMA_all_orders_cv_rmse"] = {f"({a},{b},{c})": r2(sc[0]) for sc, (a, b, c) in scores}
    tune_tbl["ARIMA"] = {"candidates": len(ARIMA_GRID), "best_params": {"order": list(o)},
                         "cv_rmse": r2(s), "fold_rmse": [r2(f) for f in folds]}
    s, folds = cv_score_ml(pre, "Linear regression", {})
    tune_tbl["Linear regression"] = {"candidates": 1, "best_params": {}, "cv_rmse": r2(s),
                                     "fold_rmse": [r2(f) for f in folds]}
    sn_f = [rmse(pre[pre.season_end_year == y].y - pre[pre.season_end_year == y].snaive) for y in FOLD_YEARS]
    tune_tbl["Seasonal naive"] = {"candidates": 1, "best_params": {}, "cv_rmse": r2(np.mean(sn_f)),
                                  "fold_rmse": [r2(f) for f in sn_f]}
    out["tuning"] = tune_tbl

    # ---- Walk-forward test: before each test month I refit every model on all
    #      earlier months, then predict that month ----
    preds = {m: np.full(len(test), np.nan) for m in ["Linear regression", "ARIMA", "Random Forest", "XGBoost"]}
    preds["Seasonal naive"] = test.snaive.values
    leak_ok = True
    for ym in sorted(test.ym.unique()):
        idx = np.where(test.ym.values == ym)[0]
        hist = d[d.ym < ym]
        leak_ok &= bool(hist.ym.max() < ym)
        for name in ["Linear regression", "Random Forest", "XGBoost"]:
            m = make_ml(name, best.get(name, {})).fit(hist[ML_FEATURES], hist.y)
            preds[name][idx] = m.predict(test.iloc[idx][ML_FEATURES])
        for i in idx:
            l = test.league.iloc[i]
            preds["ARIMA"][i] = arima_next(hist[hist.league == l].y.values, best["ARIMA"], test.hist_mean.iloc[i])
    check(f"{label}: walk-forward never trains on the month it predicts", leak_ok)

    # For MASE I scale by the seasonal naive's in-sample MAE on 2013-2023
    # (Hyndman & Koehler), so anything below 1 beats my benchmark
    scale = float(np.mean(np.abs(pre.y - pre.snaive)))
    met, dm_rows = {}, {}
    for name, p in preds.items():
        e = test.y.values - p
        met[name] = {"RMSE": r2(rmse(e)), "MAE": r2(np.mean(np.abs(e))),
                     "MAPE": r2(100 * np.mean(np.abs(e) / test.y.values)),
                     "MASE": r2(np.mean(np.abs(e)) / scale),
                     "skill_vs_naive_pct": None}
        test[f"pred_{name}"] = p
    e_sn = test.y.values - preds["Seasonal naive"]
    for name in ["Linear regression", "ARIMA", "Random Forest", "XGBoost"]:
        e = test.y.values - preds[name]
        met[name]["skill_vs_naive_pct"] = r2(100 * (1 - np.mean(np.abs(e)) / np.mean(np.abs(e_sn))))
        dm_rows[name] = dm_test(e, e_sn)
    holm = multipletests([v[1] for v in dm_rows.values()], alpha=ALPHA, method="holm")[1]
    for (name, (stat, p)), ph in zip(dm_rows.items(), holm):
        met[name].update({"DM": r2(stat), "DM_p": p_fmt(p), "DM_p_holm": p_fmt(ph), "beats_naive": bool(ph < ALPHA and stat < 0)})
    out["metrics"] = met
    out["mase_scale"] = round(scale, 4)
    best_model = min(["Linear regression", "ARIMA", "Random Forest", "XGBoost"], key=lambda m: met[m]["RMSE"])
    out["best_model"] = best_model

    # ---- Error analysis: where does my best model miss, and why? ----
    test["err"] = test.y - test[f"pred_{best_model}"]
    test["abs_err"] = test.err.abs()
    out["rmse_by_league"] = {l: r2(rmse(g.err)) for l, g in test.groupby("league")}
    out["naive_rmse_by_league"] = {l: r2(rmse(g.y - g.snaive)) for l, g in test.groupby("league")}
    rho, p_rho = stats.spearmanr(test.fixtures, test.abs_err)
    out["spearman_abs_err_fixtures"] = {"rho": r2(rho), "p": p_fmt(p_rho)}
    edge = test.month_of_season.isin([1]) | (test.groupby(["league", "season_end_year"]).month_of_season.transform("max") == test.month_of_season)
    out["mae_first_last_month"] = r2(test[edge].abs_err.mean())
    out["mae_mid_season"] = r2(test[~edge].abs_err.mean())
    mw = stats.mannwhitneyu(test[edge].abs_err, test[~edge].abs_err, alternative="two-sided")
    out["mw_edge_vs_mid_p"] = p_fmt(mw.pvalue)
    out["n_edge_months"] = int(edge.sum())
    out["mean_bias"] = r2(test.err.mean())
    top = test.nlargest(5, "abs_err")[["league", "year_month", "fixtures", "y", f"pred_{best_model}", "err"]]
    out["top_errors"] = [{"league": r.league, "month": r.year_month, "fixtures": int(r.fixtures),
                          "actual": r2(r.y), "pred": r2(r[f"pred_{best_model}"]), "error": r2(r.err)}
                         for _, r in top.iterrows()]
    out["test_mean_actual"] = r2(test.y.mean())
    out["test_sd_actual"] = r2(test.y.std())

    # ---- Interpretability: I fit XGBoost on 2013-2023 and use SHAP values to see
    #      which features drive its test forecasts ----
    m = make_ml("XGBoost", best["XGBoost"]).fit(pre[ML_FEATURES], pre.y)
    contrib = m.get_booster().predict(xgb.DMatrix(test[ML_FEATURES]), pred_contribs=True)[:, :-1]
    shap_abs = pd.Series(np.abs(contrib).mean(axis=0), index=ML_FEATURES)
    league_sum = shap_abs[[c for c in ML_FEATURES if c.startswith("lg_")]].sum()
    shap_abs = shap_abs.drop([c for c in ML_FEATURES if c.startswith("lg_")])
    shap_abs["league (one-hot, summed)"] = league_sum
    out["shap_mean_abs"] = {k: round(float(v), 4) for k, v in shap_abs.sort_values(ascending=False).items()}
    out["shap_share_pct"] = {k: r2(100 * v / shap_abs.sum()) for k, v in shap_abs.sort_values(ascending=False).items()}
    FC[label] = {"out": out, "test": test, "pre": pre}
    R[f"forecast_{label}"] = out
    print(f"    {label}: n={len(d)} test={len(test)}  best={best_model}  ({time.time() - t1:.0f}s)")
    for name, v in met.items():
        print(f"      {name:18s} RMSE {v['RMSE']:.2f}  MAE {v['MAE']:.2f}  MAPE {v['MAPE']:.2f}%  "
              f"MASE {v['MASE']:.2f}  DM p(Holm) {v.get('DM_p_holm', '-')}")
    check(f"{label}: tuning folds use no test seasons", pre.season_end_year.max() < TEST_FROM)

# =============================================================================
# SECTION B - MATCH-WIN CLASSIFIER
# =============================================================================
print("[B] Match-win classifier...")
sb = pd.read_sql("SELECT * FROM sb_team_match_features", con)
CLF_FEATURES = ["xg", "xg_against", "shots", "shots_on_target", "shots_on_target_against",
                "possession_pct", "pass_pct", "pass_pct_against", "is_home"]
CLF_LABELS = {"xg": "xG created", "xg_against": "xG conceded", "shots": "Shots",
              "shots_on_target": "Shots on target", "shots_on_target_against": "Shots on target conceded",
              "possession_pct": "Possession %", "pass_pct": "Pass completion %",
              "pass_pct_against": "Opponent pass completion %", "is_home": "Home"}
sb = sb.dropna(subset=CLF_FEATURES + ["win"]).reset_index(drop=True)
tr = sb[sb.split == "train"]
va = sb[sb.split == "validation"]
te = sb[sb.split == "test"]
trva = pd.concat([tr, va])
R["clf_n"] = {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
              "test_matches": int(te.match_id.nunique()),
              "win_rate_train": r2(100 * tr.win.mean()), "win_rate_test": r2(100 * te.win.mean()),
              "draw_rate_test": r2(100 * (te.result == "D").mean())}
check("Classifier: no test match appears in training or validation",
      not set(te.match_id) & set(trva.match_id))

# I picked 5 folds: with 888 training rows that's about 178 team-matches and
# 75 wins per fold, enough for a stable log loss. 10 folds would leave only
# about 37 wins per fold. I group by match so both team rows of one game
# always land in the same fold.
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
searches = {
    "Logistic regression": GridSearchCV(logit, {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
                                        scoring="neg_log_loss", cv=cv),
    "Random Forest": GridSearchCV(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
                                  {"max_depth": [4, 6, None], "min_samples_leaf": [5, 20],
                                   "max_features": ["sqrt", 0.5]}, scoring="neg_log_loss", cv=cv),
    "XGBoost": RandomizedSearchCV(
        xgb.XGBClassifier(random_state=SEED, n_jobs=4, eval_metric="logloss"),
        {"max_depth": [2, 3, 4, 5], "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
         "n_estimators": [100, 200, 300, 400, 600], "subsample": [0.7, 0.85, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5, 10]},
        n_iter=30, scoring="neg_log_loss", cv=cv, random_state=SEED),
}
clf_tuning, final, val_scores = {}, {}, {}
for name, s in searches.items():
    s.fit(tr[CLF_FEATURES], tr.win, groups=tr.match_id)
    pv = s.best_estimator_.predict_proba(va[CLF_FEATURES])[:, 1]
    val_scores[name] = {"AUC": r2(roc_auc_score(va.win, pv)), "log_loss": r2(log_loss(va.win, pv)),
                        "Brier": round(float(brier_score_loss(va.win, pv)), 3)}
    clf_tuning[name] = {"candidates": len(s.cv_results_["params"]),
                        "best_params": {k.split("__")[-1]: v for k, v in s.best_params_.items()},
                        "cv_log_loss": r2(-s.best_score_)}
    # Once I've picked the settings, I refit them on train + validation
    # (2018-2023) before scoring the test seasons
    final[name] = clone(s.best_estimator_).fit(trva[CLF_FEATURES], trva.win)
R["clf_tuning"] = clf_tuning
R["clf_validation"] = val_scores

P = {"No-skill (training win rate)": np.full(len(te), trva.win.mean())}
for name, m in final.items():
    P[name] = m.predict_proba(te[CLF_FEATURES])[:, 1]
yt = te.win.values
clf_met = {}
for name, p in P.items():
    clf_met[name] = {"AUC": r2(roc_auc_score(yt, p)) if p.std() > 0 else 0.5,
                     "Brier": round(float(brier_score_loss(yt, p)), 3),
                     "log_loss": r2(log_loss(yt, p)),
                     "accuracy_pct": r2(100 * accuracy_score(yt, p >= 0.5)),
                     "F1": r2(f1_score(yt, p >= 0.5, zero_division=0))}

# To check whether the tree models really differ from logistic regression,
# I bootstrap the test matches 2,000 times and look at the paired difference
rng = np.random.default_rng(SEED)
mids = te.match_id.values
uniq = np.unique(mids)
pos = {m: np.where(mids == m)[0] for m in uniq}
boot = {n: {"AUC": [], "Brier": []} for n in ["Random Forest", "XGBoost"]}
for _ in range(2000):
    ix = np.concatenate([pos[m] for m in rng.choice(uniq, len(uniq), replace=True)])
    base_auc = roc_auc_score(yt[ix], P["Logistic regression"][ix])
    base_b = brier_score_loss(yt[ix], P["Logistic regression"][ix])
    for n in boot:
        boot[n]["AUC"].append(roc_auc_score(yt[ix], P[n][ix]) - base_auc)
        boot[n]["Brier"].append(brier_score_loss(yt[ix], P[n][ix]) - base_b)
for n in boot:
    for k in ["AUC", "Brier"]:
        lo, hi = np.percentile(boot[n][k], [2.5, 97.5])
        clf_met[n][f"{k}_diff_vs_logit_95CI"] = [round(float(lo), 3), round(float(hi), 3)]
        clf_met[n][f"{k}_diff_excludes_0"] = bool(lo > 0 or hi < 0)
R["clf_test"] = clf_met
best_clf = max(["Logistic regression", "Random Forest", "XGBoost"], key=lambda n: clf_met[n]["AUC"])
R["clf_best"] = best_clf
for n, v in clf_met.items():
    print(f"    {n:30s} AUC {v['AUC']:.2f}  Brier {v['Brier']:.3f}  log loss {v['log_loss']:.2f}  acc {v['accuracy_pct']:.2f}%")

# ---- Error analysis: which matches does XGBoost get wrong at a 0.5 threshold? ----
te = te.copy()
te["p"] = P["XGBoost"]
te["pred"] = (te.p >= 0.5).astype(int)
te["goals_minus_xg"] = te.goals - te.xg
te["opp_goals_minus_xg"] = te.goals_against - te.xg_against
cm = confusion_matrix(yt, te.pred)
fp = te[(te.pred == 1) & (te.win == 0)]
fn = te[(te.pred == 0) & (te.win == 1)]
tp = te[(te.pred == 1) & (te.win == 1)]
tn = te[(te.pred == 0) & (te.win == 0)]
R["clf_errors"] = {
    "confusion": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
    "fp_draw_share_pct": r2(100 * (fp.result == "D").mean()),
    "fn_goals_minus_xg_mean": r2(fn.goals_minus_xg.mean()),
    "tp_goals_minus_xg_mean": r2(tp.goals_minus_xg.mean()),
    "fp_opp_goals_minus_xg_mean": r2(fp.opp_goals_minus_xg.mean()),
    "tn_opp_goals_minus_xg_mean": r2(tn.opp_goals_minus_xg.mean()),
    "fn_xg_mean": r2(fn.xg.mean()), "fn_xg_against_mean": r2(fn.xg_against.mean()),
    "error_rate_by_league_pct": {l: r2(100 * (g.pred != g.win).mean()) for l, g in te.groupby("league")},
    "error_rate_close_xg_pct": r2(100 * (te[(te.xg - te.xg_against).abs() < 0.5].pred
                                         != te[(te.xg - te.xg_against).abs() < 0.5].win).mean()),
    "error_rate_clear_xg_pct": r2(100 * (te[(te.xg - te.xg_against).abs() >= 1].pred
                                         != te[(te.xg - te.xg_against).abs() >= 1].win).mean()),
    "share_close_xg_pct": r2(100 * ((te.xg - te.xg_against).abs() < 0.5).mean()),
}
te["xg_margin"] = pd.cut((te.xg - te.xg_against).abs(), [-0.01, 0.5, 1.0, 99],
                         labels=["< 0.5", "0.5 to 1", "1 or more"])
te["wrong"] = (te.pred != te.win).astype(int)
mtab = te.groupby("xg_margin", observed=True).agg(n=("wrong", "size"), err=("wrong", "mean"),
                                                   draw=("result", lambda r: (r == "D").mean()))
R["clf_errors"]["by_xg_margin"] = {str(k): {"n": int(v.n), "error_rate_pct": r2(100 * v.err),
                                             "draw_rate_pct": r2(100 * v.draw)} for k, v in mtab.iterrows()}
chi = stats.chi2_contingency(pd.crosstab(te.xg_margin, te.wrong))
R["clf_errors"]["chi2_margin"] = {"chi2": r2(chi[0]), "df": int(chi[2]), "p": p_fmt(chi[1])}
R["clf_errors"]["error_rate_draws_pct"] = r2(100 * te[te.result == "D"].wrong.mean())
R["clf_errors"]["error_rate_decisive_pct"] = r2(100 * te[te.result != "D"].wrong.mean())
mw_fn = stats.mannwhitneyu(fn.goals_minus_xg, tp.goals_minus_xg, alternative="two-sided")
R["clf_errors"]["mw_fn_vs_tp_finishing_p"] = p_fmt(mw_fn.pvalue)
te["ll"] = -(te.win * np.log(te.p) + (1 - te.win) * np.log(1 - te.p))
worst = te.nlargest(5, "ll")
R["clf_errors"]["worst5"] = [{"league": r.league, "season": r.season, "team": r.team, "opponent": r.opponent,
                              "score": f"{r.goals}-{r.goals_against}", "xg": r2(r.xg), "xg_against": r2(r.xg_against),
                              "p_win": r2(r.p)} for _, r in worst.iterrows()]

# ---- Interpretability: what drives the classifier? ----
booster = final["XGBoost"].get_booster()
contrib = booster.predict(xgb.DMatrix(te[CLF_FEATURES]), pred_contribs=True)
shap_vals = contrib[:, :-1]
check("XGBoost SHAP values add up to the model's log-odds",
      np.allclose(contrib.sum(axis=1), booster.predict(xgb.DMatrix(te[CLF_FEATURES]), output_margin=True), atol=1e-3))
shap_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=CLF_FEATURES).sort_values(ascending=False)
R["clf_shap_mean_abs"] = {CLF_LABELS[k]: r2(v) for k, v in shap_imp.items()}
R["clf_shap_share_pct"] = {CLF_LABELS[k]: r2(100 * v / shap_imp.sum()) for k, v in shap_imp.items()}
perm = {}
for name in ["Logistic regression", "Random Forest", "XGBoost"]:
    pi = permutation_importance(final[name], te[CLF_FEATURES], yt, scoring="roc_auc",
                                n_repeats=20, random_state=SEED)
    perm[name] = {CLF_LABELS[f]: round(float(v), 3) for f, v in zip(CLF_FEATURES, pi.importances_mean)}
R["clf_permutation_auc_drop"] = perm
lr = final["Logistic regression"].named_steps["logisticregression"]
R["clf_logit_or_per_sd"] = {CLF_LABELS[f]: r2(np.exp(c)) for f, c in zip(CLF_FEATURES, lr.coef_[0])}
# Here I find the xG value where the SHAP contribution of xG created turns
# positive; this is the benchmark I report on the slides
xg_shap = pd.DataFrame({"xg": te.xg.values, "s": shap_vals[:, CLF_FEATURES.index("xg")]}).sort_values("xg")
xg_shap["s_smooth"] = xg_shap.s.rolling(51, center=True, min_periods=10).mean()
cross = xg_shap[(xg_shap.s_smooth > 0)].xg.min()
R["clf_xg_shap_crossover"] = r2(cross)
R["test_mean_xg"] = r2(te.xg.mean())

# =============================================================================
# SECTION C - FIGURES
# =============================================================================
print("[C] Figures...")
MODEL_ORDER = ["Seasonal naive", "Linear regression", "ARIMA", "Random Forest", "XGBoost"]
MODEL_COLOR = {"Seasonal naive": "#B8BFC7", "Linear regression": SLATE, "ARIMA": "#4F6D86",
               "Random Forest": "#2C4A63", "XGBoost": ACCENT}

# p4_1 Forecast accuracy: test RMSE (with MASE) for every model, both targets
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=False)
for ax, lab, title in zip(axes, ["goals", "xg"], ["Goals per team-match", "xG per team-match"]):
    met = R[f"forecast_{lab}"]["metrics"]
    ys = np.arange(len(MODEL_ORDER))[::-1]
    vals = [met[m]["RMSE"] for m in MODEL_ORDER]
    ax.barh(ys, vals, color=[MODEL_COLOR[m] for m in MODEL_ORDER], height=0.62)
    for y_, v, m in zip(ys, vals, MODEL_ORDER):
        ax.text(v + 0.004, y_, f"{v:.2f}   MASE {met[m]['MASE']:.2f}", va="center", fontsize=13, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_title(title, loc="left"); ax.set_xlabel("Test RMSE (lower is better)")
    ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/p4_1_forecast_metrics.png", dpi=150); plt.close(fig)

# p4_2 Actual vs. forecast goals, one panel per league
lab = "goals"
tst = FC[lab]["test"]; best_m = FC[lab]["out"]["best_model"]
fig, axes = plt.subplots(1, 5, figsize=(18, 4.6), sharey=True)
for ax, l in zip(axes, LEAGUES):
    g = tst[tst.league == l].reset_index(drop=True)
    x = np.arange(len(g))
    ax.plot(x, g.y, "-o", color=INK, markersize=6, label="Actual")
    ax.plot(x, g["pred_Seasonal naive"], "--", color="#9AA3AD", linewidth=1.8, label="Seasonal naive")
    ax.plot(x, g[f"pred_{best_m}"], "-", color=ACCENT, label=best_m)
    ax.set_title(l, fontsize=14, loc="left")
    step = max(1, len(g) // 4)
    ax.set_xticks(x[::step]); ax.set_xticklabels(g.year_month[::step], rotation=45, fontsize=11)
axes[0].set_ylabel("Goals per team-match")
axes[0].legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout(); fig.savefig("figures/p4_2_goals_actual_vs_forecast.png", dpi=150); plt.close(fig)

# p4_3 Does forecast error depend on how many fixtures a month had?
fig, ax = plt.subplots(figsize=(8.5, 5.4))
for l in LEAGUES:
    g = tst[tst.league == l]
    ax.scatter(g.fixtures, g.abs_err, s=70, color=LEAGUE_COLOR[l], edgecolor=SURFACE, linewidth=1.5, label=l)
ax.set_xlabel("Fixtures played in the month"); ax.set_ylabel(f"|Error|, goals ({best_m})")
ax.legend(frameon=False, fontsize=11, ncol=2)
fig.tight_layout(); fig.savefig("figures/p4_3_forecast_error_vs_fixtures.png", dpi=150); plt.close(fig)

# p4_4 ROC curves for my three classifiers
CLF_ORDER = ["Logistic regression", "Random Forest", "XGBoost"]
CLF_COLOR = {"Logistic regression": SLATE, "Random Forest": "#2C4A63", "XGBoost": ACCENT}
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.plot([0, 1], [0, 1], "--", color="#9AA3AD", linewidth=1.5, label="No skill (AUC 0.50)")
for n in CLF_ORDER:
    f_, t_, _ = roc_curve(yt, P[n])
    ax.plot(f_, t_, color=CLF_COLOR[n], label=f"{n} (AUC {clf_met[n]['AUC']:.2f})")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(frameon=False, fontsize=12, loc="lower right")
fig.tight_layout(); fig.savefig("figures/p4_4_roc.png", dpi=150); plt.close(fig)

# p4_5 Calibration: predicted vs. observed win rate in 10 equal-count bins
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.plot([0, 1], [0, 1], "--", color="#9AA3AD", linewidth=1.5, label="Perfect calibration")
for n in CLF_ORDER:
    b = pd.qcut(P[n], 10, duplicates="drop")
    c = pd.DataFrame({"p": P[n], "y": yt}).groupby(b, observed=True).mean()
    ax.plot(c.p, c.y, "-o", color=CLF_COLOR[n], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=n)
ax.set_xlabel("Predicted win probability"); ax.set_ylabel("Observed win rate")
ax.legend(frameon=False, fontsize=12, loc="upper left")
fig.tight_layout(); fig.savefig("figures/p4_5_calibration.png", dpi=150); plt.close(fig)

# p4_6 Misclassifications: confusion matrix + error rate by xG margin
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#EAF3EE", ACCENT]))
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=20,
                color="#FFFFFF" if cm[i, j] > cm.max() * 0.6 else INK, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted not win", "Predicted win"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Not win", "Win"])
ax.grid(False); ax.set_title("XGBoost, threshold 0.5", loc="left")
ax = axes[1]
mt = R["clf_errors"]["by_xg_margin"]
labels = list(mt.keys())
xs = np.arange(len(labels))
vals = [mt[k]["error_rate_pct"] for k in labels]
ax.bar(xs, vals, color=[ACCENT if v == max(vals) else SLATE for v in vals], width=0.6)
for x_, k, v in zip(xs, labels, vals):
    ax.text(x_, v + 1, f"{v:.2f}%", ha="center", fontsize=13, color=INK)
    ax.text(x_, -6.5, f"n = {mt[k]['n']:,}", ha="center", fontsize=11, color=MUTED)
ax.set_xticks(xs); ax.set_xticklabels([f"|xG diff| {k}" for k in labels])
ax.set_ylim(-9, max(vals) * 1.25); ax.axhline(0, color=MUTED, linewidth=1)
ax.set_ylabel("Misclassified (%)")
ax.set_title("Errors concentrate in close-xG matches", loc="left")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("figures/p4_6_misclassifications.png", dpi=150); plt.close(fig)

# p4_7 Feature importance: XGBoost SHAP next to permutation importance for all three
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
order = list(shap_imp.index)[::-1]
ax = axes[0]
ax.barh([CLF_LABELS[f] for f in order], [shap_imp[f] for f in order], color=ACCENT, height=0.62)
ax.set_xlabel("Mean |SHAP value| (log-odds of a win)"); ax.set_title("XGBoost SHAP importance", loc="left")
ax.grid(axis="y", visible=False)
ax = axes[1]
ys = np.arange(len(order))
for k, n in enumerate(CLF_ORDER):
    ax.barh(ys + (k - 1) * 0.26, [perm[n][CLF_LABELS[f]] for f in order], height=0.24,
            color=CLF_COLOR[n], label=n)
ax.set_yticks(ys); ax.set_yticklabels([])
ax.set_xlabel("Drop in test AUC when the feature is shuffled"); ax.set_title("Permutation importance", loc="left")
ax.legend(frameon=False, fontsize=11, loc="lower right"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/p4_7_feature_importance.png", dpi=150); plt.close(fig)

# p4_8 How the SHAP contribution of xG changes with xG created
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.scatter(xg_shap.xg, xg_shap.s, s=14, color=SLATE, alpha=0.5, edgecolor="none")
ax.plot(xg_shap.xg, xg_shap.s_smooth, color=ACCENT, linewidth=2.6)
ax.axhline(0, color=MUTED, linewidth=1)
if cross == cross:
    ax.axvline(cross, color=INK, linestyle="--", linewidth=1.2)
    ax.annotate(f"Contribution turns positive\nat about {cross:.2f} xG", (cross, ax.get_ylim()[1] * 0.75),
                xytext=(10, 0), textcoords="offset points", fontsize=12, color=INK)
ax.set_xlabel("xG created in the match"); ax.set_ylabel("SHAP value of xG (log-odds of a win)")
fig.tight_layout(); fig.savefig("figures/p4_8_shap_xg_dependence.png", dpi=150); plt.close(fig)

# p4_9 What drives my goals forecast (XGBoost SHAP shares)
fig, ax = plt.subplots(figsize=(8.5, 5))
sh = pd.Series(R["forecast_goals"]["shap_share_pct"])
nice = {"lag1": "Last month", "lag2": "Two months ago", "roll3": "3-month rolling mean",
        "prev_season_mean": "Previous-season mean", "hist_mean": "League running mean",
        "month_of_season": "Month of season", "league (one-hot, summed)": "League"}
sh = sh.rename(index=nice).sort_values()
ax.barh(sh.index, sh.values, color=ACCENT, height=0.6)
for y_, v in enumerate(sh.values):
    ax.text(v + 0.5, y_, f"{v:.2f}%", va="center", fontsize=12, color=INK)
ax.set_xlim(0, sh.max() * 1.25)
ax.set_xlabel("Share of mean |SHAP|, goals forecast (XGBoost)"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/p4_9_forecast_shap.png", dpi=150); plt.close(fig)

# =============================================================================
# SECTION D - SAVE AND CHECK
# =============================================================================
for lab in ["goals", "xg"]:
    FC[lab]["test"].drop(columns=["ym"]).to_csv(f"{PROCESSED}/model_forecast_test_{lab}.csv", index=False)
te.to_csv(f"{PROCESSED}/model_classifier_test.csv", index=False)

m_g = R["forecast_goals"]["metrics"]
check("Forecast MASE = MAE / naive scale for every model",
      all(abs(m_g[m]["MASE"] - np.mean(np.abs(FC["goals"]["test"].y - FC["goals"]["test"][f"pred_{m}"])) / R["forecast_goals"]["mase_scale"]) < 0.01 for m in MODEL_ORDER))
check("Confusion matrix adds up to the test rows", cm.sum() == len(te))
check("Classifier features exclude goals and result (no leakage)",
      not {"goals", "goals_against", "result", "points"} & set(CLF_FEATURES))
check("Classifier tuning used training seasons only (<= 2022)", tr.season_end_year.max() <= 2022)
check("Every forecast test row is from 2024-25",
      all(FC[l]["test"].season_end_year.min() >= TEST_FROM for l in FC))

R["runtime_min"] = r2((time.time() - T0) / 60)
with open(f"{PROCESSED}/model_results.json", "w") as f:
    json.dump(R, f, indent=2, default=str)

print("[D] Checks")
for n, ok in CHECKS:
    print(f"    {'PASS' if ok else 'FAIL'}  {n}")
print(f"{sum(ok for _, ok in CHECKS)}/{len(CHECKS)} checks passed. "
      f"Figures in figures/p4_*.png, numbers in {PROCESSED}/model_results.json "
      f"({R['runtime_min']} min)")
