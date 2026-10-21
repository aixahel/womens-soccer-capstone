"""
===============================================================================
 DS 785 Capstone - The Rise of Women's Soccer
 Presentation 3: Exploratory Data Analysis and Feature Engineering
 Author: Aixa Helbringer-Escobar
===============================================================================

Run capstone_pipeline.py first (it builds data/capstone.db and data/processed/).
Then:
  py -m pip install pymannkendall scikit-learn matplotlib
  py capstone_eda.py

What this script does:
  1. Descriptive statistics and distribution shape (skew, kurtosis,
     Shapiro-Wilk, Poisson dispersion for goal counts)
  2. Correlation analysis (Spearman matrix, VIF for multicollinearity)
  3. Stage 1 of my screen: Mann-Kendall trend test per metric, alpha = 0.05,
     per league and pooled across leagues (Regional Kendall test)
  4. Stage 2 of my screen: logistic regression of match win on each metric,
     Wald test, Holm-adjusted alpha = 0.05, standard errors clustered by match
  5. Feature engineering (ratios, opponent differentials, competitive
     balance, lag/rolling features for the league-month series)
  6. Stationarity of the league-month series (ADF + KPSS), for ARIMA
  7. PCA on the standardized match-level features (fit on training seasons)
  8. Figures for the slides (figures/) and every quoted number
     (data/processed/eda_results.json), then a PASS/FAIL check of the numbers
"""

# =============================================================================
# SECTION 0 - SETUP
# =============================================================================
import json
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

PROCESSED = "data/processed"
ALPHA = 0.05
R = {}   # every number the slides quote ends up in here

LEAGUES = ["NWSL", "WSL", "Frauen Bundesliga", "Serie A Women", "Liga F"]

# Slide palette: same deck colors as Presentations 1-2, plus five league
# hues in a fixed order (validated for colorblind separation as line series)
INK, MUTED, GRID, SURFACE = "#16202E", "#6B7684", "#E3E6E4", "#F7F5EF"
ACCENT = "#177A52"
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
    """My standing rule: round reported numbers to two decimals."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def p_fmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


sb = pd.read_csv(f"{PROCESSED}/team_match_statsbomb.csv", parse_dates=["match_date"])
fb = pd.read_csv(f"{PROCESSED}/team_match_fbref.csv", parse_dates=["date"])
ts = pd.read_csv(f"{PROCESSED}/team_season_integrated.csv")
print(f"Loaded: {len(sb)} StatsBomb team-matches, {len(fb)} FBref team-matches, {len(ts)} team-seasons")

# Opponent's side of the same StatsBomb match, so every row knows what it allowed
opp_cols = ["shots", "shots_on_target", "xg", "pass_pct"]
opp = sb[["match_id", "team"] + opp_cols].rename(
    columns={"team": "opponent", **{c: f"{c}_against" for c in opp_cols}})
sb = sb.merge(opp, on=["match_id", "opponent"], how="left", validate="one_to_one")

# Performance stats leave out awarded results (no game was played)
fbp = fb[~fb.awarded.astype(bool)].copy()


# =============================================================================
# SECTION 1 - DESCRIPTIVE STATISTICS AND DISTRIBUTION SHAPE
# =============================================================================
print("\n[1] Descriptive statistics...")
desc_metrics = {"pass_pct": "Pass completion %", "possession_pct": "Possession %",
                "shots": "Shots", "shots_on_target": "Shots on target",
                "xg": "xG", "goals": "Goals"}
rows = []
for c, label in desc_metrics.items():
    s = sb[c].dropna()
    # Shapiro-Wilk is limited to 5,000 rows; 2,620 fits
    sw_w, sw_p = stats.shapiro(s)
    rows.append({"metric": label, "n": len(s), "mean": r2(s.mean()), "sd": r2(s.std()),
                 "median": r2(s.median()), "iqr": r2(s.quantile(.75) - s.quantile(.25)),
                 "min": r2(s.min()), "max": r2(s.max()), "skew": r2(s.skew()),
                 "kurtosis": r2(s.kurt()), "shapiro_p": float(sw_p)})
desc = pd.DataFrame(rows)
desc.to_csv(f"{PROCESSED}/eda_descriptives.csv", index=False)
R["descriptives"] = desc.to_dict("records")
print(desc[["metric", "mean", "sd", "median", "skew", "kurtosis"]].to_string(index=False))

# Goals are counts. Poisson assumes variance = mean; a dispersion index well
# above 1 means a count model would need a negative binomial instead.
g = fbp.goals
R["fb_goals_mean"] = r2(g.mean())
R["fb_goals_var"] = r2(g.var())
R["fb_goals_dispersion"] = r2(g.var() / g.mean())
R["fb_goals_skew"] = r2(g.skew())
R["fb_team_matches_played"] = int(len(fbp))
R["sb_goals_dispersion"] = r2(sb.goals.var() / sb.goals.mean())

# Home advantage (FBref, all seasons): chi-square of result by venue
ct = pd.crosstab(fbp.venue, fbp.result)
chi2, chi_p, chi_df, _ = stats.chi2_contingency(ct)
R["home_win_pct"] = r2(100 * (fbp[fbp.venue == "Home"].result == "W").mean())
R["away_win_pct"] = r2(100 * (fbp[fbp.venue == "Away"].result == "W").mean())
R["home_chi2"], R["home_chi2_df"], R["home_chi2_p"] = r2(chi2), int(chi_df), float(chi_p)

# Do the leagues differ? Kruskal-Wallis (rank-based, since goals are skewed)
kw = stats.kruskal(*[fbp[fbp.league == l].goals for l in LEAGUES])
R["kw_goals_league_H"], R["kw_goals_league_p"] = r2(kw.statistic), float(kw.pvalue)
R["goals_pg_by_league"] = {l: r2(fbp[fbp.league == l].goals.mean()) for l in LEAGUES}

# Figure 1: distributions (skew is what matters for model choice)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
vc = fbp.goals.value_counts().sort_index()
vc = vc[vc.index <= 8]
pois = stats.poisson.pmf(vc.index, g.mean()) * len(fbp)
ax.bar(vc.index, vc.values, color="#7B98AF", width=0.75)
ax.plot(vc.index, pois, "o-", color=INK, markersize=8, label="Poisson fit")
ax.set_xlabel("Goals"); ax.set_ylabel("Team-matches")
ax.legend(frameon=False)
ax = axes[1]
ax.hist(sb.xg, bins=40, color="#7B98AF", edgecolor=SURFACE, linewidth=1)
ax.set_xlabel("Expected goals"); ax.set_ylabel("Team-matches")
ax = axes[2]
ax.hist(sb.pass_pct, bins=40, color="#7B98AF", edgecolor=SURFACE, linewidth=1)
ax.set_xlabel("Pass completion %"); ax.set_ylabel("Team-matches")
fig.tight_layout()
fig.savefig("figures/fig1_distributions.png", dpi=150)
plt.close(fig)


# =============================================================================
# SECTION 2 - CORRELATION AND MULTICOLLINEARITY
# =============================================================================
print("\n[2] Correlations...")
corr_cols = {"pass_pct": "Pass %", "possession_pct": "Possession %", "shots": "Shots",
             "shots_on_target": "Shots on target", "xg": "xG", "goals": "Goals",
             "xg_against": "xG against", "win": "Win (1/0)"}
cm = sb[list(corr_cols)].corr(method="spearman").rename(index=corr_cols, columns=corr_cols)
cm.round(2).to_csv(f"{PROCESSED}/eda_spearman.csv")
R["rho_pass_possession"] = r2(cm.loc["Pass %", "Possession %"])
R["rho_shots_sot"] = r2(cm.loc["Shots", "Shots on target"])
R["rho_xg_sot"] = r2(cm.loc["xG", "Shots on target"])
R["rho_xg_goals"] = r2(cm.loc["xG", "Goals"])
R["rho_win_xg"] = r2(cm.loc["Win (1/0)", "xG"])
R["rho_win_possession"] = r2(cm.loc["Win (1/0)", "Possession %"])
R["rho_win_pass"] = r2(cm.loc["Win (1/0)", "Pass %"])
R["rho_win_xg_against"] = r2(cm.loc["Win (1/0)", "xG against"])

# VIF on the candidate predictors (goals left out: they define the result)
vif_cols = ["pass_pct", "possession_pct", "shots", "shots_on_target", "xg"]
X = sm.add_constant(sb[vif_cols].dropna())
R["vif"] = {c: r2(variance_inflation_factor(X.values, i)) for i, c in enumerate(X.columns) if c != "const"}
print("    VIF:", R["vif"])

# Figure 2: heatmap, diverging blue-gray-orange, numbers in every cell
fig, ax = plt.subplots(figsize=(9.5, 8))
from matplotlib.colors import LinearSegmentedColormap
div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#F0EFEA", "#d95926"])
im = ax.imshow(cm.values, cmap=div, vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=40, ha="right")
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index)
ax.grid(False)
for i in range(len(cm)):
    for j in range(len(cm)):
        v = cm.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                color="#FFFFFF" if abs(v) > 0.6 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Spearman ρ")
fig.tight_layout()
fig.savefig("figures/fig2_spearman_heatmap.png", dpi=150)
plt.close(fig)


# =============================================================================
# SECTION 3 - STAGE 1: MANN-KENDALL TREND TESTS (alpha = 0.05)
# =============================================================================
# One value per league-season, so each test sees a short series. Two ways in:
#   - per league: original Mann-Kendall (Hamed-Rao variance not needed at
#     n <= 12; autocorrelation can't be estimated that short)
#   - pooled: Regional Kendall test (Helsel & Frans 2006) - add up each
#     league's S statistic and variance, so leagues are compared only with
#     themselves and differences in league level can't fake a trend.
print("\n[3] Stage 1: Mann-Kendall trend tests...")


def mk_s_var(x):
    """Mann-Kendall S and its variance (with the tie correction)."""
    x = np.asarray(x, float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in counts)) / 18
    return s, var


def regional_kendall(series_by_league, min_n=4):
    S = V = 0
    used = []
    for l, s in series_by_league.items():
        s = s.dropna()
        if len(s) >= min_n:
            si, vi = mk_s_var(s.values)
            S += si; V += vi; used.append(l)
    if V == 0:
        return None
    z = (S - np.sign(S)) / np.sqrt(V) if S != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"S": int(S), "z": r2(z), "p": float(p), "leagues": used}


# League-season series. FBref covers results from 2013 and xG from 2019+;
# StatsBomb covers pass %, possession and shots on target in fewer seasons.
fbp_season = (fbp.groupby(["league", "season_end_year"])
                  .agg(goals_pg=("goals", "mean"), xg_pg=("xg", "mean"),
                       draw_rate=("result", lambda s: (s == "D").mean()),
                       home_win_rate=("result", lambda s: np.nan),  # filled below
                       n=("goals", "size"))
                  .reset_index())
hw = (fbp[fbp.venue == "Home"].groupby(["league", "season_end_year"]).result
      .apply(lambda s: (s == "W").mean()).rename("home_win_rate"))
fbp_season = fbp_season.drop(columns="home_win_rate").merge(hw.reset_index(), on=["league", "season_end_year"])
att = (fb[(fb.venue == "Home") & fb.attendance.notna()]
       .groupby(["league", "season_end_year"]).attendance.agg(["mean", "size"]))
# same 70% coverage rule as preprocessing
home_n = fb[fb.venue == "Home"].groupby(["league", "season_end_year"]).size()
att = att[(att["size"] / home_n.reindex(att.index)) >= 0.70]["mean"].rename("avg_attendance")
fbp_season = fbp_season.merge(att.reset_index(), on=["league", "season_end_year"], how="left")
# xG only where the season has full xG coverage
xg_cov = fbp.groupby(["league", "season_end_year"]).xg.apply(lambda s: s.notna().mean())
fbp_season.loc[fbp_season.set_index(["league", "season_end_year"]).index.map(xg_cov).values < 0.90, "xg_pg"] = np.nan
# competitive balance: SD of points per game across a league-season's teams
bal = ts.groupby(["league", "season_end_year"]).ppg.std().rename("ppg_sd")
fbp_season = fbp_season.merge(bal.reset_index(), on=["league", "season_end_year"], how="left")
# COVID-curtailed 2019/20 European seasons are kept but flagged; NWSL 2020 had no season
fbp_season.to_csv(f"{PROCESSED}/eda_league_season.csv", index=False)

sb_clubs = sb[sb.international == 0]
sb_season = (sb_clubs.groupby(["league", "season_end_year"])
                 .agg(pass_pct=("pass_pct", "mean"), possession_pct=("possession_pct", "mean"),
                      sot_pg=("shots_on_target", "mean"), shots_pg=("shots", "mean"),
                      sb_xg_pg=("xg", "mean"), n=("match_id", "nunique"))
                 .reset_index())
sb_season.to_csv(f"{PROCESSED}/eda_sb_league_season.csv", index=False)
R["sb_club_league_seasons"] = int(len(sb_season))
R["sb_league_season_counts"] = sb_season.groupby("league").size().to_dict()

stage1_metrics = {
    "goals_pg": ("Goals per team-match", fbp_season, "FBref"),
    "xg_pg": ("xG per team-match", fbp_season, "FBref"),
    "draw_rate": ("Draw rate", fbp_season, "FBref"),
    "home_win_rate": ("Home win rate", fbp_season, "FBref"),
    "avg_attendance": ("Avg home attendance", fbp_season, "FBref"),
    "ppg_sd": ("Competitive balance (SD of PPG)", fbp_season, "FBref"),
    "pass_pct": ("Pass completion %", sb_season, "StatsBomb"),
    "sot_pg": ("Shots on target per team-match", sb_season, "StatsBomb"),
}
s1_rows = []
for col, (label, df, src) in stage1_metrics.items():
    by_league = {l: df[df.league == l].sort_values("season_end_year").set_index("season_end_year")[col]
                 for l in LEAGUES}
    reg = regional_kendall(by_league)
    per = {}
    for l, s in by_league.items():
        s = s.dropna()
        if len(s) >= 4:
            res = mk.original_test(s.values)
            per[l] = {"n": len(s), "tau": r2(res.Tau), "p": float(res.p),
                      "sen_slope": float(res.slope), "trend": res.trend}
    years = df.loc[df[col].notna(), "season_end_year"]
    s1_rows.append({
        "metric": label, "col": col, "source": src,
        "seasons": f"{int(years.min())}-{int(years.max())}" if len(years) else "",
        "leagues_tested": len(reg["leagues"]) if reg else 0,
        "S": reg["S"] if reg else None, "z": reg["z"] if reg else None,
        "p": reg["p"] if reg else None,
        "passes_stage1": bool(reg and reg["p"] < ALPHA),
        "per_league": per,
    })
s1 = pd.DataFrame(s1_rows)
R["stage1"] = s1.drop(columns="per_league").to_dict("records")
R["stage1_per_league"] = {r["col"]: r["per_league"] for r in s1_rows}
for r in s1_rows:
    print(f"    {r['metric']:<34} leagues={r['leagues_tested']} z={r['z']} p={r['p'] if r['p'] is None else round(r['p'], 4)} "
          f"{'TREND' if r['passes_stage1'] else '-'}")

# Possession % averages 50 in every league-season by construction (it's a
# share between two teams), so it can't trend at league level - noted, not tested.
R["possession_league_mean_range"] = [r2(sb_season.possession_pct.min()), r2(sb_season.possession_pct.max())]

# Figure 3: goals per team-match by league-season (NWSL 2020 was cancelled,
# so its line breaks there instead of drawing through the gap)
LABEL_NUDGE = {"WSL": 10, "Serie A Women": 0, "Frauen Bundesliga": -12, "Liga F": 0, "NWSL": 0}
fig, ax = plt.subplots(figsize=(13, 5.6))
for l in LEAGUES:
    d = (fbp_season[fbp_season.league == l].set_index("season_end_year")
         .reindex(range(fbp_season[fbp_season.league == l].season_end_year.min(),
                        fbp_season[fbp_season.league == l].season_end_year.max() + 1)))
    ax.plot(d.index, d.goals_pg, "-o", color=LEAGUE_COLOR[l], markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    last = d.goals_pg.dropna()
    ax.annotate(l, (last.index[-1], last.iloc[-1]), xytext=(10, LABEL_NUDGE[l]),
                textcoords="offset points", va="center", fontsize=13, color=INK)
ax.set_xlim(2012.5, 2027.6)
ax.set_xticks(range(2013, 2026))
ax.set_ylabel("Goals per team-match")
ax.set_xlabel("Season (end year)")
fig.tight_layout()
fig.savefig("figures/fig3_goals_trend.png", dpi=150)
plt.close(fig)

# Figure 3b: attendance (log scale, it spans an order of magnitude)
fig, ax = plt.subplots(figsize=(13, 5.6))
for l in LEAGUES:
    d = fbp_season[(fbp_season.league == l) & fbp_season.avg_attendance.notna()].sort_values("season_end_year")
    if len(d) == 0:
        continue
    ax.plot(d.season_end_year, d.avg_attendance, "-o", color=LEAGUE_COLOR[l], markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label=l)
    ax.annotate(l, (d.season_end_year.iloc[-1], d.avg_attendance.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=13, color=INK)
ax.set_yscale("log")
ax.set_yticks([500, 1000, 2000, 5000, 10000])
ax.set_yticklabels(["500", "1,000", "2,000", "5,000", "10,000"])
ax.minorticks_off()
ax.set_xlim(2012.5, 2027.3)
ax.set_xticks(range(2013, 2026))
ax.set_ylabel("Avg home attendance (log scale)")
ax.set_xlabel("Season (end year)")
fig.tight_layout()
fig.savefig("figures/fig3b_attendance_trend.png", dpi=150)
plt.close(fig)


# =============================================================================
# SECTION 4 - STAGE 2: LOGISTIC REGRESSION SCREEN (Wald, Holm alpha = 0.05)
# =============================================================================
# win (1/0) ~ z(metric) + home, one metric at a time, on the training seasons
# (<= 2022) plus validation (2023); the 2024-25 test seasons stay untouched.
# The two rows of a match are mirror images, so I cluster standard errors by
# match. Goals are left out: a win is defined by goals, so it's not a screen.
print("\n[4] Stage 2: logistic regression screen...")
dev = sb[sb.split.isin(["train", "validation"])].copy()
fbdev = fbp[fbp.split.isin(["train", "validation"]) & fbp.xg.notna()].copy()
fbdev["win"] = (fbdev.result == "W").astype(int)
fbdev["is_home"] = (fbdev.venue == "Home").astype(int)
fbdev["xg_diff"] = fbdev.xg - fbdev.xg_against
fbdev["match_key"] = fbdev.match_url
dev["xg_diff"] = dev.xg - dev.xg_against
dev["match_key"] = dev.match_id

screen = [
    ("pass_pct", "Pass completion %", dev),
    ("possession_pct", "Possession %", dev),
    ("shots", "Shots", dev),
    ("shots_on_target", "Shots on target", dev),
    ("xg", "xG (StatsBomb)", dev),
    ("xg", "xG (FBref)", fbdev),
]
s2_rows = []
for col, label, df in screen:
    d = df[[col, "win", "is_home", "match_key"]].dropna()
    z = (d[col] - d[col].mean()) / d[col].std()
    Xs = sm.add_constant(pd.DataFrame({"z": z, "is_home": d.is_home}))
    fit = sm.Logit(d.win, Xs).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.match_key})
    b, se, p = fit.params["z"], fit.bse["z"], fit.pvalues["z"]
    s2_rows.append({"metric": label, "n": len(d), "beta_per_sd": r2(b), "odds_ratio": r2(np.exp(b)),
                    "or_lo": r2(np.exp(b - 1.96 * se)), "or_hi": r2(np.exp(b + 1.96 * se)),
                    "wald_z": r2(b / se), "p_raw": float(p),
                    "auc": r2(roc_auc_score(d.win, fit.predict(Xs)))})
s2 = pd.DataFrame(s2_rows)
s2["p_holm"] = multipletests(s2.p_raw, alpha=ALPHA, method="holm")[1]
s2["passes_stage2"] = s2.p_holm < ALPHA
R["stage2"] = s2.to_dict("records")
R["stage2_n_sb"] = int(len(dev))
R["stage2_n_fb"] = int(len(fbdev))
print(s2[["metric", "n", "odds_ratio", "or_lo", "or_hi", "p_holm", "auc"]].to_string(index=False))

# =============================================================================
# SECTION 5 - FEATURE ENGINEERING
# =============================================================================
print("\n[5] Feature engineering...")
# 5a. Match-level ratios and opponent differentials (StatsBomb)
sb["shot_accuracy"] = np.where(sb.shots > 0, sb.shots_on_target / sb.shots, np.nan)
sb["xg_per_shot"] = np.where(sb.shots > 0, sb.xg / sb.shots, np.nan)
sb["xg_diff"] = sb.xg - sb.xg_against
sb["sot_diff"] = sb.shots_on_target - sb.shots_on_target_against
sb["pass_pct_diff"] = sb.pass_pct - sb.pass_pct_against
sb["finishing_vs_xg"] = sb.goals - sb.xg
R["shot_accuracy_mean"] = r2(100 * sb.shot_accuracy.mean())
R["xg_per_shot_mean"] = r2(sb.xg_per_shot.mean())
R["zero_shot_rows"] = int((sb.shots == 0).sum())

# Same Stage 2 test on the engineered differentials, so they can be compared
dev = sb[sb.split.isin(["train", "validation"])].copy()
dev["match_key"] = dev.match_id
fe_rows = []
for col, label in [("xg_diff", "xG differential"), ("sot_diff", "Shots-on-target differential"),
                   ("pass_pct_diff", "Pass % differential"), ("shot_accuracy", "Shot accuracy"),
                   ("xg_per_shot", "xG per shot")]:
    d = dev[[col, "win", "is_home", "match_key"]].dropna()
    z = (d[col] - d[col].mean()) / d[col].std()
    Xs = sm.add_constant(pd.DataFrame({"z": z, "is_home": d.is_home}))
    fit = sm.Logit(d.win, Xs).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.match_key})
    b_, se_ = fit.params["z"], fit.bse["z"]
    fe_rows.append({"feature": label, "n": len(d), "odds_ratio": r2(np.exp(b_)),
                    "or_lo": r2(np.exp(b_ - 1.96 * se_)), "or_hi": r2(np.exp(b_ + 1.96 * se_)),
                    "p_raw": float(fit.pvalues["z"]),
                    "auc": r2(roc_auc_score(d.win, fit.predict(Xs)))})
fe = pd.DataFrame(fe_rows)
fe["p_holm"] = multipletests(fe.p_raw, method="holm")[1]
R["engineered_screen"] = fe.to_dict("records")
print(fe.to_string(index=False))

# Figure 4: odds ratio per 1 SD with 95% CI - raw metrics and the two
# strongest engineered differentials side by side
plot_rows = [(r["metric"], r["odds_ratio"], r["or_lo"], r["or_hi"], "raw") for r in s2_rows]
for r in fe_rows[:2]:
    plot_rows.append((r["feature"], r["odds_ratio"], r["or_lo"], r["or_hi"], "engineered"))
plot_rows.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(12, 6))
for yi, (name, orr, lo, hi, kind) in enumerate(plot_rows):
    col = ACCENT if kind == "engineered" else "#7B98AF"
    ax.plot([lo, hi], [yi, yi], color=col, linewidth=3, solid_capstyle="round")
    ax.plot(orr, yi, "o", color=col, markersize=11, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{orr:.2f}", (hi, yi), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=13, color=INK)
ax.axvline(1, color=INK, linewidth=1.2, linestyle="--")
ax.set_yticks(range(len(plot_rows))); ax.set_yticklabels([t[0] for t in plot_rows])
ax.set_xlim(0.8, 9.5)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xlabel("Odds ratio of a win per +1 SD, 95% CI (SEs clustered by match)")
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color="#7B98AF", marker="o", lw=3, ms=10, label="Raw metric"),
                   Line2D([], [], color=ACCENT, marker="o", lw=3, ms=10, label="Engineered differential")],
          frameon=False, loc="lower right", fontsize=13)
fig.tight_layout()
fig.savefig("figures/fig4_stage2_odds_ratios.png", dpi=150)
plt.close(fig)

# 5b. Team-season features (FBref backbone)
ts["gd_pg"] = ts.goals_pg - ts.goals_against_pg
R["rho_ppg_gdpg"] = r2(stats.spearmanr(ts.gd_pg, ts.ppg, nan_policy="omit")[0])
xgts = ts.dropna(subset=["xg_pg"])
R["n_team_seasons_with_xg"] = int(len(xgts))

# 5c. League-month series features for the forecasting models (Presentation 4)
fbp["year_month"] = fbp.date.dt.to_period("M")
lm = (fbp.groupby(["league", "season_end_year", "year_month"])
          .agg(goals_pg=("goals", "mean"), xg_pg=("xg", "mean"), fixtures=("goals", "size"))
          .reset_index())
lm["fixtures"] = lm.fixtures // 2
lm = lm[lm.fixtures >= 3].sort_values(["league", "year_month"]).reset_index(drop=True)
lm["month_of_season"] = lm.groupby(["league", "season_end_year"]).cumcount() + 1
g_ = lm.groupby("league").goals_pg
lm["lag1"] = g_.shift(1)
lm["lag2"] = g_.shift(2)
lm["roll3"] = g_.transform(lambda s: s.shift(1).rolling(3).mean())
prev_season = (lm.groupby(["league", "season_end_year"]).goals_pg.mean()
                 .groupby(level=0).shift(1).rename("prev_season_mean").reset_index())
lm = lm.merge(prev_season, on=["league", "season_end_year"], how="left")
lm["split"] = np.where(lm.season_end_year <= 2022, "train",
                       np.where(lm.season_end_year == 2023, "validation", "test"))
lm.to_csv(f"{PROCESSED}/eda_league_month_features.csv", index=False)
R["league_month_rows"] = int(len(lm))
R["league_month_rows_by_league"] = lm.groupby("league").size().to_dict()
R["rho_goals_lag1"] = r2(lm[["goals_pg", "lag1"]].dropna().corr(method="spearman").iloc[0, 1])
R["rho_goals_roll3"] = r2(lm[["goals_pg", "roll3"]].dropna().corr(method="spearman").iloc[0, 1])
R["rho_goals_prev_season"] = r2(lm[["goals_pg", "prev_season_mean"]].dropna().corr(method="spearman").iloc[0, 1])


# =============================================================================
# SECTION 6 - STATIONARITY OF THE LEAGUE-MONTH SERIES (for ARIMA)
# =============================================================================
# ADF: H0 = unit root (non-stationary). KPSS: H0 = stationary. Using both
# guards against the low power of either one on short series.
print("\n[6] Stationarity tests...")
stat_rows = []
for l in LEAGUES:
    s = lm[lm.league == l].goals_pg.values
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    stat_rows.append({"league": l, "n_months": len(s), "adf_p": float(adf_p), "kpss_p": float(kpss_p),
                      "stationary_both": bool(adf_p < ALPHA and kpss_p >= ALPHA)})
R["stationarity"] = stat_rows

# Sensitivity check on Stage 1 with more data points: league-month means
# regressed on time (years) with league and month-of-season fixed effects,
# Newey-West (HAC, 3 lags) standard errors for autocorrelation.
import statsmodels.formula.api as smf
lm_t = lm.copy()
lm_t["t_years"] = (lm_t.year_month.dt.to_timestamp() - pd.Timestamp("2013-01-01")).dt.days / 365.25
for y in ["goals_pg", "xg_pg"]:
    d = lm_t.dropna(subset=[y])
    m = smf.ols(f"{y} ~ t_years + C(league) + C(month_of_season)", d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3})
    R[f"sens_{y}_slope_per_year"] = r2(m.params["t_years"]) if abs(m.params["t_years"]) >= 0.005 else float(m.params["t_years"])
    R[f"sens_{y}_p"] = float(m.pvalues["t_years"])
    R[f"sens_{y}_n"] = int(len(d))
    print(f"    sensitivity {y}: n={len(d)} slope/yr={m.params['t_years']:.4f} p={m.pvalues['t_years']:.3f}")
for r in stat_rows:
    print(f"    {r['league']:<18} n={r['n_months']:>3} ADF p={r['adf_p']:.3f} KPSS p={r['kpss_p']:.3f}")


# =============================================================================
# SECTION 7 - PCA (dimensionality reduction)
# =============================================================================
# Standardized match-level features, scaler and PCA fit on training seasons
# only. I compare a logistic model on the raw features against one on the
# leading components, scored on the 2023 validation season.
print("\n[7] PCA...")
pca_cols = {"pass_pct": "Pass %", "possession_pct": "Possession %", "shots": "Shots",
            "shots_on_target": "Shots on target", "xg": "xG", "xg_per_shot": "xG per shot",
            "shots_against": "Shots against", "shots_on_target_against": "SoT against",
            "xg_against": "xG against", "pass_pct_against": "Pass % against"}
P = sb.dropna(subset=list(pca_cols)).copy()
tr, va = P[P.split == "train"], P[P.split == "validation"]
scaler = StandardScaler().fit(tr[list(pca_cols)])
pca = PCA().fit(scaler.transform(tr[list(pca_cols)]))
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
eig = pca.explained_variance_
R["pca_n_train"] = int(len(tr))
R["pca_n_validation"] = int(len(va))
R["pca_evr"] = [r2(100 * v) for v in evr]
R["pca_cum"] = [r2(100 * v) for v in cum]
R["pca_kaiser_k"] = int((eig > 1).sum())
R["pca_k_80"] = int(np.argmax(cum >= 0.80) + 1)
load = pd.DataFrame(pca.components_[:3].T, index=list(pca_cols.values()), columns=["PC1", "PC2", "PC3"])
load.round(2).to_csv(f"{PROCESSED}/eda_pca_loadings.csv")
R["pca_loadings"] = load.round(2).to_dict()

k = R["pca_kaiser_k"]
Xtr_raw, Xva_raw = scaler.transform(tr[list(pca_cols)]), scaler.transform(va[list(pca_cols)])
Xtr_pc, Xva_pc = pca.transform(Xtr_raw)[:, :k], pca.transform(Xva_raw)[:, :k]
m_raw = LogisticRegression(max_iter=1000).fit(Xtr_raw, tr.win)
m_pc = LogisticRegression(max_iter=1000).fit(Xtr_pc, tr.win)
R["auc_val_raw"] = r2(roc_auc_score(va.win, m_raw.predict_proba(Xva_raw)[:, 1]))
R["auc_val_pca"] = r2(roc_auc_score(va.win, m_pc.predict_proba(Xva_pc)[:, 1]))
print(f"    Kaiser k = {k}; first {k} PCs explain {R['pca_cum'][k-1]}%; "
      f"validation AUC raw {R['auc_val_raw']} vs PCA {R['auc_val_pca']}")

# Figure 5: scree + PC1/PC2 loadings
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
xs = np.arange(1, len(evr) + 1)
ax.bar(xs, 100 * evr, color="#7B98AF", width=0.7)
ax.plot(xs, 100 * cum, "-o", color=INK, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
ax.axvline(k + 0.5, color=ACCENT, linestyle="--", linewidth=1.5)
ax.annotate(f"eigenvalue > 1: {k} PCs\n{R['pca_cum'][k-1]}% of variance", (k + 0.6, 55),
            fontsize=13, color=ACCENT)
ax.set_xticks(xs)
ax.set_xlabel("Principal component"); ax.set_ylabel("% of variance")
ax = axes[1]
names = list(load.index)[::-1]
yy = np.arange(len(names))
ax.barh(yy + 0.2, load.loc[names, "PC1"], height=0.38, color=INK, label=f"PC1 ({R['pca_evr'][0]}%)")
ax.barh(yy - 0.2, load.loc[names, "PC2"], height=0.38, color="#7B98AF", label=f"PC2 ({R['pca_evr'][1]}%)")
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_yticks(yy); ax.set_yticklabels(names)
ax.set_xlabel("Loading")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper right", fontsize=13)
fig.tight_layout()
fig.savefig("figures/fig5_pca.png", dpi=150)
plt.close(fig)


# =============================================================================
# SECTION 8 - SAVE AND VERIFY
# =============================================================================
with open(f"{PROCESSED}/eda_results.json", "w") as f:
    json.dump(R, f, indent=2, default=str)

# Write the engineered tables back to the database for Presentation 4
con = sqlite3.connect("data/capstone.db")
sb.to_sql("sb_team_match_features", con, if_exists="replace", index=False)
lm.assign(year_month=lm.year_month.astype(str)).to_sql("league_month_features", con, if_exists="replace", index=False)
fbp_season.to_sql("league_season_eda", con, if_exists="replace", index=False)
con.close()

# Recompute the headline numbers independently and compare
checks = {
    "StatsBomb team-matches = 2,620": len(sb) == 2620,
    "Spearman matrix is symmetric": np.allclose(cm.values, cm.values.T),
    "Holm p >= raw p for every metric": bool((s2.p_holm >= s2.p_raw - 1e-12).all()),
    "PCA variance shares sum to 100%": abs(evr.sum() - 1) < 1e-9,
    "Scaler/PCA fit on training seasons only": set(tr.split) == {"train"},
    "No test seasons in the Stage 2 screen": not dev.split.eq("test").any(),
    "League-month features use only past months": bool(
        (lm.groupby("league").apply(
            lambda d: (d.lag1.values[1:] == d.goals_pg.values[:-1]).all())).all()),
}
print("\n[8] Checks")
for name, ok in checks.items():
    print(f"    {'PASS' if ok else 'FAIL'}  {name}")
print(f"{sum(checks.values())}/{len(checks)} checks passed. Figures in figures/, numbers in {PROCESSED}/eda_results.json")
