# The Rise of Women's Soccer: A Data-Driven Case Study of Growth Trends and Performance Analytics

**DS 785 Capstone, University of Wisconsin MSDS · Aixa Helbringer-Escobar · Fall 2026**

This repository holds the code for my capstone project. It collects women's soccer data from StatsBomb and FBref, cleans and links the two sources, and builds the datasets I use to test which performance metrics are changing over time and tied to winning, and then to forecast them.

## How the code is organized

| File | Project stage | What it does |
|---|---|---|
| [`capstone_pipeline.py`](capstone_pipeline.py) | Data collection, cleaning, preprocessing (Presentation 2) | One script with eight titled sections: extract StatsBomb → build team-match metrics → clean FBref → link the sources → preprocess → load SQLite → verify every figure in the presentation → run the SQL layer |
| [`capstone_sql_pipeline.sql`](capstone_sql_pipeline.sql) | Integration and modeling datasets | Data-quality checks, success variables (points per game, win rate), the StatsBomb–FBref join, modeling views for Presentations 3 and 4, and a SQL-vs-Python reconciliation |
| [`next_steps/05_fbref_team_stats.R`](next_steps/05_fbref_team_stats.R) | Next stage (not yet run) | Pulls FBref team passing, possession and shooting tables, 2019 onward, with worldfootballR |

### Sections of `capstone_pipeline.py`

| Section | Purpose |
|---|---|
| 0 – Setup | Libraries, folders, download helper with retry and a cache that survives interruptions |
| 1 – Extract | Every women's competition-season in StatsBomb Open Data: 1,310 matches, 4,504,875 events |
| 2 – Transform | Events → one row per team per match (pass %, possession %, shots on target, goals, xG); goals validated against official scores |
| 3 – Extract + clean FBref | Five top-flight leagues, 2013–2025: scope rules, franchise key, points per game and win rate |
| 4 – Integrate | Links StatsBomb and FBref teams by matching fixtures (66/66 teams, 99.56% of team-matches) |
| 5 – Preprocess | Missing data (no imputation), 1.5 × IQR outlier flags, log and z-score scaling, encoding, season-based train/validation/test split |
| 6 – Load | Writes 6 tables to `data/capstone.db` |
| 7 – Verify | Recomputes all 42 figures quoted in Presentation 2 and prints PASS/FAIL |
| 8 – SQL pipeline | Runs `capstone_sql_pipeline.sql` and prints its checks |

## Data sources

- **StatsBomb Open Data**: https://github.com/statsbomb/open-data. Free for non-commercial research; data provided by StatsBomb.
- **FBref** (Sports Reference), through the pre-scraped pages in https://github.com/JaseZiv/worldfootballR_data. Used instead of scraping fbref.com directly.

Raw and processed data are not stored in this repository; the script downloads everything it needs.

## How to run

```bash
pip install -r requirements.txt
python capstone_pipeline.py
```

On Windows, `py` works in place of `python`. The first run downloads about 300 MB and takes 10–15 minutes; downloads are cached, so a stopped run resumes where it left off. Keep both files in the same folder.

A successful run ends with `42/42 figures reproduced from the data`, and every data-quality and reconciliation check in Section 8 shows `OK`.

## Outputs

- `data/capstone.db`: SQLite database with tables `sb_matches`, `sb_team_match`, `fb_team_match`, `team_season`, `team_crosswalk`, `fb_season_completeness`, and views `v_integrated_team_season`, `v_league_season_trends`, `v_league_month_series`, `v_logistic_screen`
- `data/processed/`: the modeling tables as CSV, plus `summary.json`

Read it from R:

```r
library(DBI)
con <- dbConnect(RSQLite::SQLite(), "data/capstone.db")
trends <- dbGetQuery(con, "SELECT * FROM v_league_season_trends")
```
