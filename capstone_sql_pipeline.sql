/* =============================================================================
   DS 785 Capstone - The Rise of Women's Soccer
   SQL pipeline: integration, success variables and modeling datasets
   Author: Aixa Helbringer-Escobar
   =============================================================================

   capstone_pipeline.py loads the cleaned source tables into data/capstone.db.
   This file does the relational work on top of them in SQL:
     1. Data-quality checks on the loaded tables
     2. Success variables (points per game, win rate) rebuilt in SQL
     3. Integration: StatsBomb metrics joined to FBref through the crosswalk
     4. Modeling datasets for Presentations 3 and 4
     5. A reconciliation check that SQL and Python agree

   capstone_pipeline.py runs this file automatically at the end (Section 8).
   I can also run it by hand: open capstone.db in DB Browser for SQLite,
   go to Execute SQL, open this file and run all.
   ============================================================================= */


/* -----------------------------------------------------------------------------
   SECTION 1 - DATA-QUALITY CHECKS
   Each query should return 0 problem rows. The Python script prints them.
   ----------------------------------------------------------------------------- */

DROP VIEW IF EXISTS dq_checks;
CREATE VIEW dq_checks AS
-- every FBref fixture should appear exactly twice (home row + away row)
SELECT 'FBref fixtures without exactly 2 rows' AS check_name,
       COUNT(*) AS problem_rows
FROM (SELECT match_url FROM fb_team_match GROUP BY match_url HAVING COUNT(*) <> 2)
UNION ALL
-- every StatsBomb match should have exactly 2 team rows
SELECT 'StatsBomb matches without exactly 2 rows',
       COUNT(*)
FROM (SELECT match_id FROM sb_team_match GROUP BY match_id HAVING COUNT(*) <> 2)
UNION ALL
-- no duplicate team-match rows
SELECT 'Duplicate StatsBomb team-match rows',
       COUNT(*)
FROM (SELECT match_id, team FROM sb_team_match GROUP BY match_id, team HAVING COUNT(*) > 1)
UNION ALL
-- a team's goals must equal its opponent's goals against (both sources)
SELECT 'FBref goals that do not mirror the opponent row',
       COUNT(*)
FROM fb_team_match a
JOIN fb_team_match b ON a.match_url = b.match_url AND a.team = b.opponent
WHERE a.goals <> b.goals_against
UNION ALL
-- percentages must sit between 0 and 100
SELECT 'StatsBomb pass % or possession % out of range',
       COUNT(*)
FROM sb_team_match
WHERE pass_pct NOT BETWEEN 0 AND 100 OR possession_pct NOT BETWEEN 0 AND 100
UNION ALL
-- possession of the two teams in a match must add to 100
SELECT 'StatsBomb possession not summing to 100',
       COUNT(*)
FROM (SELECT match_id FROM sb_team_match GROUP BY match_id
      HAVING ABS(SUM(possession_pct) - 100) > 0.01)
UNION ALL
-- every StatsBomb club team must have an FBref match in the crosswalk
SELECT 'StatsBomb club teams missing from crosswalk',
       COUNT(*)
FROM (SELECT DISTINCT s.league, s.team
      FROM sb_team_match s
      LEFT JOIN team_crosswalk x ON x.league = s.league AND x.team_sb = s.team
      WHERE s.international = 0 AND x.team_fb IS NULL)
UNION ALL
-- the crosswalk has to be one-to-one in both directions
SELECT 'Crosswalk FBref names claimed twice',
       COUNT(*)
FROM (SELECT league, team_fb FROM team_crosswalk GROUP BY league, team_fb HAVING COUNT(*) > 1);


/* -----------------------------------------------------------------------------
   SECTION 2 - SUCCESS VARIABLES (team-season)
   points per game = league points / matches played
   win rate        = wins / matches played
   I use points per game rather than points per season because season
   lengths differ between leagues and years.
   ----------------------------------------------------------------------------- */

DROP VIEW IF EXISTS v_team_season_success;
CREATE VIEW v_team_season_success AS
SELECT league,
       season_end_year,
       team,
       COUNT(*)                                        AS matches,
       SUM(result = 'W')                               AS wins,
       SUM(result = 'D')                               AS draws,
       SUM(result = 'L')                               AS losses,
       SUM(points)                                     AS points,
       ROUND(1.0 * SUM(points) / COUNT(*), 4)          AS ppg,
       ROUND(1.0 * SUM(result = 'W') / COUNT(*), 4)    AS win_rate,
       -- performance stats leave out awarded results
       AVG(CASE WHEN awarded = 0 THEN goals END)          AS goals_pg,
       AVG(CASE WHEN awarded = 0 THEN goals_against END)  AS goals_against_pg,
       AVG(CASE WHEN awarded = 0 THEN xg END)             AS xg_pg,
       MAX(covid_curtailed)                            AS covid_curtailed,
       MAX(split)                                      AS split
FROM fb_team_match
GROUP BY league, season_end_year, team;


/* -----------------------------------------------------------------------------
   SECTION 3 - INTEGRATION
   StatsBomb team-match metrics are renamed to FBref team names through the
   crosswalk, averaged to team-season, then LEFT JOINed onto the FBref
   success table. FBref is the backbone because it covers every season;
   StatsBomb only adds columns where it has data.
   ----------------------------------------------------------------------------- */

DROP VIEW IF EXISTS v_sb_team_season;
CREATE VIEW v_sb_team_season AS
SELECT s.league,
       s.season_end_year,
       x.team_fb                  AS team,
       COUNT(*)                   AS sb_matches,
       AVG(s.pass_pct)            AS sb_pass_pct,
       AVG(s.possession_pct)      AS sb_possession_pct,
       AVG(s.shots_on_target)     AS sb_sot_pg,
       AVG(s.shots)               AS sb_shots_pg,
       AVG(s.xg)                  AS sb_xg_pg,
       AVG(s.goals)               AS sb_goals_pg
FROM sb_team_match s
JOIN team_crosswalk x
  ON x.league = s.league AND x.team_sb = s.team
WHERE s.international = 0
GROUP BY s.league, s.season_end_year, x.team_fb;

DROP VIEW IF EXISTS v_integrated_team_season;
CREATE VIEW v_integrated_team_season AS
SELECT f.*,
       s.sb_matches, s.sb_pass_pct, s.sb_possession_pct,
       s.sb_sot_pg, s.sb_shots_pg, s.sb_xg_pg, s.sb_goals_pg,
       (s.sb_matches IS NOT NULL) AS has_statsbomb
FROM v_team_season_success f
LEFT JOIN v_sb_team_season s
  ON s.league = f.league
 AND s.season_end_year = f.season_end_year
 AND s.team = f.team;


/* -----------------------------------------------------------------------------
   SECTION 4 - MODELING DATASETS
   ----------------------------------------------------------------------------- */

-- 4a. League-season trend table -> Stage 1 trend tests (Presentation 3)
DROP VIEW IF EXISTS v_league_season_trends;
CREATE VIEW v_league_season_trends AS
SELECT league,
       season_end_year,
       COUNT(*) / 2                                   AS fixtures,
       COUNT(DISTINCT team)                           AS teams,
       ROUND(AVG(goals), 4)                           AS goals_per_team_game,
       ROUND(AVG(xg), 4)                              AS xg_per_team_game,
       ROUND(AVG(CASE WHEN venue = 'Home' THEN attendance END), 1) AS avg_attendance,
       ROUND(1.0 * SUM(result = 'D') / COUNT(*), 4)   AS draw_rate
FROM fb_team_match
GROUP BY league, season_end_year;

-- 4b. League-month series -> the finer time grain I proposed for ARIMA,
--     because no league has more than 11 seasons
DROP VIEW IF EXISTS v_league_month_series;
CREATE VIEW v_league_month_series AS
SELECT league,
       strftime('%Y-%m', date)      AS year_month,
       season_end_year,
       COUNT(*) / 2                 AS fixtures,
       ROUND(AVG(goals), 4)         AS goals_per_team_game,
       ROUND(AVG(xg), 4)            AS xg_per_team_game,
       MAX(split)                   AS split
FROM fb_team_match
GROUP BY league, strftime('%Y-%m', date)
HAVING COUNT(*) / 2 >= 3;            -- months with fewer than 3 games are too noisy

-- 4c. Match-level table -> Stage 2 logistic regression (win ~ metric)
DROP VIEW IF EXISTS v_logistic_screen;
CREATE VIEW v_logistic_screen AS
SELECT match_id, league, season_end_year, team, opponent, is_home, international,
       win,
       pass_pct, possession_pct, shots_on_target, xg, goals,
       z_pass_pct, z_possession_pct, z_shots_on_target, z_xg,
       split
FROM sb_team_match;


/* -----------------------------------------------------------------------------
   SECTION 5 - RECONCILIATION: SQL vs PYTHON
   The success variables and integration above were built independently of
   the Python version (team_season table). Every difference should be 0.
   ----------------------------------------------------------------------------- */

DROP VIEW IF EXISTS reconciliation;
CREATE VIEW reconciliation AS
SELECT 'Team-seasons in SQL vs Python' AS check_name,
       (SELECT COUNT(*) FROM v_integrated_team_season) - (SELECT COUNT(*) FROM team_season) AS difference
UNION ALL
SELECT 'Team-seasons where PPG differs',
       COUNT(*)
FROM v_integrated_team_season v
JOIN team_season t USING (league, season_end_year, team)
WHERE ABS(v.ppg - t.ppg) > 0.0001
UNION ALL
SELECT 'Team-seasons where win rate differs',
       COUNT(*)
FROM v_integrated_team_season v
JOIN team_season t USING (league, season_end_year, team)
WHERE ABS(v.win_rate - t.win_rate) > 0.0001
UNION ALL
SELECT 'Team-seasons where StatsBomb coverage differs',
       COUNT(*)
FROM v_integrated_team_season v
JOIN team_season t USING (league, season_end_year, team)
WHERE v.has_statsbomb <> t.has_statsbomb
UNION ALL
SELECT 'Team-seasons where StatsBomb pass % differs',
       COUNT(*)
FROM v_integrated_team_season v
JOIN team_season t USING (league, season_end_year, team)
WHERE ABS(v.sb_pass_pct - t.sb_pass_pct) > 0.0001;
