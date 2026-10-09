"""
===============================================================================
 DS 785 Capstone - The Rise of Women's Soccer
 Presentation 2: Data Collection, Cleaning and Preprocessing
 Author: Aixa Helbringer-Escobar
===============================================================================

What this script does, start to finish:
  1. Pulls every women's match from StatsBomb Open Data
  2. Turns the raw events into team-match performance metrics
  3. Pulls and cleans FBref results for five women's top-flight leagues
  4. Links StatsBomb and FBref teams by matching fixtures
  5. Applies my preprocessing rules (missing data, outliers, scaling,
     encoding, temporal split)
  6. Loads everything into a SQLite database (data/capstone.db)
  7. Re-checks every number I quote in Presentation 2
  8. Runs my SQL pipeline (capstone_sql_pipeline.sql): data-quality checks,
     integration and modeling views, and a SQL-vs-Python reconciliation

How to run it (Windows PowerShell):
  py -m pip install pandas requests pyarrow pyreadr scipy statsmodels
  py capstone_pipeline.py

The first run takes about 10-15 minutes because it downloads roughly 300 MB
of StatsBomb files. Everything gets cached, so if the connection drops I can
just run it again and it picks up where it stopped.

Data sources:
  - StatsBomb Open Data: https://github.com/statsbomb/open-data
    (free for non-commercial research; StatsBomb has to be credited)
  - FBref, through the worldfootballR_data mirror of pre-scraped pages:
    https://github.com/JaseZiv/worldfootballR_data
"""

# =============================================================================
# SECTION 0 - SETUP
# =============================================================================
import collections
import gzip
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyreadr
import requests

# I run everything relative to the folder this file lives in, so it works
# no matter where PowerShell is pointed when I start it.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RAW, INTERIM, PROCESSED = "data/raw", "data/interim", "data/processed"
for folder in [RAW, f"{RAW}/fbref", f"{INTERIM}/events", PROCESSED]:
    os.makedirs(folder, exist_ok=True)

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
FB_BASE = "https://github.com/JaseZiv/worldfootballR_data/releases/download/match_results/"
FB_COUNTRIES = ["ENG", "USA", "GER", "ESP", "ITA"]

session = requests.Session()
profile = {}   # StatsBomb profiling numbers I report on the slides
log = {}       # FBref cleaning and integration numbers
summary = {}   # preprocessing numbers


def get_statsbomb(path, retries=6):
    """Download one StatsBomb JSON file.
    My first run lost its connection halfway through, so this retries with
    a growing wait and caches each file as .json.gz. A rerun reads from the
    cache instead of downloading everything again."""
    cache = f"{RAW}/cache/{path}.gz"
    if os.path.exists(cache):
        try:
            with gzip.open(cache, "rt") as f:
                return json.load(f)
        except (EOFError, OSError, ValueError):
            # A run that got interrupted can leave a half-written file behind.
            # If a cached file is broken, I delete it and download it again.
            os.remove(cache)
    for attempt in range(retries):
        try:
            r = session.get(SB_BASE + path, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    # Write to a temporary file first and rename it only when it's complete,
    # so stopping the script mid-download can't leave a broken file.
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".part"
    with gzip.open(tmp, "wt") as f:
        json.dump(data, f)
    os.replace(tmp, cache)
    return data


def nested(d, *keys):
    """StatsBomb events are nested JSON (type -> name, pass -> outcome -> name).
    This walks down the keys and returns None if any level is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def iqr_flag(s):
    """Standard 1.5 x IQR outlier rule."""
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


# =============================================================================
# SECTION 1 - EXTRACT: STATSBOMB OPEN DATA
# =============================================================================
# This extends my DS 710 pipeline. Instead of one tournament, I pull every
# women's competition-season StatsBomb has published. Same filter as before:
# competition_gender == "female".

def extract_statsbomb():
    print("\n[1] Extracting StatsBomb women's data...")
    comps = pd.DataFrame(get_statsbomb("competitions.json"))
    women = comps[comps.competition_gender == "female"]

    match_rows = []
    for _, c in women.iterrows():
        for m in get_statsbomb(f"matches/{c.competition_id}/{c.season_id}.json"):
            match_rows.append({
                "match_id": m["match_id"], "competition_id": c.competition_id,
                "season_id": c.season_id, "competition": c.competition_name,
                "season": c.season_name, "international": c.competition_international,
                "match_date": m["match_date"], "kick_off": m.get("kick_off"),
                "stage": m.get("competition_stage", {}).get("name"),
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "home_score": m["home_score"], "away_score": m["away_score"],
                "stadium": (m.get("stadium") or {}).get("name"),
                "data_version": m.get("metadata", {}).get("data_version"),
            })
    matches = pd.DataFrame(match_rows)
    matches.to_csv(f"{INTERIM}/sb_matches.csv", index=False)
    print(f"    {len(matches)} matches across {len(women)} competition-seasons")

    # Flatten each event down to the fields I actually use
    def flatten(ev, match_id):
        loc = ev.get("location") or [None, None]
        return {
            "match_id": match_id, "event_id": ev["id"], "index": ev["index"],
            "period": ev["period"], "minute": ev["minute"], "second": ev["second"],
            "type": nested(ev, "type", "name"), "team": nested(ev, "team", "name"),
            "possession": ev.get("possession"),
            "possession_team": nested(ev, "possession_team", "name"),
            "play_pattern": nested(ev, "play_pattern", "name"),
            "player": nested(ev, "player", "name"),
            "x": loc[0], "y": loc[1], "duration": ev.get("duration"),
            "under_pressure": ev.get("under_pressure", False),
            "pass_outcome": nested(ev, "pass", "outcome", "name"),
            "pass_type": nested(ev, "pass", "type", "name"),
            "pass_length": nested(ev, "pass", "length"),
            "shot_outcome": nested(ev, "shot", "outcome", "name"),
            "shot_type": nested(ev, "shot", "type", "name"),
            "shot_xg": nested(ev, "shot", "statsbomb_xg"),
        }

    def pull(match_id):
        evs = get_statsbomb(f"events/{match_id}.json")
        return [flatten(e, match_id) for e in evs], collections.Counter(k for e in evs for k in e), len(evs)

    key_counts, total_events = collections.Counter(), 0
    for (cid, sid), grp in matches.groupby(["competition_id", "season_id"]):
        with ThreadPoolExecutor(4) as ex:
            results = list(ex.map(pull, grp.match_id))
        rows = [r for rs, _, _ in results for r in rs]
        for _, kc, n in results:
            key_counts.update(kc)
            total_events += n
        pd.DataFrame(rows).to_parquet(f"{INTERIM}/events/{cid}_{sid}.parquet", index=False)
        print(f"    {grp.competition.iat[0]} {grp.season.iat[0]}: {len(grp)} matches, {len(rows):,} events")

    # How often each raw field shows up - part of my initial data profiling
    presence = pd.DataFrame({"field": list(key_counts), "present": list(key_counts.values())})
    presence["pct_present"] = (100 * presence.present / total_events).round(2)
    presence.sort_values("pct_present").to_csv(f"{INTERIM}/sb_raw_field_presence.csv", index=False)
    print(f"    Total events: {total_events:,}")
    return matches


# =============================================================================
# SECTION 2 - TRANSFORM: EVENTS -> TEAM-MATCH METRICS
# =============================================================================
# One row per team per match, with my four candidate metrics plus xG.
# I defined each metric the way FBref does, so the two sources line up later:
#   - penalty shootouts (period 5) don't count toward anything
#   - throw-ins are not counted as passes
#   - a pass is complete when it has no outcome recorded
#   - possession % = the team's share of all passes attempted in the match
#   - a shot is on target if it was a goal, saved, or saved onto the post

def transform_statsbomb(matches):
    print("\n[2] Building team-match metrics...")
    cols = ["match_id", "period", "type", "team", "pass_outcome", "pass_type",
            "shot_outcome", "shot_xg", "x", "y", "duration"]
    ev = pd.concat([pd.read_parquet(f"{INTERIM}/events/{f}", columns=cols)
                    for f in os.listdir(f"{INTERIM}/events") if f.endswith(".parquet")],
                   ignore_index=True)

    profile.update({
        "events_total": len(ev),
        "events_period5": int((ev.period == 5).sum()),
        "matches_with_extra_time": int(ev.loc[ev.period.isin([3, 4]), "match_id"].nunique()),
        "matches_with_shootout": int(ev.loc[ev.period == 5, "match_id"].nunique()),
        "pct_events_missing_location": round(100 * ev.x.isna().mean(), 2),
        "pct_events_missing_duration": round(100 * ev.duration.isna().mean(), 2),
        "event_types": int(ev.type.nunique()),
    })
    ev = ev[ev.period != 5]

    # Problem I found: StatsBomb's own match file and event files don't agree
    # on three team names. I fix it per match, and only when the event name
    # isn't one of that match's two teams - "OL Reign" is the correct name in
    # NWSL 2023, so replacing it everywhere would have broken that season.
    name_fix = {"Aston Villa": "Aston Villa W",
                "Manchester United": "Manchester United W",
                "OL Reign": "Seattle Reign"}
    both = pd.concat([matches[["match_id", "home_team"]].rename(columns={"home_team": "t"}),
                      matches[["match_id", "away_team"]].rename(columns={"away_team": "t"})])
    valid = set(zip(both.match_id, both.t))
    bad = ~pd.Series([p in valid for p in zip(ev.match_id, ev.team)], index=ev.index) & ev.team.notna()
    profile["events_team_name_fixed"] = int(bad.sum())
    ev.loc[bad, "team"] = ev.loc[bad, "team"].replace(name_fix)

    passes = ev[(ev.type == "Pass") & (ev.pass_type.fillna("") != "Throw-in")]
    shots = ev[ev.type == "Shot"]
    on_target = {"Goal", "Saved", "Saved To Post"}

    tm = (passes.assign(complete=passes.pass_outcome.isna())
          .groupby(["match_id", "team"])
          .agg(passes_att=("complete", "size"), passes_cmp=("complete", "sum"))
          .join(shots.assign(sot=shots.shot_outcome.isin(on_target), goal=shots.shot_outcome.eq("Goal"))
                .groupby(["match_id", "team"])
                .agg(shots=("sot", "size"), shots_on_target=("sot", "sum"),
                     shot_goals=("goal", "sum"), xg=("shot_xg", "sum")), how="outer")
          .reset_index())

    # Own goals are logged as "Own Goal For" on the team that benefits
    og = ev[ev.type == "Own Goal For"].groupby(["match_id", "team"]).size().rename("own_goals_for")
    tm = tm.merge(og, on=["match_id", "team"], how="left")
    num = ["passes_att", "passes_cmp", "shots", "shots_on_target", "shot_goals", "xg", "own_goals_for"]
    tm[num] = tm[num].fillna(0)   # no shot rows means the team took zero shots - a real zero
    tm["event_goals"] = tm.shot_goals + tm.own_goals_for

    # Add match context: opponent, home/away, official score, result
    home = matches.rename(columns={"home_team": "team", "away_team": "opponent",
                                   "home_score": "goals", "away_score": "goals_against"}).assign(venue="Home")
    away = matches.rename(columns={"away_team": "team", "home_team": "opponent",
                                   "away_score": "goals", "home_score": "goals_against"}).assign(venue="Away")
    keep = ["match_id", "competition", "season", "international", "match_date", "stage",
            "team", "opponent", "venue", "goals", "goals_against"]
    tm = pd.concat([home[keep], away[keep]], ignore_index=True).merge(
        tm, on=["match_id", "team"], how="left", validate="one_to_one")

    opp = tm[["match_id", "team", "passes_att"]].rename(columns={"team": "opponent", "passes_att": "opp_passes_att"})
    tm = tm.merge(opp, on=["match_id", "opponent"], how="left")
    tm["pass_pct"] = 100 * tm.passes_cmp / tm.passes_att
    tm["possession_pct"] = 100 * tm.passes_att / (tm.passes_att + tm.opp_passes_att)
    tm["result"] = np.select([tm.goals > tm.goals_against, tm.goals < tm.goals_against], ["W", "L"], "D")
    tm["points"] = tm.result.map({"W": 3, "D": 1, "L": 0})
    tm["source"] = "StatsBomb"

    # My validation check: goals rebuilt from the events should equal the
    # official score. If they don't, my aggregation is wrong somewhere.
    tm["goal_mismatch"] = tm.event_goals != tm.goals
    profile["team_matches"] = len(tm)
    profile["goal_mismatch_rows"] = int(tm.goal_mismatch.sum())
    profile["goal_match_rate_pct"] = round(100 * (1 - tm.goal_mismatch.mean()), 2)
    print(f"    {len(tm)} team-matches; event goals match official score: {profile['goal_match_rate_pct']}%")

    tm.to_csv(f"{INTERIM}/sb_team_match.csv", index=False)
    pd.Series(profile).to_csv(f"{INTERIM}/sb_profile.csv", header=["value"])
    return tm


# =============================================================================
# SECTION 3 - EXTRACT + CLEAN: FBREF RESULTS
# =============================================================================
# FBref (Sports Reference) limits automated scraping, so instead of hitting
# the site directly I use the worldfootballR_data mirror on GitHub, which has
# the FBref match-results pages already scraped as R data files (.rds).

def extract_clean_fbref():
    print("\n[3] Downloading and cleaning FBref results...")
    for c in FB_COUNTRIES:
        path = f"{RAW}/fbref/{c}_match_results.rds"
        if not os.path.exists(path):
            r = session.get(f"{FB_BASE}{c}_match_results.rds", timeout=120)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

    raw = pd.concat([pyreadr.read_r(f"{RAW}/fbref/{c}_match_results.rds")[None] for c in FB_COUNTRIES],
                    ignore_index=True)
    log["fb_raw_rows_all_genders"] = len(raw)
    fb = raw[(raw.Gender == "F") & (raw.Tier == "1st")].copy()
    log["fb_raw_rows_women_top_flight"] = len(fb)

    # R marks missing numbers with its own special NA value. I swap those for
    # standard Python NaN so numpy doesn't throw warnings later (e.g. in log1p).
    for col in ["HomeGoals", "AwayGoals", "Home_xG", "Away_xG", "Attendance"]:
        fb[col] = fb[col].where(fb[col].notna(), np.nan).astype(float)

    # Formatting: proper date type, clean text, consistent round names
    fb["Date"] = pd.to_datetime(fb.Date, errors="coerce")
    fb["Notes"] = fb.Notes.fillna("").str.strip()
    fb["Round"] = fb.Round.fillna("Regular Season").replace({"Semifinals": "Semi-finals",
                                                              "First round": "First Round"})
    fb = fb.rename(columns={"Competition_Name": "competition", "Season_End_Year": "season_end_year"})
    fb["league"] = fb.competition.map({"National Women's Soccer League": "NWSL",
                                       "FA Women's Super League": "WSL",
                                       "Frauen-Bundesliga": "Frauen Bundesliga",
                                       "Liga F": "Liga F", "Serie A": "Serie A Women"})

    # Flags I use for the scope rules
    fb["cancelled"] = fb.Notes.str.contains("Cancelled|Suspended")
    fb["awarded"] = fb.Notes.str.startswith("Match awarded")
    fb["unplayed"] = fb.HomeGoals.isna()
    log["fb_rows_cancelled_or_suspended"] = int(fb.cancelled.sum())
    log["fb_rows_unplayed_no_score"] = int(fb.unplayed.sum())
    log["fb_rows_awarded"] = int(fb.awarded.sum())

    # Season completeness = matches played / matches scheduled
    comp = (fb.assign(played=~fb.unplayed & ~fb.cancelled)
              .groupby(["league", "season_end_year"]).played.agg(["sum", "size"]))
    comp["pct_played"] = (100 * comp["sum"] / comp["size"]).round(2)
    comp.reset_index().to_csv(f"{INTERIM}/fb_season_completeness.csv", index=False)

    # I drop any league-season under 90% played: the 2025/26 European seasons
    # and NWSL 2025 weren't finished when the mirror was captured. The one
    # exception is 2019/20 - those were cut short by COVID, so I keep them and
    # flag them instead of throwing a whole season away.
    incomplete = comp[comp.pct_played < 90].index
    fb = fb[~fb.set_index(["league", "season_end_year"]).index.isin(incomplete) | (fb.season_end_year == 2020)]
    fb["covid_curtailed"] = fb.season_end_year.eq(2020)
    log["fb_league_seasons_dropped_incomplete"] = ", ".join(f"{l} {s}" for l, s in incomplete if s != 2020)

    # Playoffs aren't part of the league table, and points per game comes
    # from the league table, so I leave them out.
    knockout = fb.Round.isin(["Semi-finals", "Final", "First Round"])
    log["fb_rows_playoff_knockout_excluded"] = int(knockout.sum())
    games = fb[~fb.unplayed & ~fb.cancelled & ~knockout].copy()
    log["fb_clean_matches"] = len(games)

    # Franchise key: FBref renames clubs over time, which would break each
    # team's season-to-season series. Same club -> one name. "Royals" is a
    # trap: it's two different clubs (the original Utah Royals folded; the
    # 2024 team is an expansion side), so I split them.
    franchise = {"Sky Blue": "Gotham FC", "Gotham": "Gotham FC",
                 "AC Milan": "Milan", "UC Sampdoria": "Sampdoria"}
    for col in ["Home", "Away"]:
        games[col] = games[col].replace(franchise)
        games.loc[(games[col] == "Royals") & (games.season_end_year <= 2020), col] = "Utah Royals (2018-20)"
        games.loc[(games[col] == "Royals") & (games.season_end_year >= 2024), col] = "Utah Royals (2024-)"

    # One row per team per fixture (home side and away side)
    def side(df, team, opp, gf, ga, xgf, xga, venue):
        return pd.DataFrame({"league": df.league, "season_end_year": df.season_end_year,
                             "date": df.Date, "team": df[team], "opponent": df[opp],
                             "goals": df[gf], "goals_against": df[ga], "xg": df[xgf],
                             "xg_against": df[xga], "attendance": df.Attendance, "venue": venue,
                             "awarded": df.awarded, "covid_curtailed": df.covid_curtailed,
                             "match_url": df.MatchURL})
    tmf = pd.concat([side(games, "Home", "Away", "HomeGoals", "AwayGoals", "Home_xG", "Away_xG", "Home"),
                     side(games, "Away", "Home", "AwayGoals", "HomeGoals", "Away_xG", "Home_xG", "Away")],
                    ignore_index=True)
    tmf["result"] = np.select([tmf.goals > tmf.goals_against, tmf.goals < tmf.goals_against], ["W", "L"], "D")
    tmf["points"] = tmf.result.map({"W": 3, "D": 1, "L": 0})
    tmf.loc[tmf.venue == "Away", "attendance"] = np.nan   # attendance belongs to the home team
    tmf.to_csv(f"{INTERIM}/fb_team_match.csv", index=False)

    # Success variables (one row per team per season):
    #   points per game = league points / matches played
    #   win rate        = wins / matches played
    # I use points per game, not total points per season, because season
    # lengths differ (NWSL went from 22 to 26 games, and COVID cut some short).
    # Awarded results count toward points but not toward performance stats.
    perf = tmf[~tmf.awarded]
    ts = (tmf.groupby(["league", "season_end_year", "team"])
             .agg(matches=("points", "size"), wins=("result", lambda s: (s == "W").sum()),
                  draws=("result", lambda s: (s == "D").sum()), points=("points", "sum"),
                  covid_curtailed=("covid_curtailed", "first"))
             .join(perf.groupby(["league", "season_end_year", "team"])
                       .agg(goals_pg=("goals", "mean"), goals_against_pg=("goals_against", "mean"),
                            xg_pg=("xg", "mean"), avg_home_attendance=("attendance", "mean")))
             .reset_index())
    ts["ppg"] = ts.points / ts.matches
    ts["win_rate"] = ts.wins / ts.matches
    log["fb_team_seasons"] = len(ts)
    print(f"    {log['fb_clean_matches']} clean league fixtures, {len(ts)} team-seasons")
    return tmf, ts


# =============================================================================
# SECTION 4 - INTEGRATE: LINKING STATSBOMB AND FBREF
# =============================================================================
# The two sources name teams differently ("Arsenal WFC" vs "Arsenal",
# "Bayern Munchen W" vs "Bayern Munich"). Instead of fuzzy string matching,
# I match on the games themselves: same league, same score, same home/away
# side, and a date within one day (NWSL night games land on different
# calendar dates in the two sources). Every match that lines up is a vote for
# a name pair; I keep the winning pair, one-to-one, with at least 3 games.

def integrate(sb, tmf, ts):
    print("\n[4] Linking StatsBomb teams to FBref...")
    sb = sb.copy()
    sb["match_date"] = pd.to_datetime(sb.match_date)
    sb["league"] = sb.competition.map({"NWSL": "NWSL", "FA Women's Super League": "WSL",
                                       "Frauen Bundesliga": "Frauen Bundesliga",
                                       "Liga F": "Liga F", "Serie A Women": "Serie A Women"})
    # Shared season key: "2023/2024" -> 2024, "2023" -> 2023
    sb["season_end_year"] = sb.season.astype(str).str[-4:].astype(int)
    clubs = sb[sb.league.notna()]

    fbk = tmf[["league", "date", "venue", "goals", "goals_against", "team", "opponent"]]
    cand = pd.concat([clubs.assign(_d=clubs.match_date + pd.Timedelta(days=k))
                           .merge(fbk, left_on=["league", "_d", "venue", "goals", "goals_against"],
                                  right_on=["league", "date", "venue", "goals", "goals_against"],
                                  suffixes=("_sb", "_fb"))
                      for k in (-1, 0, 1)], ignore_index=True)

    votes = cand.groupby(["league", "team_sb", "team_fb"]).size().rename("n").reset_index()
    taken, keep_rows = set(), []
    for _, r in votes.sort_values("n", ascending=False).iterrows():
        sb_used = {(l, s) for l, s, _ in taken}
        fb_used = {(l, f) for l, _, f in taken}
        if (r.league, r.team_sb) in sb_used or (r.league, r.team_fb) in fb_used:
            continue
        if r.n >= 3:
            taken.add((r.league, r.team_sb, r.team_fb))
            keep_rows.append(r)
    xwalk = pd.DataFrame(keep_rows)
    unmapped = set(map(tuple, clubs[["league", "team"]].drop_duplicates().values)) - set(zip(xwalk.league, xwalk.team_sb))
    log["crosswalk_unmapped"] = "; ".join(f"{l}: {t}" for l, t in sorted(unmapped))

    cand = cand.merge(xwalk[["league", "team_sb", "team_fb"]], on=["league", "team_sb", "team_fb"])
    name_map = xwalk.set_index(["league", "team_sb"]).team_fb
    cand = cand[[name_map.get((l, o)) == f for l, o, f in zip(cand.league, cand.opponent_sb, cand.opponent_fb)]]
    xwalk.to_csv(f"{INTERIM}/team_crosswalk.csv", index=False)

    log["sb_club_team_matches"] = len(clubs)
    log["sb_club_teams"] = clubs[["league", "team"]].drop_duplicates().shape[0]
    log["crosswalk_teams_mapped"] = len(xwalk)
    log["sb_team_matches_linked_to_fb"] = cand.drop_duplicates(["match_id", "team_sb"]).shape[0]
    log["link_rate_pct"] = round(100 * log["sb_team_matches_linked_to_fb"] / len(clubs), 2)

    # Team-season table: FBref is the backbone; StatsBomb metric averages are
    # added for the seasons StatsBomb covers.
    sb_ts = (clubs.assign(team_fb=[name_map.get((l, t)) for l, t in zip(clubs.league, clubs.team)])
                  .groupby(["league", "season_end_year", "team_fb"])
                  .agg(sb_matches=("match_id", "size"), sb_pass_pct=("pass_pct", "mean"),
                       sb_possession_pct=("possession_pct", "mean"),
                       sb_sot_pg=("shots_on_target", "mean"), sb_shots_pg=("shots", "mean"),
                       sb_xg_pg=("xg", "mean"), sb_goals_pg=("goals", "mean"))
                  .reset_index().rename(columns={"team_fb": "team"}))
    integrated = ts.merge(sb_ts, on=["league", "season_end_year", "team"], how="left")
    integrated["has_statsbomb"] = integrated.sb_matches.notna()
    log["integrated_team_seasons"] = len(integrated)
    log["integrated_with_statsbomb"] = int(integrated.has_statsbomb.sum())
    pd.Series(log).to_csv(f"{INTERIM}/fb_integration_log.csv", header=["value"])
    print(f"    {len(xwalk)}/{log['sb_club_teams']} teams mapped; {log['link_rate_pct']}% of team-matches linked")
    return xwalk, integrated


# =============================================================================
# SECTION 5 - PREPROCESS: MISSING DATA, OUTLIERS, SCALING, ENCODING, SPLIT
# =============================================================================

def preprocess(sb, fb, ts):
    print("\n[5] Preprocessing...")
    sb, fb, ts = sb.copy(), fb.copy(), ts.copy()

    # --- Variable reduction ---------------------------------------------------
    # Event goals were validated against the official score, so I drop the
    # duplicate goal fields and the working columns. Referee, venue name,
    # kick-off time and weekday were never carried over: they can't drive
    # team performance in this design.
    sb = sb.drop(columns=["shot_goals", "own_goals_for", "event_goals", "goal_mismatch",
                          "opp_passes_att", "source"])

    # --- Missing data -----------------------------------------------------------
    # My rule is no imputation. Most gaps are structural, and filling them in
    # would invent exactly the kind of trend I'm trying to measure.
    #   - FBref xG doesn't exist before the Opta era (2019 NWSL/WSL,
    #     2021 Serie A, 2023 Bundesliga/Liga F). Not imputed.
    #   - Attendance: I only use it for league-seasons where at least 70% of
    #     home games have it, and I never fill it with 0.
    fb["xg_available"] = fb.xg.notna()
    summary["fb_xg_missing_pct"] = round(100 * fb.xg.isna().mean(), 2)
    home = fb[fb.venue == "Home"]
    coverage = home.groupby(["league", "season_end_year"]).attendance.apply(lambda s: s.notna().mean())
    summary["attendance_missing_pct_by_league"] = home.groupby("league").attendance.apply(
        lambda s: round(100 * s.isna().mean(), 2)).to_dict()
    ts = ts.merge(coverage.rename("att_coverage").reset_index(), on=["league", "season_end_year"], how="left")
    ts.loc[ts.att_coverage < 0.70, "avg_home_attendance"] = np.nan
    summary["team_seasons_attendance_usable"] = int(ts.avg_home_attendance.notna().sum())

    # --- Outliers (flag them, keep them) -----------------------------------------
    # The extremes are real games (USA 13-0 Thailand, 60,160 at the Emirates),
    # so deleting them would erase real signal. I flag them so I can run my
    # models with and without them.
    outliers = {}
    for c in ["goals", "pass_pct", "possession_pct", "shots_on_target", "xg"]:
        sb[f"out_{c}"] = iqr_flag(sb[c])
        outliers[f"sb_{c}"] = int(sb[f"out_{c}"].sum())
    fb["out_attendance"] = iqr_flag(fb.attendance.dropna()).reindex(fb.index, fill_value=False)
    outliers["fb_attendance"] = int(fb.out_attendance.sum())
    summary["outlier_counts_iqr"] = outliers

    # --- Scaling ----------------------------------------------------------------
    # Attendance is strongly right-skewed, so I log-transform it.
    summary["attendance_skew_raw"] = round(fb.attendance.skew(), 2)
    fb["log_attendance"] = np.log1p(fb.attendance)
    summary["attendance_skew_log"] = round(fb.log_attendance.skew(), 2)
    ts["log_avg_home_attendance"] = np.log1p(ts.avg_home_attendance)

    # --- Encoding -----------------------------------------------------------------
    # Home/away -> 1/0, result -> win 1/0 for the logistic model, league ->
    # one-hot (Frauen Bundesliga is the reference level). Season stays a
    # number because it's an ordered time index, not a category.
    sb["is_home"] = (sb.venue == "Home").astype(int)
    sb["win"] = (sb.result == "W").astype(int)
    sb["international"] = sb.international.astype(int)
    sb["season_end_year"] = sb.season.astype(str).str[-4:].astype(int)
    sb["league"] = sb.competition.map({"FA Women's Super League": "WSL", "NWSL": "NWSL",
                                       "Frauen Bundesliga": "Frauen Bundesliga", "Liga F": "Liga F",
                                       "Serie A Women": "Serie A Women"}).fillna("International")
    sb = pd.concat([sb, pd.get_dummies(sb.league, prefix="lg", drop_first=True, dtype=int)], axis=1)

    # --- Class balance ------------------------------------------------------------
    # Win vs not-win is about 41/59 - mild. I'll use class weights in the
    # models rather than SMOTE, because made-up matches would break the time order.
    summary["result_dist_pct_fbref"] = fb.result.value_counts(normalize=True).mul(100).round(2).to_dict()
    summary["win_rate_pct_statsbomb"] = round(100 * sb.win.mean(), 2)

    # --- Temporal split -------------------------------------------------------------
    # Split by season, never randomly: train = up to 2022, validation = 2023,
    # test = 2024-2025. A whole season always sits in one split, so no model
    # learns from games played after its test window.
    def split(year):
        return "train" if year <= 2022 else ("validation" if year == 2023 else "test")
    for df in (sb, fb, ts):
        df["split"] = df.season_end_year.map(split)

    # --- Standardization (fit on training seasons only) -----------------------------
    # For the logistic regression screen. I take the mean and SD from the
    # training seasons only so nothing from the future leaks backward.
    scale_cols = ["pass_pct", "possession_pct", "shots_on_target", "xg", "goals"]
    train = sb[sb.split == "train"]
    mu, sd = train[scale_cols].mean(), train[scale_cols].std()
    for c in scale_cols:
        sb[f"z_{c}"] = (sb[c] - mu[c]) / sd[c]

    sb.to_csv(f"{PROCESSED}/team_match_statsbomb.csv", index=False)
    fb.to_csv(f"{PROCESSED}/team_match_fbref.csv", index=False)
    ts.to_csv(f"{PROCESSED}/team_season_integrated.csv", index=False)
    with open(f"{PROCESSED}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("    Done. Split sizes (FBref team-matches):", fb.split.value_counts().to_dict())
    return sb, fb, ts


# =============================================================================
# SECTION 6 - LOAD: SQLITE DATABASE
# =============================================================================
# Six tables. In R I read them with DBI + RSQLite for the EDA and modeling.

def load_sqlite(matches, sb, fb, ts, xwalk):
    print("\n[6] Loading data/capstone.db...")
    con = sqlite3.connect("data/capstone.db")
    tables = {"sb_matches": matches, "sb_team_match": sb, "fb_team_match": fb,
              "team_season": ts, "team_crosswalk": xwalk,
              "fb_season_completeness": pd.read_csv(f"{INTERIM}/fb_season_completeness.csv")}
    for name, df in tables.items():
        df.to_sql(name, con, if_exists="replace", index=False)
    con.executescript("""
        CREATE INDEX IF NOT EXISTS ix_sbtm ON sb_team_match(league, season_end_year, team);
        CREATE INDEX IF NOT EXISTS ix_fbtm ON fb_team_match(league, season_end_year, team);
        CREATE INDEX IF NOT EXISTS ix_ts   ON team_season(league, season_end_year, team);
    """)
    for name in tables:
        print(f"    {name}: {con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]:,} rows")
    con.close()


# =============================================================================
# SECTION 7 - VERIFY: EVERY NUMBER IN PRESENTATION 2
# =============================================================================
# This recomputes each figure on my slides straight from the database and
# compares it to what the slide says. Every line should say PASS.

def verify():
    print("\n[7] Checking the numbers in Presentation 2...")
    con = sqlite3.connect("data/capstone.db")
    q = lambda s: con.execute(s).fetchone()[0]
    split_counts = lambda t: "/".join(str(q(f"SELECT COUNT(*) FROM {t} WHERE split='{s}'"))
                                      for s in ["train", "validation", "test"])
    checks = [
        (1, "StatsBomb matches", 1310, q("SELECT COUNT(*) FROM sb_matches")),
        (1, "StatsBomb events", 4504875, profile["events_total"]),
        (1, "Clean FBref league fixtures", 5024, q("SELECT COUNT(*)/2 FROM fb_team_match")),
        (2, "StatsBomb competition-seasons", 13, q("SELECT COUNT(DISTINCT competition||season) FROM sb_matches")),
        (2, "Raw FBref women's top-flight fixtures", 6170, log["fb_raw_rows_women_top_flight"]),
        (3, "StatsBomb team-matches", 2620, q("SELECT COUNT(*) FROM sb_team_match")),
        (3, "Team-seasons", 439, q("SELECT COUNT(*) FROM team_season")),
        (3, "SQLite tables", 6, q("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")),
        (4, "Earliest FBref season", 2013, q("SELECT MIN(season_end_year) FROM fb_team_match")),
        (4, "Earliest StatsBomb club season", 2018, q("SELECT MIN(season_end_year) FROM sb_team_match")),
        (4, "Max seasons in any league", 11, q("SELECT MAX(n) FROM (SELECT COUNT(DISTINCT season_end_year) n FROM fb_team_match GROUP BY league)")),
        (5, "Teams mapped", 66, q("SELECT COUNT(*) FROM team_crosswalk")),
        (5, "Link rate %", 99.56, log["link_rate_pct"]),
        (5, "Goal reconciliation %", 100.0, profile["goal_match_rate_pct"]),
        (5, "Events with team name fixed", 78280, profile["events_team_name_fixed"]),
        (6, "FBref xG missing %", 34.38, round(q("SELECT 100.0*SUM(xg IS NULL)/COUNT(*) FROM fb_team_match"), 2)),
        (6, "NWSL home attendance missing %", 3.18, round(q("SELECT 100.0*SUM(attendance IS NULL)/COUNT(*) FROM fb_team_match WHERE venue='Home' AND league='NWSL'"), 2)),
        (6, "Liga F home attendance missing %", 100.0, round(q("SELECT 100.0*SUM(attendance IS NULL)/COUNT(*) FROM fb_team_match WHERE venue='Home' AND league='Liga F'"), 2)),
        (6, "Team-seasons with usable attendance", 301, q("SELECT COUNT(*) FROM team_season WHERE avg_home_attendance IS NOT NULL")),
        (6, "Unplayed fixtures (no score)", 877, log["fb_rows_unplayed_no_score"]),
        (6, "Event duration missing %", 25.18, profile["pct_events_missing_duration"]),
        (7, "Cancelled or suspended fixtures", 200, log["fb_rows_cancelled_or_suspended"]),
        (7, "Playoff fixtures excluded", 43, log["fb_rows_playoff_knockout_excluded"]),
        (7, "Awarded results", 3, log["fb_rows_awarded"]),
        (7, "Goals outlier flags", 59, q("SELECT SUM(out_goals) FROM sb_team_match")),
        (7, "xG outlier flags", 92, q("SELECT SUM(out_xg) FROM sb_team_match")),
        (7, "Shots-on-target outlier flags", 51, q("SELECT SUM(out_shots_on_target) FROM sb_team_match")),
        (7, "Pass % outlier flags", 30, q("SELECT SUM(out_pass_pct) FROM sb_team_match")),
        (7, "Possession outlier flags", 0, q("SELECT SUM(out_possession_pct) FROM sb_team_match")),
        (7, "Attendance outlier flags", 326, q("SELECT SUM(out_attendance) FROM fb_team_match")),
        (7, "Max goals in a match", 13, q("SELECT MAX(goals) FROM sb_team_match")),
        (7, "Max shots on target", 20, q("SELECT MAX(shots_on_target) FROM sb_team_match")),
        (7, "Max xG", 6.19, round(q("SELECT MAX(xg) FROM sb_team_match"), 2)),
        (7, "Min pass %", 42.55, round(q("SELECT MIN(pass_pct) FROM sb_team_match"), 2)),
        (7, "Max attendance", 60160, q("SELECT MAX(attendance) FROM fb_team_match")),
        (8, "Win %", 40.76, round(q("SELECT 100.0*SUM(result='W')/COUNT(*) FROM fb_team_match"), 2)),
        (8, "Draw %", 18.47, round(q("SELECT 100.0*SUM(result='D')/COUNT(*) FROM fb_team_match"), 2)),
        (8, "Attendance skew (raw)", 3.52, summary["attendance_skew_raw"]),
        (8, "Attendance skew (log1p)", -0.01, summary["attendance_skew_log"]),
        (9, "FBref train / val / test", "5616/1532/2900", split_counts("fb_team_match")),
        (9, "StatsBomb train / val / test", "888/402/1330", split_counts("sb_team_match")),
        (9, "Team-season train / val / test", "263/62/114", split_counts("team_season")),
    ]
    con.close()

    fails = 0
    print(f"    {'Slide':<6}{'Figure':<40}{'On slide':>16}{'Recomputed':>16}  Result")
    for slide, label, shown, got in checks:
        ok = shown == got if isinstance(shown, str) else abs(float(shown) - float(got)) < 0.006
        fails += not ok
        print(f"    {slide:<6}{label:<40}{str(shown):>16}{str(got):>16}  {'PASS' if ok else 'FAIL'}")
    print(f"\n    {len(checks) - fails}/{len(checks)} figures reproduced from the data.")
    if fails:
        print("    A FAIL usually means StatsBomb or the FBref mirror published new data since 10/8/2026.")


# =============================================================================
# SECTION 8 - SQL PIPELINE: INTEGRATION + MODELING VIEWS
# =============================================================================
# Python loads the cleaned tables; the relational work lives in
# capstone_sql_pipeline.sql (same folder). It builds data-quality checks,
# rebuilds the success variables and the StatsBomb-FBref join in SQL, creates
# the modeling datasets for Presentations 3 and 4, and reconciles SQL against
# Python. Every check below should show 0.

def run_sql_pipeline():
    print("\n[8] Running capstone_sql_pipeline.sql...")
    con = sqlite3.connect("data/capstone.db")
    with open("capstone_sql_pipeline.sql", encoding="utf-8") as f:
        con.executescript(f.read())

    print("    Data-quality checks (problem rows):")
    for name, n in con.execute("SELECT check_name, problem_rows FROM dq_checks"):
        print(f"      {'OK  ' if n == 0 else 'FAIL'} {name}: {n}")

    print("    SQL vs Python reconciliation (differences):")
    for name, n in con.execute("SELECT check_name, difference FROM reconciliation"):
        print(f"      {'OK  ' if n == 0 else 'FAIL'} {name}: {n}")

    print("    Modeling views:")
    for view in ["v_team_season_success", "v_integrated_team_season", "v_league_season_trends",
                 "v_league_month_series", "v_logistic_screen"]:
        print(f"      {view}: {con.execute(f'SELECT COUNT(*) FROM {view}').fetchone()[0]:,} rows")
    con.close()


# =============================================================================
# RUN EVERYTHING
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    matches = extract_statsbomb()
    sb_tm = transform_statsbomb(matches)
    fb_tm, team_season = extract_clean_fbref()
    crosswalk, integrated = integrate(sb_tm, fb_tm, team_season)
    sb_final, fb_final, ts_final = preprocess(sb_tm, fb_tm, integrated)
    load_sqlite(matches, sb_final, fb_final, ts_final, crosswalk)
    verify()
    run_sql_pipeline()
    print(f"\nFinished in {(time.time() - start) / 60:.2f} minutes.")
