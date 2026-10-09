# 05_fbref_team_stats.R
# NEXT STEP (run locally): pull FBref team-season passing, possession and shooting
# tables for the women's top flights, so passing accuracy, possession % and shots on
# target have a longer history than StatsBomb's 9 club league-seasons.
#
# NOT YET RUN: fbref.com was not reachable from the environment the rest of the
# pipeline was built in. Check the output column names after the first run and adjust
# the select() calls at the bottom.
#
# Requires: install.packages(c("worldfootballR", "dplyr", "purrr", "readr"))
# worldfootballR pauses between requests (time_pause) to stay well under
# Sports Reference's bot rate limits; do not lower it.

library(worldfootballR)
library(dplyr)
library(purrr)
library(readr)

dir.create("data/raw/fbref_team_stats", recursive = TRUE, showWarnings = FALSE)

# league (country code) and first season with Opta-era stats, from the xG-coverage profiling
leagues <- tibble::tribble(
  ~country, ~first_season,
  "USA",    2019,   # NWSL (calendar-year seasons)
  "ENG",    2019,   # WSL
  "ITA",    2021,   # Serie A Women
  "GER",    2023,   # Frauen-Bundesliga
  "ESP",    2023    # Liga F
)
stat_types <- c("passing", "possession", "shooting")

pull_one <- function(country, season, stat) {
  out <- file.path("data/raw/fbref_team_stats", sprintf("%s_%d_%s.csv", country, season, stat))
  if (file.exists(out)) return(read_csv(out, show_col_types = FALSE))   # resume-safe
  df <- tryCatch(
    fb_season_team_stats(country = country, gender = "F", season_end_year = season,
                         tier = "1st", stat_type = stat, time_pause = 6),
    error = function(e) { message("FAILED ", country, " ", season, " ", stat, ": ", e$message); NULL })
  if (!is.null(df) && nrow(df) > 0) write_csv(df, out)
  df
}

jobs <- leagues %>%
  mutate(season = map(first_season, ~ .x:2025)) %>%
  tidyr::unnest(season) %>%
  tidyr::crossing(stat = stat_types)

results <- pmap(list(jobs$country, jobs$season, jobs$stat), pull_one)

# Quick look at what came back, so the select() below can be fixed to real names
walk(stat_types, function(s) {
  f <- list.files("data/raw/fbref_team_stats", pattern = paste0("_", s, "\\.csv$"), full.names = TRUE)
  if (length(f)) { message(s, " columns:"); print(names(read_csv(f[1], show_col_types = FALSE))) }
})

# Then: filter Team_or_Opponent == "team", keep Squad, Season_End_Year and the
# pass completion %, possession % and shots-on-target columns, apply the same
# FRANCHISE renames as 03_clean_fbref_integrate.py, and join to team_season in
# data/capstone.db on (league, season_end_year, team).
