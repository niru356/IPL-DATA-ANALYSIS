"""
Data Analysis & Analytical Datasets Engine for IPL Dataset.
Builds delivery-level, match-level, innings-level, batter-level, bowler-level,
team-level, venue-level, over-level, extras, and advanced statistical datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def create_match_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the official match-level dataset (1 row per unique match).
    Enriches with scores, innings totals, winner strategy, and toss impact.
    """
    # Base match attributes
    match_cols = [
        "match_id", "date", "season_clean", "season_year", "venue", "city",
        "match_type", "event_name", "toss_winner", "toss_decision",
        "match_won_by", "win_outcome", "player_of_match", "result_type",
        "method", "superover_winner"
    ]
    avail_cols = [c for c in match_cols if c in df.columns]
    matches = df[avail_cols].drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    # Calculate innings scores per match
    innings_summary = (
        df.groupby(["match_id", "innings"], as_index=False)
        .agg(
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
            total_runs=("runs_total", "sum"),
            wickets=("is_wicket", "sum")
        )
    )

    # Separate 1st and 2nd innings
    inn1 = innings_summary[innings_summary["innings"] == 1].rename(
        columns={"batting_team": "team1", "bowling_team": "team2", "total_runs": "team1_score", "wickets": "team1_wickets"}
    ).drop(columns=["innings"])

    inn2 = innings_summary[innings_summary["innings"] == 2].rename(
        columns={"batting_team": "team2_actual", "bowling_team": "team1_actual", "total_runs": "team2_score", "wickets": "team2_wickets"}
    ).drop(columns=["innings", "team2_actual", "team1_actual"])

    # Merge innings into matches
    matches = matches.merge(inn1, on="match_id", how="left")
    matches = matches.merge(inn2, on="match_id", how="left")

    # Winner Strategy (Batting First vs Chasing)
    matches["batting_first_team"] = matches["team1"]
    matches["chasing_team"] = matches["team2"]

    conditions = [
        matches["match_won_by"] == matches["batting_first_team"],
        matches["match_won_by"] == matches["chasing_team"]
    ]
    choices = ["Batting First", "Chasing"]
    matches["winner_type"] = np.select(conditions, choices, default="Tie / No Result")

    # Toss winner match win conversion
    if "toss_winner" in matches.columns and "match_won_by" in matches.columns:
        matches["toss_winner_won"] = (matches["toss_winner"] == matches["match_won_by"])
    else:
        matches["toss_winner_won"] = False

    return matches


def create_innings_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the official innings-level dataset (1 row per innings).
    """
    # Delivery-level aggregations per innings
    innings_df = (
        df.groupby(["match_id", "innings"], as_index=False)
        .agg(
            date=("date", "first") if "date" in df.columns else ("match_id", "first"),
            season_clean=("season_clean", "first") if "season_clean" in df.columns else ("match_id", "first"),
            season_year=("season_year", "first") if "season_year" in df.columns else ("match_id", "first"),
            venue=("venue", "first") if "venue" in df.columns else ("match_id", "first"),
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
            total_runs=("runs_total", "sum"),
            batter_runs=("runs_batter", "sum"),
            extras=("runs_extras", "sum"),
            valid_balls=("valid_ball", "sum"),
            wickets=("is_wicket", "sum"),
            bowler_wickets=("is_bowler_wicket", "sum"),
            fours=("is_four", "sum") if "is_four" in df.columns else ("runs_batter", lambda x: (x == 4).sum()),
            sixes=("is_six", "sum") if "is_six" in df.columns else ("runs_batter", lambda x: (x == 6).sum()),
            dot_balls=("is_dot_ball", "sum") if "is_dot_ball" in df.columns else ("runs_total", lambda x: (x == 0).sum())
        )
    )

    # Calculate overs and run rate
    innings_df["overs"] = (innings_df["valid_balls"] / 6).round(2)
    innings_df["run_rate"] = np.where(
        innings_df["overs"] > 0,
        (innings_df["total_runs"] / (innings_df["valid_balls"] / 6)).round(2),
        0.0
    )

    # Boundaries and percentages
    innings_df["boundaries"] = innings_df["fours"] + innings_df["sixes"]
    innings_df["boundary_runs"] = innings_df["fours"] * 4 + innings_df["sixes"] * 6
    innings_df["boundary_pct"] = np.where(
        innings_df["total_runs"] > 0,
        ((innings_df["boundary_runs"] / innings_df["total_runs"]) * 100).round(2),
        0.0
    )
    innings_df["dot_ball_pct"] = np.where(
        innings_df["valid_balls"] > 0,
        ((innings_df["dot_balls"] / innings_df["valid_balls"]) * 100).round(2),
        0.0
    )

    return innings_df


def create_batting_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the official batter-level dataset with complete career metrics,
    averages, strike rates, 50s, 100s, and highest score.
    """
    # 1. Base aggregations per batter
    batting = (
        df.groupby("batter", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            innings=("match_id", lambda x: len(x.unique())),
            runs=("runs_batter", "sum"),
            balls=("balls_faced", "sum"),
            fours=("is_four", "sum") if "is_four" in df.columns else ("runs_batter", lambda x: (x == 4).sum()),
            sixes=("is_six", "sum") if "is_six" in df.columns else ("runs_batter", lambda x: (x == 6).sum()),
            dot_balls=("is_dot_ball", "sum") if "is_dot_ball" in df.columns else ("runs_total", lambda x: (x == 0).sum())
        )
    )

    # 2. Count actual innings batted (where batter faced balls or scored runs)
    batter_innings = (
        df.groupby(["batter", "match_id"], as_index=False)
        .agg(
            match_runs=("runs_batter", "sum"),
            match_balls=("balls_faced", "sum")
        )
    )
    innings_count = batter_innings.groupby("batter")["match_id"].count().reset_index(name="innings_batted")
    batting = batting.merge(innings_count, on="batter", how="left")
    batting["innings"] = batting["innings_batted"].fillna(batting["innings"]).astype(int)
    batting = batting.drop(columns=["innings_batted"])

    # 3. Calculate 50s, 100s, and Highest Score per batter
    milestones = (
        batter_innings.groupby("batter")
        .agg(
            highest_score=("match_runs", "max"),
            fifties=("match_runs", lambda x: ((x >= 50) & (x < 100)).sum()),
            hundreds=("match_runs", lambda x: (x >= 100).sum())
        )
        .reset_index()
    )
    batting = batting.merge(milestones, on="batter", how="left")

    # 4. Dismissals count
    dismissed = (
        df[df["player_out"].notna()]
        .groupby("player_out")
        .size()
        .reset_index(name="dismissals")
        .rename(columns={"player_out": "batter"})
    )
    batting = batting.merge(dismissed, on="batter", how="left")
    batting["dismissals"] = batting["dismissals"].fillna(0).astype(int)
    batting["not_outs"] = np.maximum(0, batting["innings"] - batting["dismissals"])

    # 5. Strike Rate & Batting Average
    batting["strike_rate"] = np.where(
        batting["balls"] > 0,
        ((batting["runs"] / batting["balls"]) * 100).round(2),
        0.0
    )
    batting["batting_average"] = np.where(
        batting["dismissals"] > 0,
        (batting["runs"] / batting["dismissals"]).round(2),
        batting["runs"].astype(float)
    )

    # 6. Boundary & Dot Ball Percentages
    batting["boundary_runs"] = batting["fours"] * 4 + batting["sixes"] * 6
    batting["boundary_pct"] = np.where(
        batting["runs"] > 0,
        ((batting["boundary_runs"] / batting["runs"]) * 100).round(2),
        0.0
    )
    batting["dot_ball_pct"] = np.where(
        batting["balls"] > 0,
        ((batting["dot_balls"] / batting["balls"]) * 100).round(2),
        0.0
    )

    return batting.sort_values("runs", ascending=False).reset_index(drop=True)


def create_bowling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the official bowler-level dataset with overs, economy, wickets,
    strike rate, average, 4w/5w hauls, and dot ball rate.
    """
    # 1. Base bowler aggregations
    bowling = (
        df.groupby("bowler", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            valid_balls=("valid_ball", "sum"),
            runs_conceded=("runs_bowler", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            total_wickets=("is_wicket", "sum"),
            fours_conceded=("is_four", "sum") if "is_four" in df.columns else ("runs_batter", lambda x: (x == 4).sum()),
            sixes_conceded=("is_six", "sum") if "is_six" in df.columns else ("runs_batter", lambda x: (x == 6).sum()),
            dot_balls=("is_dot_ball", "sum") if "is_dot_ball" in df.columns else ("runs_total", lambda x: (x == 0).sum())
        )
    )

    # 2. Overs, Economy, Average, Strike Rate
    bowling["overs"] = (bowling["valid_balls"] / 6).round(1)
    bowling["economy"] = np.where(
        bowling["overs"] > 0,
        (bowling["runs_conceded"] / (bowling["valid_balls"] / 6)).round(2),
        np.nan
    )
    bowling["bowling_average"] = np.where(
        bowling["wickets"] > 0,
        (bowling["runs_conceded"] / bowling["wickets"]).round(2),
        np.nan
    )
    bowling["bowling_strike_rate"] = np.where(
        bowling["wickets"] > 0,
        (bowling["valid_balls"] / bowling["wickets"]).round(2),
        np.nan
    )
    bowling["dot_ball_pct"] = np.where(
        bowling["valid_balls"] > 0,
        ((bowling["dot_balls"] / bowling["valid_balls"]) * 100).round(2),
        0.0
    )

    # 3. Match-by-match bowling performance (4w and 5w hauls, best bowling figures)
    match_bowling = (
        df.groupby(["bowler", "match_id"], as_index=False)
        .agg(
            m_wickets=("is_bowler_wicket", "sum"),
            m_runs=("runs_bowler", "sum")
        )
    )

    hauls = (
        match_bowling.groupby("bowler")
        .agg(
            innings_bowled=("match_id", "count"),
            four_wickets=("m_wickets", lambda x: ((x == 4)).sum()),
            five_wickets=("m_wickets", lambda x: (x >= 5).sum())
        )
        .reset_index()
    )
    bowling = bowling.merge(hauls, on="bowler", how="left")

    return bowling.sort_values("wickets", ascending=False).reset_index(drop=True)


def create_team_dataset(matches: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the team-level dataset with win rates, average scores, chasing vs defending records.
    """
    # Collect all teams from batting_team and bowling_team
    teams = sorted(list(set(innings["batting_team"].dropna().unique()).union(set(innings["bowling_team"].dropna().unique()))))

    team_records = []
    for team in teams:
        # Match involvements
        team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        total_played = len(team_matches)
        wins = (matches["match_won_by"] == team).sum()
        losses = total_played - wins
        win_pct = round((wins / total_played) * 100, 2) if total_played > 0 else 0.0

        # Innings batting stats
        team_inn = innings[innings["batting_team"] == team]
        inn_count = len(team_inn)
        total_runs = team_inn["total_runs"].sum()
        total_wickets_lost = team_inn["wickets"].sum()
        avg_score = round(team_inn["total_runs"].mean(), 2) if inn_count > 0 else 0.0
        highest_score = int(team_inn["total_runs"].max()) if inn_count > 0 else 0
        lowest_score = int(team_inn["total_runs"].min()) if inn_count > 0 else 0
        avg_run_rate = round(team_inn["run_rate"].mean(), 2) if inn_count > 0 else 0.0

        # Batting first vs Chasing
        bat_first_matches = matches[matches["batting_first_team"] == team]
        bat_first_played = len(bat_first_matches)
        bat_first_wins = (bat_first_matches["match_won_by"] == team).sum()
        bat_first_win_pct = round((bat_first_wins / bat_first_played) * 100, 2) if bat_first_played > 0 else 0.0

        chase_matches = matches[matches["chasing_team"] == team]
        chase_played = len(chase_matches)
        chase_wins = (chase_matches["match_won_by"] == team).sum()
        chase_win_pct = round((chase_wins / chase_played) * 100, 2) if chase_played > 0 else 0.0

        # Toss stats
        toss_won = (matches["toss_winner"] == team).sum()
        toss_win_pct = round((toss_won / total_played) * 100, 2) if total_played > 0 else 0.0
        toss_and_match = ((matches["toss_winner"] == team) & (matches["match_won_by"] == team)).sum()
        toss_match_win_pct = round((toss_and_match / toss_won) * 100, 2) if toss_won > 0 else 0.0

        team_records.append({
            "team": team,
            "matches": total_played,
            "wins": wins,
            "losses": losses,
            "win_percentage": win_pct,
            "innings_batted": inn_count,
            "total_runs": total_runs,
            "total_wickets_lost": total_wickets_lost,
            "average_score": avg_score,
            "average_run_rate": avg_run_rate,
            "highest_score": highest_score,
            "lowest_score": lowest_score,
            "batting_first_matches": bat_first_played,
            "batting_first_wins": bat_first_wins,
            "batting_first_win_pct": bat_first_win_pct,
            "chasing_matches": chase_played,
            "chasing_wins": chase_wins,
            "chasing_win_pct": chase_win_pct,
            "toss_won": toss_won,
            "toss_win_pct": toss_win_pct,
            "toss_and_match_wins": toss_and_match,
            "toss_and_match_win_pct": toss_match_win_pct
        })

    team_df = pd.DataFrame(team_records)
    return team_df.sort_values("wins", ascending=False).reset_index(drop=True)


def create_venue_dataset(matches: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the venue-level dataset with match counts, average 1st and 2nd innings scores,
    and chasing success rates.
    """
    venue_group = matches.groupby("venue", as_index=False).agg(
        matches=("match_id", "nunique"),
        city=("city", "first")
    )

    # Merge innings scores by venue
    inn_venue = innings.groupby(["venue", "innings"], as_index=False)["total_runs"].mean()
    inn1_avg = inn_venue[inn_venue["innings"] == 1].rename(columns={"total_runs": "avg_1st_innings_score"}).drop(columns=["innings"])
    inn2_avg = inn_venue[inn_venue["innings"] == 2].rename(columns={"total_runs": "avg_2nd_innings_score"}).drop(columns=["innings"])

    venue_group = venue_group.merge(inn1_avg, on="venue", how="left")
    venue_group = venue_group.merge(inn2_avg, on="venue", how="left")

    # High / low scores by venue
    venue_extremes = innings.groupby("venue")["total_runs"].agg(
        highest_score="max",
        lowest_score="min"
    ).reset_index()
    venue_group = venue_group.merge(venue_extremes, on="venue", how="left")

    # Chasing wins by venue
    chase_by_venue = (
        matches[matches["winner_type"] == "Chasing"]
        .groupby("venue")["match_id"]
        .count()
        .reset_index(name="chasing_wins")
    )
    venue_group = venue_group.merge(chase_by_venue, on="venue", how="left")
    venue_group["chasing_wins"] = venue_group["chasing_wins"].fillna(0).astype(int)
    venue_group["chasing_win_pct"] = np.where(
        venue_group["matches"] > 0,
        ((venue_group["chasing_wins"] / venue_group["matches"]) * 100).round(2),
        0.0
    )

    venue_group["avg_1st_innings_score"] = venue_group["avg_1st_innings_score"].round(1)
    venue_group["avg_2nd_innings_score"] = venue_group["avg_2nd_innings_score"].round(1)

    return venue_group.sort_values("matches", ascending=False).reset_index(drop=True)


def create_over_phase_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates over-by-over and phase-level analytical datasets.
    """
    # Over by over (1 to 20)
    over_df = (
        df.groupby("over_number", as_index=False)
        .agg(
            phase=("phase", "first"),
            runs=("runs_total", "sum"),
            batter_runs=("runs_batter", "sum"),
            extras=("runs_extras", "sum"),
            valid_balls=("valid_ball", "sum"),
            wickets=("is_wicket", "sum"),
            bowler_wickets=("is_bowler_wicket", "sum"),
            fours=("is_four", "sum") if "is_four" in df.columns else ("runs_batter", lambda x: (x == 4).sum()),
            sixes=("is_six", "sum") if "is_six" in df.columns else ("runs_batter", lambda x: (x == 6).sum()),
            dot_balls=("is_dot_ball", "sum") if "is_dot_ball" in df.columns else ("runs_total", lambda x: (x == 0).sum())
        )
    )
    over_df["overs_bowled"] = (over_df["valid_balls"] / 6).round(1)
    over_df["run_rate"] = np.where(
        over_df["overs_bowled"] > 0,
        (over_df["runs"] / over_df["overs_bowled"]).round(2),
        0.0
    )
    over_df["dot_ball_pct"] = np.where(
        over_df["valid_balls"] > 0,
        ((over_df["dot_balls"] / over_df["valid_balls"]) * 100).round(2),
        0.0
    )

    # Phase summary (Powerplay, Middle Overs, Death Overs)
    phase_order = ["Powerplay", "Middle Overs", "Death Overs"]
    phase_df = (
        df.groupby("phase", as_index=False)
        .agg(
            runs=("runs_total", "sum"),
            valid_balls=("valid_ball", "sum"),
            wickets=("is_wicket", "sum"),
            fours=("is_four", "sum") if "is_four" in df.columns else ("runs_batter", lambda x: (x == 4).sum()),
            sixes=("is_six", "sum") if "is_six" in df.columns else ("runs_batter", lambda x: (x == 6).sum()),
            dot_balls=("is_dot_ball", "sum") if "is_dot_ball" in df.columns else ("runs_total", lambda x: (x == 0).sum())
        )
    )
    phase_df["overs"] = (phase_df["valid_balls"] / 6).round(1)
    phase_df["run_rate"] = np.where(
        phase_df["overs"] > 0,
        (phase_df["runs"] / phase_df["overs"]).round(2),
        0.0
    )
    phase_df["dot_ball_pct"] = np.where(
        phase_df["valid_balls"] > 0,
        ((phase_df["dot_balls"] / phase_df["valid_balls"]) * 100).round(2),
        0.0
    )
    phase_df["boundaries"] = phase_df["fours"] + phase_df["sixes"]
    phase_df["boundary_runs"] = phase_df["fours"] * 4 + phase_df["sixes"] * 6
    phase_df["boundary_pct"] = np.where(
        phase_df["runs"] > 0,
        ((phase_df["boundary_runs"] / phase_df["runs"]) * 100).round(2),
        0.0
    )

    # Sort phase by logical cricket order
    phase_df["phase"] = pd.Categorical(phase_df["phase"], categories=phase_order, ordered=True)
    phase_df = phase_df.sort_values("phase").reset_index(drop=True)

    return over_df, phase_df


def create_extras_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates extras breakdown by type (overall, season, and team).
    """
    # By type
    extras_type = (
        df[df["extra_type"].notna()]
        .groupby("extra_type", as_index=False)
        .agg(
            deliveries=("extra_type", "count"),
            extra_runs=("runs_extras", "sum")
        )
        .sort_values("extra_runs", ascending=False)
        .reset_index(drop=True)
    )

    # By season
    extras_season = (
        df.groupby("season_clean", as_index=False)
        .agg(
            total_extras=("runs_extras", "sum"),
            total_runs=("runs_total", "sum")
        )
    )
    extras_season["extras_pct"] = ((extras_season["total_extras"] / extras_season["total_runs"]) * 100).round(2)
    extras_season = extras_season.sort_values("season_clean").reset_index(drop=True)

    return extras_type, extras_season


def create_toss_dataset(matches: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes toss decisions, toss winner match win percentage, and toss impact by decision.
    """
    total_matches = len(matches)
    decision_counts = matches["toss_decision"].value_counts().to_dict()
    toss_and_match_win_pct = round(matches["toss_winner_won"].mean() * 100, 2)

    decision_impact = (
        matches.groupby("toss_decision")["toss_winner_won"]
        .agg(matches="count", win_pct=lambda x: round(x.mean() * 100, 2))
        .reset_index()
    )

    return {
        "total_matches": total_matches,
        "decision_counts": decision_counts,
        "overall_toss_win_conversion_pct": toss_and_match_win_pct,
        "decision_impact_df": decision_impact
    }


def compute_advanced_stats(df: pd.DataFrame, matches: pd.DataFrame, innings: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes correlations, IQR outlier detection for scores, and distributions.
    """
    # 1. Correlation Matrix
    corr_cols = [
        "runs_batter", "runs_extras", "runs_total", "valid_ball",
        "runs_bowler", "balls_faced", "is_wicket", "is_four", "is_six"
    ]
    avail_corr = [c for c in corr_cols if c in df.columns]
    correlation_matrix = df[avail_corr].corr().round(3)

    # 2. IQR Outliers for Team Scores
    q1 = innings["total_runs"].quantile(0.25)
    q3 = innings["total_runs"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_innings = innings[
        (innings["total_runs"] < lower_bound) | (innings["total_runs"] > upper_bound)
    ].sort_values("total_runs", ascending=False)

    return {
        "correlation_matrix": correlation_matrix,
        "iqr_stats": {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outlier_innings)
        },
        "outlier_innings": outlier_innings
    }


def build_all_analytical_datasets(df_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """
    High-level orchestrator that generates all analytical datasets in one pass.
    """
    matches = create_match_dataset(df_cleaned)
    innings = create_innings_dataset(df_cleaned)
    batting = create_batting_dataset(df_cleaned)
    bowling = create_bowling_dataset(df_cleaned)
    teams = create_team_dataset(matches, innings)
    venues = create_venue_dataset(matches, innings)
    over_df, phase_df = create_over_phase_dataset(df_cleaned)
    extras_type, extras_season = create_extras_dataset(df_cleaned)
    toss_stats = create_toss_dataset(matches)
    advanced = compute_advanced_stats(df_cleaned, matches, innings)

    return {
        "matches": matches,
        "innings": innings,
        "batting": batting,
        "bowling": bowling,
        "teams": teams,
        "venues": venues,
        "overs": over_df,
        "phases": phase_df,
        "extras_type": extras_type,
        "extras_season": extras_season,
        "toss_stats": toss_stats,
        "advanced": advanced
    }
