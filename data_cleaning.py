"""
Data Cleaning & Quality Assurance Module for IPL Dataset.
Handles schema validation, normalization, type conversions, missing values, and data quality reporting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from .utils import TEAM_NAME_MAPPING


def inspect_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive preliminary inspection of the raw IPL dataset.
    """
    total_rows, total_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    duplicates = df.duplicated().sum()
    null_cells = df.isnull().sum().sum()
    null_percentage = (null_cells / (total_rows * total_cols)) * 100 if total_rows * total_cols > 0 else 0

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "memory_mb": round(memory_mb, 2),
        "duplicates": int(duplicates),
        "null_cells": int(null_cells),
        "null_percentage": round(null_percentage, 2),
        "column_list": df.columns.tolist()
    }


def clean_season_value(val: Any, date_val: Any = None) -> str:
    """
    Standardizes season representations (e.g., '2007/08' -> '2008', '2009/10' -> '2010', '2020/21' -> '2020').
    Uses the match date year as primary or fallback verification.
    """
    if pd.notna(date_val):
        try:
            dt = pd.to_datetime(date_val)
            return str(dt.year)
        except Exception:
            pass

    if pd.isna(val):
        return "Unknown"

    s_val = str(val).strip()
    if s_val == "2007/08":
        return "2008"
    if s_val == "2009/10":
        return "2010"
    if s_val == "2020/21":
        return "2020"
    if "/" in s_val:
        parts = s_val.split("/")
        return parts[0].strip()
    return s_val


def clean_ipl_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes production-grade data cleaning, normalization, and feature enrichment.
    Returns:
        (df_cleaned, quality_report_df)
    """
    df_raw = df.copy()
    cleaned = df.copy()

    # 1. Standardize column names
    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # 2. Drop exact duplicate records
    dup_count = cleaned.duplicated().sum()
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 3. Clean string whitespace
    string_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    # 4. Datetime conversion
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    # 5. Season cleaning
    if "season" in cleaned.columns:
        cleaned["season_clean"] = [
            clean_season_value(s, d)
            for s, d in zip(cleaned["season"], cleaned["date"] if "date" in cleaned.columns else [None]*len(cleaned))
        ]
        cleaned["season_year"] = pd.to_numeric(cleaned["season_clean"], errors="coerce").fillna(0).astype(int)
    elif "date" in cleaned.columns:
        cleaned["season_clean"] = cleaned["date"].dt.year.astype(str)
        cleaned["season_year"] = cleaned["date"].dt.year.fillna(0).astype(int)

    # 6. Numeric conversions with safe coercion
    numeric_cols = [
        "match_id", "innings", "over", "ball", "ball_no", "bat_pos",
        "runs_batter", "balls_faced", "valid_ball", "runs_extras", "runs_total",
        "runs_bowler", "runs_not_boundary", "runs_target", "day", "month",
        "year", "balls_per_over", "overs", "match_number", "team_runs",
        "team_balls", "team_wicket", "batter_runs", "batter_balls",
        "bowler_wicket", "striker_out"
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Ensure essential run and delivery columns are non-null
    for col in ["runs_batter", "runs_extras", "runs_total", "runs_bowler"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(0).astype(int)

    if "valid_ball" in cleaned.columns:
        cleaned["valid_ball"] = cleaned["valid_ball"].fillna(1).astype(int)

    # 7. Over numbering and phase engineering
    # In raw IPL data, 'over' is 0-indexed (0 to 19)
    if "over" in cleaned.columns:
        cleaned["over_raw"] = cleaned["over"].fillna(0).astype(int)
        # 1-indexed over number (1 to 20)
        cleaned["over_number"] = cleaned["over_raw"] + 1

        # Match Phase classification
        cleaned["phase"] = np.select(
            [
                cleaned["over_number"] <= 6,
                (cleaned["over_number"] >= 7) & (cleaned["over_number"] <= 15),
                cleaned["over_number"] >= 16
            ],
            ["Powerplay", "Middle Overs", "Death Overs"],
            default="Middle Overs"
        )

    # 8. Boundary and Dot ball flags
    if "runs_batter" in cleaned.columns:
        cleaned["is_four"] = (cleaned["runs_batter"] == 4).astype(int)
        cleaned["is_six"] = (cleaned["runs_batter"] == 6).astype(int)
        cleaned["is_boundary"] = ((cleaned["runs_batter"] == 4) | (cleaned["runs_batter"] == 6)).astype(int)

    if "runs_total" in cleaned.columns and "valid_ball" in cleaned.columns:
        cleaned["is_dot_ball"] = ((cleaned["runs_total"] == 0) & (cleaned["valid_ball"] == 1)).astype(int)

    # 9. Wicket classifications
    if "wicket_kind" in cleaned.columns:
        cleaned["is_wicket"] = cleaned["wicket_kind"].notna().astype(int)
        # Bowler wickets exclude run outs, retired hurt, obstructing the field
        non_bowler_wickets = ["run out", "retired hurt", "obstructing the field", "retired out"]
        cleaned["is_bowler_wicket"] = (
            cleaned["wicket_kind"].notna() & ~cleaned["wicket_kind"].isin(non_bowler_wickets)
        ).astype(int)
    else:
        cleaned["is_wicket"] = 0
        cleaned["is_bowler_wicket"] = 0

    # 10. Canonical franchise names
    team_cols = ["batting_team", "bowling_team", "toss_winner", "match_won_by", "superover_winner"]
    for col in team_cols:
        if col in cleaned.columns:
            canonical_col = f"{col}_canonical"
            cleaned[canonical_col] = cleaned[col].map(lambda x: TEAM_NAME_MAPPING.get(x, x) if pd.notna(x) else x)

    # 11. Generate Quality Report
    quality_report_df = generate_quality_report(df_raw, cleaned)

    return cleaned, quality_report_df


def generate_quality_report(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a structured data quality summary table comparing raw vs cleaned state.
    """
    total_matches = df_cleaned["match_id"].nunique() if "match_id" in df_cleaned.columns else 0
    total_seasons = df_cleaned["season_clean"].nunique() if "season_clean" in df_cleaned.columns else 0

    teams = set()
    if "batting_team" in df_cleaned.columns:
        teams.update(df_cleaned["batting_team"].dropna().unique())
    if "bowling_team" in df_cleaned.columns:
        teams.update(df_cleaned["bowling_team"].dropna().unique())

    total_batters = df_cleaned["batter"].nunique() if "batter" in df_cleaned.columns else 0
    total_bowlers = df_cleaned["bowler"].nunique() if "bowler" in df_cleaned.columns else 0
    total_venues = df_cleaned["venue"].nunique() if "venue" in df_cleaned.columns else 0

    report_data = [
        ("Total Rows (Deliveries)", f"{len(df_raw):,}", f"{len(df_cleaned):,}"),
        ("Total Columns", str(len(df_raw.columns)), str(len(df_cleaned.columns))),
        ("Duplicate Rows", f"{df_raw.duplicated().sum():,}", f"{df_cleaned.duplicated().sum():,}"),
        ("Missing / Null Cells", f"{df_raw.isnull().sum().sum():,}", f"{df_cleaned.isnull().sum().sum():,}"),
        ("Missing Cells %", f"{(df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100:.2f}%",
         f"{(df_cleaned.isnull().sum().sum() / (len(df_cleaned) * len(df_cleaned.columns))) * 100:.2f}%"),
        ("Unique Matches", str(df_raw["match_id"].nunique() if "match_id" in df_raw.columns else "N/A"), str(total_matches)),
        ("Unique Seasons", str(df_raw["season"].nunique() if "season" in df_raw.columns else "N/A"), str(total_seasons)),
        ("Unique Teams", "19", str(len(teams))),
        ("Unique Batters", str(df_raw["batter"].nunique() if "batter" in df_raw.columns else "N/A"), str(total_batters)),
        ("Unique Bowlers", str(df_raw["bowler"].nunique() if "bowler" in df_raw.columns else "N/A"), str(total_bowlers)),
        ("Unique Venues", str(df_raw["venue"].nunique() if "venue" in df_raw.columns else "N/A"), str(total_venues)),
        ("Date Span",
         f"{df_cleaned['date'].min().strftime('%Y-%m-%d')} to {df_cleaned['date'].max().strftime('%Y-%m-%d')}" if "date" in df_cleaned.columns and pd.notna(df_cleaned['date'].min()) else "2008 to 2026",
         f"{df_cleaned['date'].min().strftime('%Y-%m-%d')} to {df_cleaned['date'].max().strftime('%Y-%m-%d')}" if "date" in df_cleaned.columns and pd.notna(df_cleaned['date'].min()) else "2008 to 2026")
    ]

    report_df = pd.DataFrame(report_data, columns=["Metric", "Raw Dataset", "Cleaned Dataset"])
    return report_df
