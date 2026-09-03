# ============================================================
#              IPL DATA ANALYSIS & CLEANING PIPELINE
# ============================================================
# Dataset: IPL.csv (Ball-by-Ball IPL Data 2008 - 2026)
# Production-ready CLI runner & Analytical Data Generator
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.utils import find_dataset_path, TEAM_NAME_MAPPING, format_number
from src.data_cleaning import clean_ipl_data, generate_quality_report
from src.data_analysis import build_all_analytical_datasets

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
sns.set_theme(style="whitegrid")


def main():
    print("=" * 75)
    print("        IPL ANALYTICS - COMPLETE DATA ANALYSIS PIPELINE")
    print("=" * 75)

    # 1. Locate and Load Dataset
    print("\n[1/7] Locating IPL Dataset...")
    dataset_path = find_dataset_path()
    print(f" -> Found dataset at: {dataset_path}")

    df_raw = pd.read_csv(dataset_path, low_memory=False)
    print(f" -> Raw Dataset Shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    # 2. Clean Dataset
    print("\n[2/7] Executing Data Cleaning & Normalization Pipeline...")
    df_cleaned, quality_report = clean_ipl_data(df_raw)
    print(f" -> Cleaned Dataset Shape: {df_cleaned.shape[0]:,} rows, {df_cleaned.shape[1]} columns")
    print(f" -> Missing cells reduced to: {df_cleaned.isnull().sum().sum():,}")

    # 3. Build All Analytical Datasets
    print("\n[3/7] Generating Analytical Datasets...")
    analytics = build_all_analytical_datasets(df_cleaned)

    matches = analytics["matches"]
    innings = analytics["innings"]
    batting = analytics["batting"]
    bowling = analytics["bowling"]
    teams = analytics["teams"]
    venues = analytics["venues"]
    overs = analytics["overs"]
    phases = analytics["phases"]
    extras_type = analytics["extras_type"]
    extras_season = analytics["extras_season"]
    advanced = analytics["advanced"]

    print(f" -> Matches Dataset: {matches.shape[0]:,} records")
    print(f" -> Innings Dataset: {innings.shape[0]:,} records")
    print(f" -> Batting Dataset: {batting.shape[0]:,} batters")
    print(f" -> Bowling Dataset: {bowling.shape[0]:,} bowlers")
    print(f" -> Teams Dataset:   {teams.shape[0]} franchises")
    print(f" -> Venues Dataset:  {venues.shape[0]} stadiums")

    # 4. Print Summary Analytics
    print("\n" + "=" * 75)
    print("                         KEY ANALYTICAL HIGHLIGHTS")
    print("=" * 75)

    print("\n[+] FRANCHISE LEADERBOARD (Top 5 Teams by Wins):")
    print(teams[["team", "matches", "wins", "losses", "win_percentage", "average_score"]].head(5).to_string(index=False))

    print("\n[+] TOP 5 RUN SCORERS:")
    print(batting[["batter", "runs", "balls", "strike_rate", "batting_average", "fours", "sixes", "hundreds"]].head(5).to_string(index=False))

    print("\n[+] TOP 5 WICKET TAKERS:")
    print(bowling[["bowler", "wickets", "overs", "economy", "bowling_average", "dot_ball_pct"]].head(5).to_string(index=False))

    print("\n[+] MATCH PHASES RUN RATE & WICKETS:")
    print(phases[["phase", "runs", "overs", "run_rate", "wickets", "boundary_pct", "dot_ball_pct"]].to_string(index=False))

    # 5. Export Analytical Datasets
    print("\n[4/7] Exporting Datasets to CSV & Excel...")
    output_dirs = ["outputs", "IPL_Analysis_Output", "data"]
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)

    # Save cleaned delivery-level data
    df_cleaned.to_csv("data/cleaned_ipl.csv", index=False)
    df_cleaned.to_csv("outputs/cleaned_ipl.csv", index=False)
    df_cleaned.to_csv("IPL_Analysis_Output/IPL_Cleaned_Ball_By_Ball.csv", index=False)

    # Save specific analytical CSVs
    exports = {
        "match_data.csv": matches,
        "innings_data.csv": innings,
        "batting_data.csv": batting,
        "bowling_data.csv": bowling,
        "team_data.csv": teams,
        "venue_data.csv": venues,
        "over_phase_data.csv": overs,
        "extras_data.csv": extras_type,
        "data_quality_report.csv": quality_report
    }

    for filename, dataset in exports.items():
        dataset.to_csv(f"outputs/{filename}", index=False)
        dataset.to_csv(f"IPL_Analysis_Output/IPL_{filename.replace('.csv', '').title().replace('_', '_')}.csv", index=False)

    # 6. Export Multi-Sheet Comprehensive Excel Workbook
    print("\n[5/7] Compiling Comprehensive Excel Report (outputs/IPL_Complete_Analysis.xlsx)...")
    excel_path = "outputs/IPL_Complete_Analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        quality_report.to_excel(writer, sheet_name="Data_Quality", index=False)
        matches.to_excel(writer, sheet_name="Matches", index=False)
        innings.to_excel(writer, sheet_name="Innings", index=False)
        batting.head(100).to_excel(writer, sheet_name="Top_Batters", index=False)
        bowling.head(100).to_excel(writer, sheet_name="Top_Bowlers", index=False)
        teams.to_excel(writer, sheet_name="Team_Performance", index=False)
        venues.to_excel(writer, sheet_name="Venue_Analysis", index=False)
        phases.to_excel(writer, sheet_name="Phase_Analysis", index=False)
        overs.to_excel(writer, sheet_name="Over_Analysis", index=False)
        extras_type.to_excel(writer, sheet_name="Extras_Breakdown", index=False)
        advanced["correlation_matrix"].to_excel(writer, sheet_name="Correlation_Matrix")

    # Mirror to IPL_Analysis_Output for compatibility
    try:
        import shutil
        shutil.copyfile(excel_path, "IPL_Analysis_Output/IPL_Complete_Analysis.xlsx")
    except Exception:
        pass

    print(f" -> Multi-sheet Excel report saved at: {excel_path}")

    # 7. Final KPI Summary
    print("\n" + "=" * 75)
    print("                         IPL PIPELINE EXECUTION COMPLETE")
    print("=" * 75)
    print(f"Total Matches Analyzed : {matches['match_id'].nunique():,}")
    print(f"Total Deliveries       : {len(df_cleaned):,}")
    print(f"Total Runs Scored      : {df_cleaned['runs_total'].sum():,}")
    print(f"Total Wickets Taken    : {df_cleaned['is_wicket'].sum():,}")
    print(f"Total Fours            : {df_cleaned['is_four'].sum():,}")
    print(f"Total Sixes            : {df_cleaned['is_six'].sum():,}")
    print(f"Highest Team Score     : {innings['total_runs'].max()} runs")
    print(f"Top Run Scorer         : {batting.iloc[0]['batter']} ({batting.iloc[0]['runs']:,} runs)")
    print(f"Top Wicket Taker       : {bowling.iloc[0]['bowler']} ({bowling.iloc[0]['wickets']:,} wickets)")
    print("=" * 75)
    print("\nTo launch the interactive 3D Glassmorphism Dashboard, run:")
    print("    streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()