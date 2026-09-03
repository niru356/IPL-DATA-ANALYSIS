"""
Visualization Suite for IPL Data Analysis.
Generates premium, interactive Plotly visualizations with dark luxury 3D glassmorphic styling,
custom hover tooltips, and official IPL team color themes.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict
from .utils import get_team_color, TEAM_COLORS, ACCENT_COLORS

# Theme Configuration
PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(18, 24, 41, 0.4)",
    font=dict(family="Outfit, Inter, sans-serif", color="#E2E8F0", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.12)"
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.12)"
    ),
    hoverlabel=dict(
        bgcolor="rgba(15, 23, 42, 0.95)",
        bordercolor="rgba(99, 102, 241, 0.6)",
        font=dict(family="Outfit, Inter, sans-serif", color="#F8FAFC", size=13)
    ),
    legend=dict(
        bgcolor="rgba(22, 27, 46, 0.6)",
        bordercolor="rgba(255, 255, 255, 0.1)",
        borderwidth=1
    )
)


def apply_glassmorphism_layout(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Applies premium glassmorphism styling to any Plotly figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color="#F8FAFC"),
            x=0.02,
            y=0.96
        ),
        height=height
    )
    return fig


def plot_matches_and_runs_by_season(matches_df: pd.DataFrame, innings_df: pd.DataFrame) -> go.Figure:
    """Combo chart showing matches played and total runs scored per season."""
    season_matches = matches_df.groupby("season_clean")["match_id"].nunique().reset_index(name="matches")
    season_runs = innings_df.groupby("season_clean")["total_runs"].sum().reset_index(name="runs")
    season_avg = innings_df.groupby("season_clean")["total_runs"].mean().reset_index(name="avg_score")

    season_data = season_matches.merge(season_runs, on="season_clean").merge(season_avg, on="season_clean")
    season_data = season_data.sort_values("season_clean")

    fig = go.Figure()

    # Bar: Matches
    fig.add_trace(go.Bar(
        x=season_data["season_clean"],
        y=season_data["matches"],
        name="Matches",
        marker=dict(
            color="#6366F1",
            line=dict(color="#818CF8", width=1.5),
            opacity=0.85
        ),
        yaxis="y"
    ))

    # Line: Total Runs (Secondary axis)
    fig.add_trace(go.Scatter(
        x=season_data["season_clean"],
        y=season_data["runs"],
        name="Total Runs",
        mode="lines+markers",
        line=dict(color="#06B6D4", width=3, shape="spline"),
        marker=dict(size=8, color="#22D3EE", symbol="diamond"),
        yaxis="y2"
    ))

    fig.update_layout(
        yaxis=dict(title="Number of Matches", side="left"),
        yaxis2=dict(
            title="Total Runs",
            side="right",
            overlaying="y",
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_glassmorphism_layout(fig, "IPL Season Growth: Matches vs Total Runs Scored", 460)


def plot_top_run_scorers(batting_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top IPL run scorers with strike rate indicators."""
    top_batters = batting_df.head(top_n).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=top_batters["runs"],
        y=top_batters["batter"],
        orientation="h",
        text=top_batters["runs"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        customdata=np.stack((
            top_batters["strike_rate"],
            top_batters["batting_average"],
            top_batters["fours"],
            top_batters["sixes"],
            top_batters["fifties"],
            top_batters["hundreds"]
        ), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Runs: <b>%{x:,}</b><br>" +
            "Strike Rate: <b>%{customdata[0]:.2f}</b><br>" +
            "Average: <b>%{customdata[1]:.2f}</b><br>" +
            "Fours: %{customdata[2]} | Sixes: %{customdata[3]}<br>" +
            "50s: %{customdata[4]} | 100s: %{customdata[5]}<extra></extra>"
        ),
        marker=dict(
            color=top_batters["strike_rate"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Strike Rate", thickness=15, len=0.8)
        )
    ))

    fig.update_layout(
        xaxis=dict(title="Total Career Runs"),
        yaxis=dict(title="")
    )

    return apply_glassmorphism_layout(fig, f"Top {top_n} IPL Run Scorers (Color = Strike Rate)", 500)


def plot_strike_rate_vs_average(batting_df: pd.DataFrame, min_runs: int = 500) -> go.Figure:
    """Interactive quadrant bubble chart of Batting Average vs Strike Rate."""
    filtered = batting_df[batting_df["runs"] >= min_runs].copy()

    avg_mean = filtered["batting_average"].median()
    sr_mean = filtered["strike_rate"].median()

    fig = px.scatter(
        filtered,
        x="batting_average",
        y="strike_rate",
        size="runs",
        color="sixes",
        hover_name="batter",
        hover_data={"runs": ":,", "balls": True, "fours": True, "sixes": True, "hundreds": True},
        color_continuous_scale="Turbo",
        labels={"batting_average": "Batting Average", "strike_rate": "Strike Rate", "sixes": "Total Sixes"}
    )

    # Quadrant crosshairs
    fig.add_vline(x=avg_mean, line_dash="dash", line_color="rgba(255, 255, 255, 0.25)", annotation_text="Median Avg")
    fig.add_hline(y=sr_mean, line_dash="dash", line_color="rgba(255, 255, 255, 0.25)", annotation_text="Median SR")

    return apply_glassmorphism_layout(fig, f"Batting Power Matrix: Average vs Strike Rate (Min {min_runs} Runs)", 500)


def plot_top_wicket_takers(bowling_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top IPL wicket takers with economy rate scale."""
    top_bowlers = bowling_df.head(top_n).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=top_bowlers["wickets"],
        y=top_bowlers["bowler"],
        orientation="h",
        text=top_bowlers["wickets"],
        textposition="outside",
        customdata=np.stack((
            top_bowlers["economy"],
            top_bowlers["bowling_average"],
            top_bowlers["bowling_strike_rate"],
            top_bowlers["overs"],
            top_bowlers["dot_ball_pct"]
        ), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Wickets: <b>%{x}</b><br>" +
            "Economy: <b>%{customdata[0]:.2f}</b><br>" +
            "Bowling Avg: <b>%{customdata[1]:.2f}</b><br>" +
            "Strike Rate: <b>%{customdata[2]:.2f}</b><br>" +
            "Overs: %{customdata[3]} | Dot Ball %: %{customdata[4]}%<extra></extra>"
        ),
        marker=dict(
            color=top_bowlers["economy"],
            colorscale="Plasma_r",
            showscale=True,
            colorbar=dict(title="Economy Rate", thickness=15, len=0.8)
        )
    ))

    fig.update_layout(xaxis=dict(title="Total Career Wickets"), yaxis=dict(title=""))
    return apply_glassmorphism_layout(fig, f"Top {top_n} IPL Wicket Takers (Color = Economy)", 500)


def plot_economy_vs_strike_rate(bowling_df: pd.DataFrame, min_overs: int = 50) -> go.Figure:
    """Scatter chart of Economy Rate vs Bowling Strike Rate."""
    filtered = bowling_df[bowling_df["overs"] >= min_overs].copy()

    fig = px.scatter(
        filtered,
        x="economy",
        y="bowling_strike_rate",
        size="wickets",
        color="dot_ball_pct",
        hover_name="bowler",
        hover_data={"wickets": True, "overs": True, "runs_conceded": True},
        color_continuous_scale="Cividis",
        labels={"economy": "Economy Rate", "bowling_strike_rate": "Bowling Strike Rate (Balls/Wicket)", "dot_ball_pct": "Dot Ball %"}
    )

    return apply_glassmorphism_layout(fig, f"Bowling Efficiency Matrix: Economy vs Strike Rate (Min {min_overs} Overs)", 500)


def plot_team_win_percentage(team_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing all IPL teams by win percentage and total wins."""
    sorted_teams = team_df.sort_values("win_percentage", ascending=True)

    colors = [get_team_color(t) for t in sorted_teams["team"]]

    fig = go.Figure(go.Bar(
        x=sorted_teams["win_percentage"],
        y=sorted_teams["team"],
        orientation="h",
        text=sorted_teams["win_percentage"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        customdata=np.stack((sorted_teams["matches"], sorted_teams["wins"], sorted_teams["losses"]), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Win Rate: <b>%{x:.2f}%</b><br>" +
            "Matches: %{customdata[0]} | Wins: <b>%{customdata[1]}</b> | Losses: %{customdata[2]}<extra></extra>"
        ),
        marker=dict(color=colors, line=dict(color="rgba(255, 255, 255, 0.2)", width=1))
    ))

    fig.update_layout(xaxis=dict(title="Win Percentage (%)"), yaxis=dict(title=""))
    return apply_glassmorphism_layout(fig, "IPL Franchise Leaderboard: Win Percentage", 520)


def plot_batting_first_vs_chasing(matches_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing Batting First vs Chasing victory distribution."""
    counts = matches_df["winner_type"].value_counts().reset_index()
    counts.columns = ["strategy", "wins"]

    color_map = {
        "Chasing": "#10B981",
        "Batting First": "#3B82F6",
        "Tie / No Result": "#64748B"
    }

    fig = go.Figure(go.Pie(
        labels=counts["strategy"],
        values=counts["wins"],
        hole=0.55,
        marker=dict(colors=[color_map.get(s, "#8B5CF6") for s in counts["strategy"]]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Wins: <b>%{value}</b> (%{percent})<extra></extra>"
    ))

    return apply_glassmorphism_layout(fig, "Batting First vs Chasing Success Ratio", 400)


def plot_toss_impact(matches_df: pd.DataFrame) -> go.Figure:
    """Donut chart illustrating the match win conversion rate of toss winners."""
    toss_won_matches = matches_df["toss_winner_won"].value_counts().reset_index()
    toss_won_matches.columns = ["won_match", "count"]
    toss_won_matches["label"] = toss_won_matches["won_match"].map({True: "Toss Winner Won Match", False: "Toss Winner Lost Match"})

    fig = go.Figure(go.Pie(
        labels=toss_won_matches["label"],
        values=toss_won_matches["count"],
        hole=0.55,
        marker=dict(colors=["#06B6D4", "#F43F5E"]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Matches: <b>%{value}</b> (%{percent})<extra></extra>"
    ))

    return apply_glassmorphism_layout(fig, "Toss Advantage: Win Conversion Rate", 400)


def plot_venue_insights(venue_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Grouped bar chart showing average 1st and 2nd innings score across stadiums."""
    top_venues = venue_df.head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_venues["venue"],
        y=top_venues["avg_1st_innings_score"],
        name="1st Innings Avg",
        marker_color="#3B82F6"
    ))
    fig.add_trace(go.Bar(
        x=top_venues["venue"],
        y=top_venues["avg_2nd_innings_score"],
        name="2nd Innings Avg",
        marker_color="#10B981"
    ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(title="", tickangle=-35),
        yaxis=dict(title="Average Innings Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_glassmorphism_layout(fig, f"Top {top_n} Stadiums: 1st vs 2nd Innings Scoring Trends", 480)


def plot_over_run_rate_and_wickets(over_df: pd.DataFrame) -> go.Figure:
    """Combined chart of runs scored and wickets taken over-by-over (Overs 1 to 20)."""
    fig = go.Figure()

    # Bar: Runs per over
    fig.add_trace(go.Bar(
        x=over_df["over_number"],
        y=over_df["runs"],
        name="Total Runs Scored",
        marker=dict(
            color=over_df["run_rate"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Run Rate", thickness=12, len=0.8)
        ),
        yaxis="y"
    ))

    # Line: Wickets per over
    fig.add_trace(go.Scatter(
        x=over_df["over_number"],
        y=over_df["wickets"],
        name="Total Wickets Lost",
        mode="lines+markers",
        line=dict(color="#F43F5E", width=3),
        marker=dict(size=8, color="#FB7185"),
        yaxis="y2"
    ))

    fig.update_layout(
        xaxis=dict(title="Over Number (1 to 20)", dtick=1),
        yaxis=dict(title="Total Runs", side="left"),
        yaxis2=dict(title="Total Wickets Lost", side="right", overlaying="y", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_glassmorphism_layout(fig, "Over Progression (1-20): Runs & Wickets Dynamic", 460)


def plot_phase_analysis(phase_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Powerplay, Middle, and Death phases."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=phase_df["phase"],
        y=phase_df["run_rate"],
        name="Run Rate (RPO)",
        marker_color="#8B5CF6",
        text=phase_df["run_rate"],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=phase_df["phase"],
        y=phase_df["boundary_pct"],
        name="Boundary %",
        marker_color="#F59E0B",
        text=phase_df["boundary_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=phase_df["phase"],
        y=phase_df["dot_ball_pct"],
        name="Dot Ball %",
        marker_color="#06B6D4",
        text=phase_df["dot_ball_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Metric Value"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_glassmorphism_layout(fig, "Match Phases: Powerplay vs Middle vs Death Overs", 450)


def plot_extras_breakdown(extras_type_df: pd.DataFrame) -> go.Figure:
    """Donut chart of extras distribution (wides, noballs, byes, legbyes)."""
    fig = go.Figure(go.Pie(
        labels=extras_type_df["extra_type"].str.upper(),
        values=extras_type_df["extra_runs"],
        hole=0.55,
        marker=dict(colors=["#F59E0B", "#EF4444", "#3B82F6", "#10B981", "#8B5CF6"]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Extra Runs: <b>%{value:,}</b> (%{percent})<extra></extra>"
    ))

    return apply_glassmorphism_layout(fig, "IPL Extras Breakdown by Delivery Type", 400)


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Plotly interactive heatmap showing numerical correlation matrix."""
    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="Viridis",
        zmin=-1,
        zmax=1,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        hoverongaps=False
    ))

    return apply_glassmorphism_layout(fig, "IPL Numerical Variables Correlation Matrix", 480)


def plot_score_distribution(innings_df: pd.DataFrame) -> go.Figure:
    """Histogram with box plot for IPL team innings scores."""
    fig = px.histogram(
        innings_df,
        x="total_runs",
        nbins=40,
        marginal="box",
        color_discrete_sequence=["#6366F1"],
        labels={"total_runs": "Total Innings Score"}
    )

    return apply_glassmorphism_layout(fig, "Distribution & IQR Outliers of IPL Innings Scores", 480)


def plot_batter_comparison_radar(batting_df: pd.DataFrame, batter1: str, batter2: str) -> go.Figure:
    """Radar chart comparing two batters across normalized key cricket metrics."""
    b1 = batting_df[batting_df["batter"] == batter1]
    b2 = batting_df[batting_df["batter"] == batter2]

    if b1.empty or b2.empty:
        return go.Figure()

    r1 = b1.iloc[0]
    r2 = b2.iloc[0]

    # Metrics with normalization targets
    categories = ["Strike Rate", "Batting Average", "Boundary %", "Innings / 50", "Dot Ball Resistance"]

    # Calculate normalized scores (0 to 100 scale)
    b1_vals = [
        min(100, (r1["strike_rate"] / 180) * 100),
        min(100, (r1["batting_average"] / 50) * 100),
        min(100, (r1["boundary_pct"] / 30) * 100),
        min(100, ((r1["fifties"] + r1["hundreds"]) / max(1, r1["innings"])) * 200),
        max(0, 100 - r1["dot_ball_pct"])
    ]

    b2_vals = [
        min(100, (r2["strike_rate"] / 180) * 100),
        min(100, (r2["batting_average"] / 50) * 100),
        min(100, (r2["boundary_pct"] / 30) * 100),
        min(100, ((r2["fifties"] + r2["hundreds"]) / max(1, r2["innings"])) * 200),
        max(0, 100 - r2["dot_ball_pct"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=b1_vals + [b1_vals[0]], theta=categories + [categories[0]], fill="toself", name=batter1, line_color="#06B6D4"))
    fig.add_trace(go.Scatterpolar(r=b2_vals + [b2_vals[0]], theta=categories + [categories[0]], fill="toself", name=batter2, line_color="#F43F5E"))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_glassmorphism_layout(fig, f"Head-to-Head Batter Radar: {batter1} vs {batter2}", 460)


def plot_bowler_comparison_radar(bowling_df: pd.DataFrame, bowler1: str, bowler2: str) -> go.Figure:
    """Radar chart comparing two bowlers across normalized key cricket metrics."""
    b1 = bowling_df[bowling_df["bowler"] == bowler1]
    b2 = bowling_df[bowling_df["bowler"] == bowler2]

    if b1.empty or b2.empty:
        return go.Figure()

    r1 = b1.iloc[0]
    r2 = b2.iloc[0]

    categories = ["Economy Control", "Strike Rate", "Average", "Dot Ball %", "Hauls Impact"]

    b1_vals = [
        max(0, (12 - (r1["economy"] if pd.notna(r1["economy"]) else 10)) * 15),
        min(100, (30 / max(1, r1["bowling_strike_rate"] if pd.notna(r1["bowling_strike_rate"]) else 30)) * 60),
        min(100, (35 / max(1, r1["bowling_average"] if pd.notna(r1["bowling_average"]) else 35)) * 60),
        min(100, (r1["dot_ball_pct"] / 60) * 100),
        min(100, (r1["four_wickets"] + r1["five_wickets"] * 2) * 20)
    ]

    b2_vals = [
        max(0, (12 - (r2["economy"] if pd.notna(r2["economy"]) else 10)) * 15),
        min(100, (30 / max(1, r2["bowling_strike_rate"] if pd.notna(r2["bowling_strike_rate"]) else 30)) * 60),
        min(100, (35 / max(1, r2["bowling_average"] if pd.notna(r2["bowling_average"]) else 35)) * 60),
        min(100, (r2["dot_ball_pct"] / 60) * 100),
        min(100, (r2["four_wickets"] + r2["five_wickets"] * 2) * 20)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=b1_vals + [b1_vals[0]], theta=categories + [categories[0]], fill="toself", name=bowler1, line_color="#8B5CF6"))
    fig.add_trace(go.Scatterpolar(r=b2_vals + [b2_vals[0]], theta=categories + [categories[0]], fill="toself", name=bowler2, line_color="#F59E0B"))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_glassmorphism_layout(fig, f"Head-to-Head Bowler Radar: {bowler1} vs {bowler2}", 460)
