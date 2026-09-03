"""
Utility functions and configuration constants for IPL Data Analysis & Dashboard.
"""

import os
from pathlib import Path
from typing import Optional, Dict

# Team name standardization / canonical mapping
TEAM_NAME_MAPPING: Dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

# Reverse mapping if needed or list of active/historic franchises
CANONICAL_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
    "Deccan Chargers",
    "Gujarat Lions",
    "Kochi Tuskers Kerala",
    "Pune Warriors",
    "Rising Pune Supergiant"
]

# Team official color themes (Hex)
TEAM_COLORS: Dict[str, str] = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Royal Challengers Bangalore": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Rajasthan Royals": "#EA1A85",
    "Delhi Capitals": "#0078BC",
    "Delhi Daredevils": "#17479E",
    "Punjab Kings": "#DD1F2D",
    "Kings XI Punjab": "#D71920",
    "Sunrisers Hyderabad": "#F74B00",
    "Gujarat Titans": "#1B2133",
    "Lucknow Super Giants": "#0057E7",
    "Deccan Chargers": "#D9E3E8",
    "Gujarat Lions": "#E04F16",
    "Pune Warriors": "#2F9BE3",
    "Rising Pune Supergiant": "#D11D53",
    "Rising Pune Supergiants": "#D11D53",
    "Kochi Tuskers Kerala": "#6C2D58"
}

# Neutral / Fallback theme colors
ACCENT_COLORS = {
    "cyan": "#06B6D4",
    "purple": "#8B5CF6",
    "amber": "#F59E0B",
    "emerald": "#10B981",
    "rose": "#F43F5E",
    "blue": "#3B82F6",
    "indigo": "#6366F1",
    "dark_bg": "#0B0F19",
    "card_bg": "rgba(22, 27, 46, 0.75)",
    "border_color": "rgba(255, 255, 255, 0.08)"
}


def find_dataset_path(preferred_name: Optional[str] = None) -> str:
    """
    Locates the IPL CSV dataset dynamically across standard workspace directories.
    """
    if preferred_name and os.path.exists(preferred_name):
        return os.path.abspath(preferred_name)

    candidates = [
        "data/IPL.csv",
        "data/IPL(1).csv",
        "data/cleaned_ipl.csv",
        "IPL.csv",
        "IPL(1).csv",
        "../data/IPL.csv",
        "../IPL.csv",
        os.path.join(os.getcwd(), "IPL.csv"),
        os.path.join(os.getcwd(), "data", "IPL.csv"),
    ]

    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return os.path.abspath(path)

    # Search current directory recursively for matching CSV
    current_dir = Path(os.getcwd())
    for file in current_dir.rglob("*.csv"):
        if "ipl" in file.name.lower():
            return str(file.resolve())

    raise FileNotFoundError(
        "Could not automatically locate the IPL CSV dataset. Please place 'IPL.csv' in the project directory."
    )


def get_team_color(team_name: str, fallback: str = "#6366F1") -> str:
    """Returns the hex color code for an IPL team."""
    if not isinstance(team_name, str):
        return fallback
    return TEAM_COLORS.get(team_name.strip(), fallback)


def format_number(val: float, decimals: int = 2) -> str:
    """Formats numeric values into human-readable strings."""
    if val is None or val != val:  # NaN check
        return "N/A"
    if isinstance(val, int) or val.is_integer():
        return f"{int(val):,}"
    return f"{val:,.{decimals}f}"


def format_percentage(val: float, decimals: int = 1) -> str:
    """Formats percentage float to string with % sign."""
    if val is None or val != val:
        return "0.0%"
    return f"{val:.{decimals}f}%"
