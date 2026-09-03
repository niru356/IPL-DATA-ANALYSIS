"""
IPL Analysis & Analytics Suite
Package initialization
"""

from .utils import find_dataset_path, TEAM_NAME_MAPPING, TEAM_COLORS, format_number
from .data_cleaning import clean_ipl_data, generate_quality_report, inspect_raw_data
from .data_analysis import (
    create_match_dataset,
    create_innings_dataset,
    create_batting_dataset,
    create_bowling_dataset,
    create_team_dataset,
    create_venue_dataset,
    create_over_phase_dataset,
    create_extras_dataset,
    create_toss_dataset,
    compute_advanced_stats,
    build_all_analytical_datasets
)
