# utils/audit/list_features.py
import joblib
from pathlib import Path
import argparse
import yaml

def list_model_features(problem: str, model_type: str):
    """
    Lists the feature columns for the most recently trained model
    of a specific type and problem.
    """
    model_dir = Path('output/models')
    
    # Find the latest model file
    model_files = sorted(model_dir.glob(f"{model_type.lower()}_{problem.lower()}_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_files:
        print(f"No {model_type} models found for target '{problem}' in {model_dir}")
        return

    latest_model_path = model_files[0]
    
    # Derive the corresponding feature columns file path
    name_parts = latest_model_path.stem.split('_')
    timestamp = f"{name_parts[-2]}_{name_parts[-1]}"
    feature_col_path = model_dir / f"feature_columns_{problem.lower()}_{timestamp}.pkl"

    if not feature_col_path.exists():
        print(f"Could not find feature column file for model: {latest_model_path.name}")
        return

    features = joblib.load(feature_col_path)
    
    # Organize features by prefix/group for readable output
    col_groups = {
        'id': ['game_id', 'game_date', 'player_id', 'team', 'opponent'],
        'rolling': [c for c in features if c.startswith('rw_') or c.startswith('ps_')],
        'odds': [c for c in features if any(p in c for p in ['moneyline_', 'spread_', 'total_', 'implied_prob_', 'odds_'])],
        'injury': [c for c in features if c.startswith('injury_') or c.startswith('practice_')],
        'team_ctx': [c for c in features if c.startswith('team_ctx_') or c.startswith('opp_ctx_')],
        'weather': [c for c in features if c.startswith('weather_')],
        'temporal': [c for c in features if 'days_' in c or 'week' in c or 'season' in c]
    }

    print(f"--- Feature Columns for Model: {latest_model_path.name} ---")
    
    categorized_features = set()
    for group_name, group_cols in col_groups.items():
        # Filter for features that are actually in the list
        current_group_features = [f for f in group_cols if f in features]
        if current_group_features:
            print(f"\n# --- {group_name.upper()} Features ---")
            for feature in sorted(current_group_features):
                print(f"- {feature}")
            categorized_features.update(current_group_features)
            
    # Print any uncategorized features
    uncategorized = sorted([f for f in features if f not in categorized_features])
    if uncategorized:
        print("\n# --- OTHER FEATURES ---")
        for feature in uncategorized:
            print(f"- {feature}")

def main():
    parser = argparse.ArgumentParser(description="List feature columns for the latest trained model.")
    parser.add_argument('--problem', type=str, default='GETS_HIT', help="The prediction problem (e.g., 'GETS_HIT').")
    parser.add_argument('--model_type', type=str, default='LightGBM', help="The model type (e.g., 'LightGBM').")
    args = parser.parse_args()
    
    list_model_features(args.problem, args.model_type)

if __name__ == "__main__":
    main() 