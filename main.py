# main.py
"""NFL Player Prediction Pipeline - Main Orchestrator

Runs the complete pipeline from data collection to prediction:
1. Collect: Fetch NFL play-by-play data
2. Clean: Normalize and add derived stats
3. Feature: Build rolling window features
4. Train: Train models on player targets (anytime_td, yards, etc.)
5. Predict: Generate predictions for upcoming games
"""

import logging
import sys
from datetime import datetime, date
from zoneinfo import ZoneInfo
from pathlib import Path

from pipeline.collect import collect_nfl_data
from pipeline.clean import clean
from pipeline.feature import build_feature_matrix, refresh_context_histories
from pipeline.train import train
from pipeline.predict import main as run_predict
from utils.general.config import load_config, get_pipeline_config

# Load config using centralized loader
config = load_config()


def _get_debug_flags() -> dict:
    """Get debug flags from config, with fallback defaults."""
    pipeline_cfg = config.get('pipeline', {})
    debug_cfg = pipeline_cfg.get('debug', {})
    return {
        'skip_collection': debug_cfg.get('skip_data_collection', False),
        'skip_cleaning': debug_cfg.get('skip_data_cleaning', False),
        'skip_features': debug_cfg.get('skip_feature_engineering', False),
        'skip_training': debug_cfg.get('skip_model_training', False),
        'skip_prediction': debug_cfg.get('skip_prediction', True),
    }


# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

file_handler = logging.FileHandler('logs/nfl_pipeline.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)


# ========== PIPELINE FUNCTIONS ========== #
def main():
    """Main entry point - runs full pipeline."""
    logger.info("=" * 80)
    logger.info("NFL PLAYER PREDICTION PIPELINE")
    logger.info("=" * 80)
    
    # Get debug flags from config
    debug = _get_debug_flags()
    
    # Get date range from config
    pipeline_cfg = config.get('pipeline', {})
    start_date_str = pipeline_cfg.get('start_date', '2024-09-05')
    end_date_str = pipeline_cfg.get('end_date')
    
    start_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str) if end_date_str else date.today()
    
    logger.info(f"Pipeline date range: {start_date} → {end_date}")
    
    # Step 1: Data Collection (incremental by default)
    if not debug['skip_collection']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA COLLECTION (INCREMENTAL)")
        logger.info("=" * 80)
        collect_nfl_data(start_date=start_date, end_date=end_date, force_full_refresh=False)
        logger.info("✅ Data collection complete")
    else:
        logger.info("\nStep 1: Data Collection - SKIPPED")
    
    # Step 2: Data Cleaning
    if not debug['skip_cleaning']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA CLEANING")
        logger.info("=" * 80)
        clean()
        logger.info("✅ Data cleaning complete")
    else:
        logger.info("\nStep 2: Data Cleaning - SKIPPED")
    
    # Step 3: Feature Engineering
    if not debug['skip_features']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        build_feature_matrix(
            start_date=start_date,
            end_date=end_date,
            build_rolling=True,
        )
        logger.info("✅ Feature engineering complete")
    else:
        logger.info("\nStep 3: Feature Engineering - SKIPPED")
    
    # Step 4: Model Training
    if not debug['skip_training']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 80)
        train()
        logger.info("✅ Model training complete")
    else:
        logger.info("\nStep 4: Model Training - SKIPPED")
    
    # Step 5: Prediction (only if current NFL season)
    if not debug['skip_prediction']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: PREDICTION")
        logger.info("=" * 80)
        today = datetime.now(ZoneInfo("America/New_York")).date()
        logger.info(f"Generating predictions for: {today}")
        try:
            run_predict()
            logger.info("✅ Predictions generated")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
    else:
        logger.info("\nStep 5: Prediction - SKIPPED")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Get pipeline config for CLI commands
    pipeline_cfg = config.get('pipeline', {})
    default_start = pipeline_cfg.get('start_date', '2024-09-05')
    
    # Individual step execution
    steps = {
        "collect": lambda: collect_nfl_data(
            start_date=date.fromisoformat(default_start),
            end_date=date.today(),
            force_full_refresh=False
        ),
        "collect-full": lambda: collect_nfl_data(
            start_date=date.fromisoformat(default_start),
            end_date=date.today(),
            force_full_refresh=True
        ),
        "clean": lambda: clean(
            start_date=date.fromisoformat(default_start),
            end_date=date.today()
        ),
        "feature": lambda: build_feature_matrix(
            start_date=date.fromisoformat(default_start),
            end_date=date.today(),
            build_rolling=True,
        ),
        "train": train,
        "predict": run_predict,
        "context-refresh": refresh_context_histories,
    }

    if len(sys.argv) > 1:
        step_arg = sys.argv[1].lower()
        if step_arg in steps:
            logger.info(f"Executing pipeline step: {step_arg}")
            steps[step_arg]()
            logger.info(f"✅ Step '{step_arg}' completed.")
        else:
            logger.error(f"Invalid argument '{step_arg}'. Valid steps are: {list(steps.keys())}")
            sys.exit(1)
    else:
        main()
