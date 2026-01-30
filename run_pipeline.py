"""
Run Complete Pipeline Script
Executes the full data science workflow from data loading to model deployment
"""

import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from mlops_pipeline import MLOpsPipeline

# Setup logger
logger = setup_logger("run_pipeline", log_dir="logs", level="INFO")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the complete ML pipeline for stock market prediction"
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    
    parser.add_argument(
        '--auto-deploy',
        action='store_true',
        default=True,
        help='Automatically deploy the best model'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['linear_regression', 'random_forest', 'xgboost', 'lightgbm'],
        help='Models to train (space-separated)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution"""
    
    # Parse arguments
    args = parse_args()
    
    logger.info("=" * 100)
    logger.info("STARTING COMPLETE ML PIPELINE")
    logger.info("=" * 100)
    
    try:
        # Initialize pipeline
        pipeline = MLOpsPipeline(config_path=args.config)
        
        if not args.skip_training:
            # Run full pipeline
            logger.info("\n📊 Running full pipeline with the following configuration:")
            logger.info(f"   Models to train: {args.models}")
            logger.info(f"   Auto-deploy: {args.auto_deploy}")
            
            pipeline.run_full_pipeline(auto_deploy=args.auto_deploy)
        else:
            logger.info("\n⏭️  Skipping training (--skip-training flag set)")
        
        logger.info("\n" + "=" * 100)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)
        
        logger.info("\n📝 Next steps:")
        logger.info("   1. Check the logs/ directory for detailed execution logs")
        logger.info("   2. Review model performance in logs/evaluation/model_comparison.csv")
        logger.info("   3. The best model has been saved to the models/ directory")
        logger.info("   4. Start the Flask API: python flask_app.py")
        logger.info("   5. Visit http://localhost:5000 to test the API")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
