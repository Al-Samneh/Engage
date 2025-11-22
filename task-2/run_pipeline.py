"""
Single entry point that stitches together Task 2 steps:
load data → engineer features → add embeddings → train model.
"""
import argparse

import seaborn as sns

from config import TASK_DIR
from src.data_utils import load_data
from src.embeddings import add_embeddings
from src.features import engineer_merged_features
from src.model import train_xgboost

sns.set_style("whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="Restaurant rating prediction pipeline")
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Force regeneration of review embeddings (ignore cache)",
    )
    return parser.parse_args()


def main(args):
    """
    Orchestrate the training run using the helper modules.
    """
    df_restaurants, df_reviews, df_users, df_trends = load_data()
    df_merged = engineer_merged_features(df_restaurants, df_reviews, df_users, df_trends)
    df_final, embedding_cols = add_embeddings(
        df_merged, force_recompute=args.recompute_embeddings
    )
    train_xgboost(df_final, embedding_cols, output_dir=TASK_DIR)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
