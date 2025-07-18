import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

def main():
    parser = argparse.ArgumentParser(description='Visualize Word2Vec embeddings')
    parser.add_argument('--input', default='evaluation_results.csv')
    args = parser.parse_args()
    # Load the evaluation results
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} word pairs")
    print(df.head())

    # Clean data - remove N/A values
    df_clean = df[df['model_score'] != 'N/A'].copy()
    df_clean['model_score'] = pd.to_numeric(df_clean['model_score'])
    df_clean['human_score'] = pd.to_numeric(df_clean['human_score'])

    print(f"\nValid pairs: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")

    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(df_clean['human_score'], df_clean['model_score'])
    pearson_corr, pearson_p = pearsonr(df_clean['human_score'], df_clean['model_score'])

    print(f"\nCorrelations:")
    print(f"Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"Pearson:  {pearson_corr:.4f} (p={pearson_p:.4f})")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Word2Vec Model Evaluation on SimLex-999', fontsize=16)

    # 1. Scatter plot: Human vs Model scores
    axes[0, 0].scatter(df_clean['human_score'], df_clean['model_score'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Human Similarity Scores')
    axes[0, 0].set_ylabel('Model Similarity Scores')
    axes[0, 0].set_title(f'Human vs Model Scores\nSpearman ρ = {spearman_corr:.3f}')
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(df_clean['human_score'], df_clean['model_score'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df_clean['human_score'], p(df_clean['human_score']), "r--", alpha=0.8)

    # 2. Distribution of human scores
    axes[0, 1].hist(df_clean['human_score'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Human Similarity Scores')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Human Scores')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribution of model scores
    axes[1, 0].hist(df_clean['model_score'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Model Similarity Scores')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Model Scores')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals plot
    residuals = df_clean['model_score'] - df_clean['human_score']
    axes[1, 1].scatter(df_clean['human_score'], residuals, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Human Similarity Scores')
    axes[1, 1].set_ylabel('Residuals (Model - Human)')
    axes[1, 1].set_title('Residuals Plot')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simlex_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional analysis
    print(f"\nDetailed Analysis:")
    print(f"Model score range: {df_clean['model_score'].min():.3f} to {df_clean['model_score'].max():.3f}")
    print(f"Human score range: {df_clean['human_score'].min():.3f} to {df_clean['human_score'].max():.3f}")
    print(f"Mean absolute error: {np.mean(np.abs(residuals)):.3f}")
    print(f"Root mean square error: {np.sqrt(np.mean(residuals**2)):.3f}")

    # Show worst and best predictions
    df_clean['abs_error'] = np.abs(residuals)
    worst_predictions = df_clean.nlargest(5, 'abs_error')
    best_predictions = df_clean.nsmallest(5, 'abs_error')

    print(f"\nWorst predictions (highest error):")
    print(worst_predictions[['word1', 'word2', 'human_score', 'model_score', 'abs_error']])

    print(f"\nBest predictions (lowest error):")
    print(best_predictions[['word1', 'word2', 'human_score', 'model_score', 'abs_error']])

    # Quality benchmark
    if spearman_corr >= 0.4:
        quality = "EXCELLENT"
    elif spearman_corr >= 0.3:
        quality = "GOOD"
    elif spearman_corr >= 0.2:
        quality = "FAIR"
    else:
        quality = "POOR"

    print(f"\nModel Quality: {quality}")
    print(f"Coverage: {len(df_clean)/len(df)*100:.1f}%")

if __name__ == "__main__":
    main()
