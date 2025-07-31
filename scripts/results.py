import argparse
import pandas as pd
import matplotlib.pyplot as plt



def create_plot(path: str):

    df = pd.read_csv(path)
    df = df[df['model_score'] != "N/A"]  # Remove rows where model_score is not available
    df['model_score'] = pd.to_numeric(df['model_score'])

    print("Pearson correlation:", df['human_score'].corr(df['model_score']))

    plt.figure(figsize=(8, 8))
    plt.scatter(df['human_score'], df['model_score'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
    plt.xlabel('SimLex (Human Score)')
    plt.ylabel('Model Score')
    plt.title('Model vs SimLex Human Scores')
    plt.legend()
    plt.grid(True)
    plt.show()

    df['diff'] = (df['human_score'] - df['model_score']).abs()
    plt.figure(figsize=(8, 4))
    plt.hist(df['diff'], bins=40, color='skyblue', edgecolor='black')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Count')
    plt.title('Distribution of Absolute Differences (Model vs SimLex)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Parse Word2Vec results')
    parser.add_argument('--file', help='Path to the results file')
    args = parser.parse_args()
    create_plot(args.file)

if __name__ == "__main__":
    main()
