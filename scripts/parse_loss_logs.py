#!/usr/bin/env python3
"""
Parse Word2Vec training logs and visualize loss function
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import numpy as np

def parse_log_file(log_file_path):
    """Parse the JSON log file and extract epoch and loss data"""
    epochs = []
    losses = []

    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())

                # Check if this is a training epoch log
                if (log_entry.get('message') == 'Training epoch' and
                    'epoch' in log_entry and
                    'epoch_loss' in log_entry):

                    epoch = log_entry['epoch']
                    loss = log_entry['epoch_loss']

                    # Skip null/NaN losses
                    if loss is not None and isinstance(loss, (int, float)):
                        epochs.append(epoch)
                        losses.append(loss)

            except json.JSONDecodeError:
                continue

    return epochs, losses

def create_loss_chart(epochs, losses, output_file='loss_chart.png'):
    """Create and save a loss function chart"""
    plt.figure(figsize=(12, 8))

    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, losses, 'b-', linewidth=2, alpha=0.7)
    plt.title('Word2Vec Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(epochs) > 1:
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.2f})')
        plt.legend()

    # Log scale plot (useful for high losses)
    plt.subplot(2, 1, 2)
    plt.semilogy(epochs, losses, 'g-', linewidth=2, alpha=0.7)
    plt.title('Word2Vec Training Loss (Log Scale)', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Loss chart saved as {output_file}")

def analyze_loss_trend(epochs, losses):
    """Analyze the loss trend and provide insights"""
    if len(losses) < 2:
        print("Not enough data points for analysis")
        return

    print(f"\nLoss Analysis:")
    print(f"="*40)
    print(f"Initial loss: {losses[0]:.2f}")
    print(f"Final loss: {losses[-1]:.2f}")
    print(f"Loss change: {losses[-1] - losses[0]:.2f}")
    print(f"Relative change: {((losses[-1] - losses[0]) / losses[0]) * 100:.1f}%")

    # Check if loss is decreasing
    if losses[-1] < losses[0]:
        print("✅ Loss is decreasing (good)")
    else:
        print("❌ Loss is increasing (problematic)")

    # Check for stability
    recent_losses = losses[-10:] if len(losses) > 10 else losses
    loss_std = np.std(recent_losses)
    loss_mean = np.mean(recent_losses)

    if loss_std / loss_mean < 0.1:
        print("✅ Loss is stable")
    else:
        print("⚠️  Loss is unstable")

    # Check for reasonable loss values
    if losses[-1] > 100:
        print("❌ Loss is very high - check your implementation")
    elif losses[-1] > 10:
        print("⚠️  Loss is high - consider adjusting hyperparameters")
    else:
        print("✅ Loss is in reasonable range")

def main():
    parser = argparse.ArgumentParser(description='Parse Word2Vec training logs and create loss chart')
    parser.add_argument('--log-file', default='logs/word2vec_train_100_750_0.025.log',
                       help='Path to the log file')
    parser.add_argument('--output', default='loss_chart.png',
                       help='Output file for the chart')

    args = parser.parse_args()

    # Check if log file exists
    if not Path(args.log_file).exists():
        print(f"Log file not found: {args.log_file}")
        return

    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    epochs, losses = parse_log_file(args.log_file)

    if not epochs:
        print("No training data found in log file")
        return

    print(f"Found {len(epochs)} training epochs")

    # Create DataFrame for easy manipulation
    df = pd.DataFrame({'epoch': epochs, 'loss': losses})
    print(f"\nFirst 10 entries:")
    print(df.head(10))

    # Analyze loss trend
    analyze_loss_trend(epochs, losses)

    # Create chart
    import numpy as np
    create_loss_chart(epochs, losses, args.output)

if __name__ == "__main__":
    main()
