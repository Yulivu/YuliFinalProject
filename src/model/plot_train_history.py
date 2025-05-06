import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
from datetime import datetime

def load_history(history_path):
    ext = os.path.splitext(history_path)[-1]
    if ext == '.csv':
        return pd.read_csv(history_path)
    elif ext == '.pkl':
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    elif ext == '.json':
        with open(history_path, 'r') as f:
            return pd.DataFrame(json.load(f))
    else:
        raise ValueError('不支持的文件格式: ' + ext)

def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Group metrics
    train_metrics = [col for col in history.columns if col.startswith(('accuracy', 'auc', 'loss', 'precision', 'recall', 'learning_rate')) and not col.startswith('val_')]
    # Draw each metric
    for metric in train_metrics:
        if metric == 'learning_rate':
            plt.figure()
            plt.plot(history[metric], label=metric, color='tab:orange')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} over epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'))
            plt.close()
        else:
            plt.figure()
            plt.plot(history[metric], label=f'train_{metric}')
            val_metric = f'val_{metric}' if f'val_{metric}' in history.columns else None
            if val_metric:
                plt.plot(history[val_metric], label=val_metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} and validation over epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'))
            plt.close()
    print(f'All curves saved to {save_dir}')

if __name__ == '__main__':
    # Your csv path
    history_path = r'C:\Users\debuf\Desktop\YuliFinalProject\result\unet\draw\unet_32_20250506_013740_training_history.csv'
    # Create a unique subfolder for each run
    base_dir = r'C:\Users\debuf\Desktop\YuliFinalProject\result\unet\training_his'
    run_name = 'run_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_dir, run_name)
    history = pd.read_csv(history_path)
    plot_history(history, save_dir)