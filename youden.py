import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

os.chdir('/path/to/dir')

def calculate_youden_index(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    max_index = np.argmax(youden_index)

    return {
        'youden_index': youden_index[max_index],
        'optimal_threshold': thresholds[max_index],
        'sensitivity': tpr[max_index],
        'specificity': 1 - fpr[max_index],
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_index': max_index
    }

# === 1. Load Excel File ===
file_path = "youden_input.xlsx"  
df = pd.read_excel(file_path)

# Ensure proper column names
y_true = df['y_true']
y_prob = df['y_prob']

# === 2. Calculate Youden Index ===
results = calculate_youden_index(y_true, y_prob)

# === 3. Print Results ===
print("Youden Index:", results['youden_index'])
print("Optimal Threshold:", results['optimal_threshold'])
print("Sensitivity:", results['sensitivity'])
print("Specificity:", results['specificity'])

# === 4. (Optional) Plot ROC and mark optimal point ===
plt.figure(figsize=(8, 6))
plt.plot(results['fpr'], results['tpr'], label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.scatter(results['fpr'][results['optimal_index']],
            results['tpr'][results['optimal_index']],
            color='red', label=f'Optimal threshold: {results["optimal_threshold"]:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Youden Index Optimal Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
