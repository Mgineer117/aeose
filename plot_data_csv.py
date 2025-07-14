import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === CONFIG ===
csv_path = "data.csv"
smoothing_window = 5  # Adjust to control smoothness
reward_change_step = 40000

# === Load CSV ===
df = pd.read_csv(csv_path, sep=None, engine="python")  # Auto-detect separator
df.columns = df.columns.str.strip()  # Strip whitespace

# Debug print
print("Detected columns:", df.columns.tolist())

if "Step" not in df.columns:
    raise KeyError("Could not find 'Step' column.")

steps = df["Step"]

n_lines = len(df.columns[1:])
colors = cm.viridis(np.linspace(1.0, 0.3, n_lines))  # reversed: intense to pale

plt.figure(figsize=(12, 7))

lines = []
labels = []

for i, col in enumerate(df.columns[1:]):
    raw = df[col]
    smoothed = raw.rolling(window=smoothing_window, min_periods=1, center=True).mean()
    (line,) = plt.plot(steps, smoothed, label=col, c=colors[i])
    lines.append(line)
    labels.append(col)

# Reverse legend order
plt.legend(
    lines[::-1],
    labels[::-1],
    fontsize=16,
    title="Hidden Layer Nodes (Parameters)",
    title_fontsize=18,
)
plt.xlabel("Environment Time Steps", fontsize=24)
plt.ylabel("Evaluation Return", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.grid(True)

ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(24)  # 👈 This line enlarges '1e6'

plt.tight_layout()
plt.show()
