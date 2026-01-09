import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=";")

# Save locally
df.to_csv("data/winequality-red.csv", index=False)

print("Dataset saved successfully!")