import re
import numpy as np
import pandas as pd

# Load raw text data
with open("pollution.txt", "r") as file:
    raw = file.read()

# Define column names
cols = ["PREC", "JANT", "JULT", "OVR65", "POPN", "EDUC", "HOUS",
        "DENS", "NONW", "WWDRK", "POOR", "HC", "NOX", "SO2",
        "HUMID", "MORT"]

# Step 2a: Skip non-data lines and collect numeric tokens
started = False
keepers = []

for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    numeric_tokens = re.findall(r'-?\d+(?:\.\d+)?', line)
    if not started:
        if re.match(r'^\d', line) and len(numeric_tokens) >= len(cols):
            started = True
        else:
            continue
    if started:
        keepers.extend(numeric_tokens)

# Step 2b: Convert to floats and reshape
numbers = np.array(keepers, dtype=float)
if numbers.size % len(cols) != 0:
    raise ValueError(f"Total numbers ({numbers.size}) not divisible by {len(cols)} – check raw file format.")

df = pd.DataFrame(numbers.reshape(-1, len(cols)), columns=cols)

# Display results
print("Shape:", df.shape)
print(df.head())
