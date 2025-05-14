import json
import pandas as pd
import os
import numpy as np

# This script generates payload.json for model testing
# It reads the first row of the processed dataset and dumps all columns + values

# 1. Load the processed data
df = pd.read_csv(os.path.join("data", "ames_processed.csv"))

# 2. Choose a sample row (e.g., the first one)
sample = df.iloc[0]

# 3. Convert values to native Python types to ensure JSON serializability
columns = sample.index.tolist()
data_row = []
for v in sample.values:
    if isinstance(v, (np.integer,)):
        data_row.append(int(v))
    elif isinstance(v, (np.floating,)):
        data_row.append(float(v))
    elif isinstance(v, (np.bool_,)):
        data_row.append(bool(v))
    else:
        data_row.append(v)

# 4. Build payload structure
payload = {
    "columns": columns,
    "data": [data_row]
}

# 5. Write payload.json to project root
with open("payload.json", "w") as f:
    json.dump(payload, f)

print(f"Wrote payload.json with {len(columns)} columns.")