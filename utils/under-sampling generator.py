import pandas as pd

# original dataset
df = pd.read_csv("../data/new_data.csv")

fraud = df[df['isFraud'] == 1]
nonfraud = df[df['isFraud'] == 0]

sampled_fraud = fraud.sample(n=2500, random_state=42)
sampled_nonfraud = nonfraud.sample(n=7500, random_state=42)

df_balanced = pd.concat([sampled_fraud, sampled_nonfraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# save undersampled dataset
df_balanced.to_csv("../data/upi_fraud_detection_dataset_under_sampled.csv", index=False)

print("Saved as upi_fraud_detection_dataset_under_sampled.csv")
print(df_balanced['isFraud'].value_counts())