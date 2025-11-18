import pandas as pd, numpy as np
from sklearn.preprocessing import QuantileTransformer

orig = pd.read_csv('../data/new_data.csv')

fraud = orig[orig['isFraud'] == 1].drop(columns=['isFraud'])
non_fraud = orig[orig['isFraud'] == 0].drop(columns=['isFraud'])

def synthesize(df, n):
    num_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
    cat_cols = ['type','nameOrig','nameDest']

    num = df[num_cols]

    # Normalise numeric to uniform via quantile transform
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    Z = qt.fit_transform(num)
    # Fit covariance
    mean, cov = Z.mean(axis=0), np.cov(Z, rowvar=False)
    # Sample correlated Gaussian noise
    Z_new = np.random.multivariate_normal(mean, cov, n)
    # Inverse transform to original scale
    num_new = pd.DataFrame(qt.inverse_transform(Z_new), columns=num_cols)

    # Sample categorical features independently
    # regenerate IDs
    cat_new = pd.DataFrame()
    cat_new['type'] = np.random.choice(df['type'].unique(), size=n,
                                       p=df['type'].value_counts(normalize=True).values)
    cat_new['nameOrig'] = ['C' + str(np.random.randint(1_000_000, 9_999_999_999, dtype=np.int64)) for _ in range(n)]
    cat_new['nameDest'] = ['M' + str(np.random.randint(1_000_000, 9_999_999_999, dtype=np.int64)) for _ in range(n)]

    return pd.concat([num_new, cat_new], axis=1)

new_fraud = synthesize(fraud, 2500)
new_non_fraud = synthesize(non_fraud, 7500)

final = pd.concat([
    new_fraud.assign(isFraud=1),
    new_non_fraud.assign(isFraud=0)
], ignore_index=True).sample(frac=1, random_state=42)

# round and clip to valid ranges
final['step'] = final['step'].round().clip(1,743)
num_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
final[num_cols[1:]] = final[num_cols[1:]].clip(lower=0)

final.to_csv('../data/synthetic_upi_fraud_detection_dataset_statistical.csv', index=False)
print("Saved as synthetic_upi_fraud_detection_dataset_statistical.csv")
print(final['isFraud'].value_counts())