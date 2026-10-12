import pandas as pd

FILE = "C:/Users/tanma/phishing-detector/ml/data/raw/PhiUSIIL_Phishing_URL_Dataset.csv/PhiUSIIL_Phishing_URL_Dataset.csv"

print("Loading CSV... please wait")
df = pd.read_csv(FILE)

print("Total rows    :", len(df))
print("Total columns :", len(df.columns))
print("\nColumn names:")
for col in df.columns:
    print(" -", col)
print("\nLabel distribution:")
print(df.iloc[:, -1].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("\nDone!")