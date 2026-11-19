import pandas as pd

# Membaca dataset dengan pemisah titik koma
df = pd.read_csv("dataset/diabetes_risk.csv", sep=";")

print("===== UKURAN DATASET =====")
print(df.shape)

print("\n===== NAMA KOLOM =====")
print(df.columns.tolist())

print("\n===== LIMA DATA PERTAMA =====")
print(df.head().to_string())

print("\n===== TIPE DATA =====")
print(df.dtypes)

print("\n===== DATA KOSONG =====")
print(df.isnull().sum())

print("\n===== DISTRIBUSI TARGET =====")
print(df["diabetes_risk"].value_counts())

print("\n===== INFORMASI DATASET =====")
df.info()
