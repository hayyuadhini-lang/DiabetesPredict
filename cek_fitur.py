import pandas as pd

# Membaca dataset
df = pd.read_csv("dataset/diabetes_risk.csv", sep=";")

# Fitur numerik yang akan digunakan untuk slider
fitur_numerik = [
    "age",
    "bmi",
    "hours_sleep_per_night",
    "stress_level",
    "fasting_blood_sugar",
    "hba1c_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "waist_circumference_cm"
]

print("===== RENTANG FITUR NUMERIK =====")

for kolom in fitur_numerik:
    print(f"\n{kolom}")
    print("Minimum :", df[kolom].min())
    print("Maksimum:", df[kolom].max())
    print("Rata-rata:", round(df[kolom].mean(), 2))

# Melihat pilihan fitur kategorik
fitur_kategorik = [
    "gender",
    "city",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "income_bracket"
]

print("\n===== PILIHAN FITUR KATEGORIK =====")

for kolom in fitur_kategorik:
    print(f"\n{kolom}")
    print(df[kolom].value_counts(dropna=False).to_string())
