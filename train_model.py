import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ==========================================
# 1. MEMBACA DATASET
# ==========================================

print("Membaca dataset...")

df = pd.read_csv("dataset/diabetes_risk.csv", sep=";")

# Menghapus spasi pada nama kolom dan target
df.columns = df.columns.str.strip()
df["diabetes_risk"] = df["diabetes_risk"].str.strip()

print("Jumlah data:", df.shape)


# ==========================================
# 2. MENENTUKAN FITUR DAN TARGET
# ==========================================

# Kolom target yang ingin diprediksi
target = "diabetes_risk"

# ID pasien tidak digunakan sebagai fitur
X = df.drop(columns=[target, "patient_id"])
y = df[target]

print("\nFitur yang digunakan:")
print(X.columns.tolist())

print("\nKategori target:")
print(y.value_counts())


# ==========================================
# 3. MEMBAGI FITUR NUMERIK DAN KATEGORIK
# ==========================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print("\nFitur numerik:", numeric_features)
print("Fitur kategorik:", categorical_features)


# ==========================================
# 4. MENANGANI DATA KOSONG
# ==========================================

# Data numerik diisi menggunakan nilai median
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Data kategori diisi menggunakan kategori terbanyak
# Kemudian diubah menjadi angka menggunakan OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Menggabungkan proses pengolahan data
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ==========================================
# 5. MEMBAGI DATA TRAINING DAN TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))


# ==========================================
# 6. MEMBUAT MODEL RANDOM FOREST
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])


# ==========================================
# 7. MELATIH MODEL
# ==========================================

print("\nSedang melatih model...")

model.fit(X_train, y_train)

print("Pelatihan model selesai!")


# ==========================================
# 8. MELAKUKAN PREDIKSI
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. EVALUASI MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n========== HASIL EVALUASI ==========")

print(f"Akurasi: {accuracy * 100:.2f}%")
print(f"Macro F1-Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Moderate", "High"]
))


# ==========================================
# 10. MENYIMPAN MODEL
# ==========================================

os.makedirs("prediction", exist_ok=True)

joblib.dump(model, "prediction/model_diabetes.joblib")

print("\nModel berhasil disimpan!")
print("Lokasi: prediction/model_diabetes.joblib")
