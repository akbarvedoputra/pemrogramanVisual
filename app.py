import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="KNN Diabetes Classifier", layout="wide")

st.title("🩺 KNN Diabetes Classification App")
st.write("Aplikasi ini memprediksi kemungkinan diabetes menggunakan algoritma **KNN**")

st.sidebar.header("⚙️ Pengaturan Model")

# Upload file
uploaded_file = st.sidebar.file_uploader("diabetes.csv", type=["csv"])

# Default dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Gunakan dataset default `diabetes.csv`")
    data = pd.read_csv("diabetes.csv")

st.write("### 📊 Data Preview")
st.dataframe(data.head())

# Membersihkan nilai nol pada kolom tertentu
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    data[col] = data[col].replace(0, data[col].median())

# Pisahkan fitur dan label
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# SMOTE untuk balance data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Scaling
scaler = MinMaxScaler()
X_res = scaler.fit_transform(X_res)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Pilih nilai K
k = st.sidebar.slider("Pilih nilai K", 1, 21, 3, step=2)

# Train model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Akurasi
acc = accuracy_score(y_test, y_pred)

st.write(f"### ✅ Akurasi Model (K={k}): **{acc:.4f}**")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

st.write("### 🔎 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.pyplot(fig)

# Classification Report
st.write("### 📑 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Uji akurasi beberapa nilai K
nilai_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
akurasi = []

for kk in nilai_k:
    knn = KNeighborsClassifier(n_neighbors=kk)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    akurasi.append(accuracy_score(y_test, pred))

st.write("### 📈 Grafik Akurasi untuk berbagai nilai K")
fig2, ax2 = plt.subplots()
ax2.plot(nilai_k, akurasi, marker="o", linestyle="--")
ax2.set_xlabel("Nilai K")
ax2.set_ylabel("Akurasi")
ax2.set_title("Akurasi vs Nilai K")
st.pyplot(fig2)

st.success("✔️ Selesai! Silakan ubah nilai K di sidebar untuk melihat perubahan akurasi")
