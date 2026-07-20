# 🛒 Segmentasi Perilaku Belanja Konsumen Menggunakan K-Means Clustering

Aplikasi ini merupakan implementasi algoritma **K-Means Clustering** untuk melakukan segmentasi pelanggan berdasarkan perilaku belanja. Hasil segmentasi divisualisasikan melalui dashboard interaktif menggunakan **Streamlit**.

---

# 📌 Fitur

- 📊 Visualisasi distribusi cluster
- 📈 Profil masing-masing segmen pelanggan
- 🔮 Prediksi cluster pelanggan baru
- 💡 Rekomendasi strategi pemasaran berdasarkan hasil segmentasi
- 📉 Visualisasi profil pelanggan
- 🎨 Dashboard interaktif berbasis Streamlit

---

# 🛠️ Teknologi

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Joblib

---

# 📂 Struktur Folder

```
SEGMEN_PERILAKU/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── hasil_segmentasi.csv
│
└── model/
    ├── scaler.pkl
    └── kmeans_model.pkl
```

---

# 🚀 Cara Menjalankan Project

## 1. Clone Repository

```bash
git clone https://github.com/putriii0/segmen_perilaku1.git
```

Masuk ke folder project

```bash
cd segmen_perilaku1
```

---

## 2. Install Library

```bash
pip install -r requirements.txt
```

Apabila file `requirements.txt` masih kosong, jalankan:

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

---

## 3. Jalankan Website


```bash
python -m streamlit run app.py
```

---

## 4. Buka Dashboard

Setelah berhasil dijalankan, Streamlit akan menampilkan alamat seperti:

```
Local URL: http://localhost:8501
```

Buka browser kemudian akses:

```
http://localhost:8501
```

---

# 📊 Menu Dashboard

## 📈 Dataset & Visualisasi

Menampilkan:

- Jumlah data
- Jumlah fitur
- Jumlah cluster
- Distribusi cluster
- Proporsi cluster
- Profil setiap segmen pelanggan

---

## 🔮 Demo Prediksi Cluster

Masukkan data pelanggan berupa:

- Jumlah Pesanan Online/Bulan
- Jumlah Kunjungan Toko/Bulan
- Rata-rata Belanja Online
- Rata-rata Belanja Offline

Sistem akan menghasilkan:

- Hasil segmentasi
- Nama segmen
- Strategi pemasaran
- Ringkasan data input
- Visualisasi profil pelanggan

---

# 🤖 Model Machine Learning

Model yang digunakan:

- StandardScaler
- K-Means Clustering (8 Cluster)

Model disimpan pada folder:

```
model/
```

---

# 📁 Dataset

Dataset yang digunakan telah melalui tahapan:

- Data Cleaning
- Feature Selection
- Standardisasi Data
- Clustering menggunakan K-Means

Hasil clustering disimpan pada file:

```
data/hasil_segmentasi.csv
```

---

# 👩‍💻 Author

**Putri**

GitHub: https://github.com/putriii0

Repository:
https://github.com/putriii0/segmen_perilaku1