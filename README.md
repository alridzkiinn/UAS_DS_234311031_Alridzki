# 📘 Judul Proyek
*(Prediksi Rating Hotel di Las Vegas Strip Menggunakan Machine Learning dan Deep Learning)*

## 👤 Informasi
- **Nama:** [Alridzki Innama Nur Razzaaq]  
- **Repo:** [https://github.com/alridzkiinn/UAS_DS_234311031_Alridzki.git]  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi **rating hotel di kawasan Las Vegas Strip** berdasarkan data ulasan pengguna TripAdvisor menggunakan pendekatan **Machine Learning dan Deep Learning**.  
Dataset yang digunakan berupa data tabular dengan fitur numerik yang merepresentasikan aspek penilaian hotel seperti jumlah ulasan, vote helpful, kualitas kamar, pelayanan, dan faktor lainnya.

Eksperimen dilakukan dengan membangun dan membandingkan tiga model:
1. **Baseline Model:** Logistic Regression  
2. **Advanced Machine Learning Model:** Random Forest  
3. **Deep Learning Model:** Multilayer Perceptron (MLP)  

Setiap model dievaluasi menggunakan metrik klasifikasi untuk menentukan performa dan stabilitas terbaik.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- [Bagaimana memprediksi rating hotel berdasarkan fitur numerik yang tersedia pada dataset TripAdvisor?]  
- [Apakah model machine learning dan deep learning mampu meningkatkan performa dibandingkan baseline sederhana?]  

**Goals:**  
- [Membangun pipeline klasifikasi rating hotel secara end-to-end.]  
- [Membandingkan performa baseline, advanced ML, dan deep learning.]
- [Menentukan model yang paling stabil dan layak digunakan berdasarkan hasil evaluasi.]

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── DSUAS_234311031_Alridzki_Innama_Nur_Razzaaq.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── deep_learning_mlp.h5
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** [https://archive.ics.uci.edu/dataset/397/las+vegas+strip]  
- **Jumlah Data:** [±500 data ulasan]  
- **Tipe:** [Tabular]  

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| nr_reviews | Jumlah ulasan pengguna |
| helpful_votes | Jumlah vote helpful |
| nr_rooms | Jumlah kamar hotel |
| nr_hotel_reviews | Jumlah total ulasan hotel |
| member_years | Lama keanggotaan pengguna |
| cleanliness | Skor kebersihan |
| location | Skor lokasi |
| rooms | Skor kualitas kamar |
| service | Skor pelayanan |
| value | Skor nilai harga |

---

# 4. 🔧 Data Preparation
Tahapan data preparation yang dilakukan:
- **Cleaning:** Pengecekan missing value dan duplikasi data
- **Feature Selection:** Menggunakan hanya fitur numerik
- **Scaling:** StandardScaler pada data training dan testing
- **Splitting:** Data dibagi menjadi 80% training dan 20% testing dengan stratifikasi target
- **Data Balancing:** Tidak dilakukan karena distribusi kelas relatif seimbang 

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** [Model ini digunakan sebagai pembanding awal karena sederhana dan stabil pada data tabular.]  
- **Model 2 – Advanced ML:** [Random Forest digunakan untuk menangkap hubungan non-linear antar fitur dan menyediakan feature importance.]  
- **Model 3 – Deep Learning:** [Model deep learning berbasis MLP digunakan untuk mengevaluasi kemampuan neural network pada data tabular dengan arsitektur fully connected.]  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Hasil Singkat
| Model | Accuracy | Catatan |
|------|----------|---------|
| Logistic Regression | ~0.47 | Paling stabil |
| Random Forest | ~0.44 | Tidak meningkat |
| Deep Learning (MLP) | ~0.44 | Hasil fluktuatif |

Model deep learning menunjukkan hasil yang tidak konsisten antar percobaan, sedangkan baseline memberikan performa yang relatif stabil.

---

# 7. 🏁 Kesimpulan
- Model terbaik: [Logistic Regression]  
- Alasan: [Memberikan akurasi tertinggi dibandingkan model lain]
          [Stabil dan tidak sensitif terhadap inisialisasi acak]  
- Insight penting: [Deep learning tidak selalu unggul, terutama pada dataset tabular dengan ukuran data terbatas.]  

---

# 8. 🔮 Future Work
- [ ] Menambah jumlah dan variasi data
- [ ] Feature engineering lanjutan
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Mencoba ensemble methods
- [ ] Deployment model ke aplikasi web 

---

# 9. 🔁 Reproducibility
Gunakan environment:
**Python Version:** 3.10

## Dependencies
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tensorflow==2.14.0
