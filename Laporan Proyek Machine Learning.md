## INFORMASI PROYEK

**Judul Proyek:**  
PREDIKSI RATING HOTEL DI LAS VEGAS STRIP MENGGUNAKAN MACHINE LEARNING DAN DEEP LEARNING

**Nama Mahasiswa:** [Alridzki Innama nur Razzaaq]  
**NIM:** [234311031]  
**Program Studi:** [Teknologi Rekayasa Perangkat Lunak]  
**Mata Kuliah:** [Sata Science]  
**Dosen Pengampu:** [Gus Nanang Syaifuddiin, S.Kom., M.Kom.]  
**Tahun Akademik:** [2025/Semester 5]
**Link GitHub Repository:** [https://github.com/alridzkiinn/UAS_DS_234311031_Alridzki.git]
**Link Video Pembahasan:** [https://youtu.be/z1aEti95hk8]

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah dan merumuskan problem statement secara jelas
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. Melakukan data preparation yang sesuai dengan karakteristik dataset
4. Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis
7. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
8. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
**Isi bagian ini dengan:**
Industri perhotelan merupakan salah satu sektor yang sangat kompetitif, khususnya di kawasan wisata populer seperti Las Vegas Strip. Persaingan antar hotel tidak hanya ditentukan oleh lokasi dan harga, tetapi juga oleh kualitas layanan dan tingkat kepuasan pelanggan. Dalam konteks ini, rating hotel menjadi indikator penting yang mencerminkan pengalaman tamu selama menginap dan sering dijadikan dasar pengambilan keputusan oleh calon pelanggan.
Seiring dengan meningkatnya penggunaan platform digital, ulasan dan rating hotel yang dihasilkan oleh pengguna (user-generated content) semakin melimpah dan bersifat data-driven. Namun, jumlah data yang besar tersebut sulit dianalisis secara manual. Oleh karena itu, diperlukan pendekatan berbasis machine learning dan deep learning untuk mengekstraksi pola, hubungan antar fitur, serta memprediksi rating hotel secara otomatis berdasarkan karakteristik tertentu, seperti jumlah ulasan, fasilitas, dan profil pengguna.
Prediksi rating hotel memiliki manfaat praktis bagi berbagai pihak. Bagi manajemen hotel, hasil prediksi dapat digunakan sebagai alat evaluasi kualitas layanan dan pengambilan keputusan strategis. Bagi platform reservasi atau sistem rekomendasi, model prediksi rating dapat meningkatkan akurasi rekomendasi hotel kepada pengguna. Selain itu, dari sisi akademik, proyek ini menjadi studi kasus penerapan algoritma machine learning dan deep learning pada data tabular di domain pariwisata.
Dengan memanfaatkan dataset Las Vegas Strip dari UCI Machine Learning Repository, proyek ini bertujuan untuk membangun dan membandingkan beberapa pendekatan pemodelan, mulai dari model baseline, model machine learning lanjutan, hingga model deep learning, guna memprediksi rating hotel secara akurat dan sistematis.

**Contoh referensi (berformat APA/IEEE):**
UCI Machine Learning Repository. (2015). Las Vegas Strip Dataset.
Aggarwal, C. C. (2018). Machine Learning for Recommendation Systems. Springer.
Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
### 3.1 Problem Statements
1.	Bagaimana membangun model yang mampu memprediksi rating hotel berdasarkan fitur numerik yang tersedia?
2.	Bagaimana perbandingan performa antara model baseline, model machine learning lanjutan, dan model deep learning?
3.	Apakah pendekatan deep learning memberikan peningkatan performa yang signifikan dibandingkan model klasik?
4.	Bagaimana memastikan proses eksperimen bersifat reproducible?

### 3.2 Goals
1.	Membangun model klasifikasi untuk memprediksi rating hotel (1–5).
2.	Membandingkan performa tiga pendekatan model menggunakan metrik evaluasi klasifikasi.
3.	Menentukan model terbaik berdasarkan hasil evaluasi.
4.	Menghasilkan pipeline eksperimen yang dapat dijalankan ulang.

### 3.3 Solution Approach

Mahasiswa **WAJIB** menggunakan minimal **tiga model** dengan komposisi sebagai berikut:
#### **Model 1 – Baseline Model**
Model sederhana sebagai pembanding dasar.
**Pilihan model:**
Logistic Regression (Baseline)   Model ini dipilih karena sederhana, cepat dilatih, dan sering digunakan sebagai pembanding awal dalam tugas klasifikasi.

#### **Model 2 – Advanced / ML Model**
Model machine learning yang lebih kompleks.
Random Forest Classifier (Advanced ML)   Model ini mampu menangkap hubungan non-linear antar fitur dan menyediakan informasi feature importance.

#### **Model 3 – Deep Learning Model (WAJIB)**
Model deep learning yang sesuai dengan jenis data.
Multilayer Perceptron / MLP (Deep Learning)   Model ini sesuai untuk data tabular dan mampu mempelajari representasi fitur yang lebih kompleks dibandingkan model klasik.

---

## 4. DATA UNDERSTANDING
### 4.1 Informasi Dataset
**Sumber Dataset:**  
[https://archive.ics.uci.edu/dataset/397/las+vegas+strip]

**Deskripsi Dataset:**
Deskripsi Dataset:
- Jumlah baris (rows): 505
- Jumlah kolom (columns/features): 20 fitur
- Tipe data: Tabular
- Ukuran dataset: 60 KB
- Format file: CSV

### 4.2 Deskripsi Fitur
Jelaskan setiap fitur/kolom yang ada dalam dataset.
**Contoh tabel:**
| Nama Fitur | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| User country | String | Negara asal pengguna yang memberikan ulasan |
| Nr. reviews | Integer | Jumlah total ulasan yang pernah ditulis oleh pengguna |
| Nr. hotel reviews | Integer | Jumlah ulasan yang dimiliki hotel |
| Helpful votes | Integer | Jumlah suara "helpful" yang diterima pengguna |
| Score | Integer | Rating hotel yang diberikan pengguna (1–5) |
| Period of stay | String | Periode menginap pengguna |
| Traveler type | String | Jenis perjalanan pengguna |
| Pool | String | Ketersediaan kolam renang |
| Gym | String | Ketersediaan fasilitas gym |
| Tennis court | String | Ketersediaan lapangan tenis |
| Spa | String | Ketersediaan fasilitas spa |
| Casino | String | Ketersediaan fasilitas kasino |
| Free internet | String | Ketersediaan akses internet gratis |
| Hotel name | String | Nama hotel yang diulas |
| Hotel stars | Integer | Jumlah bintang hotel |
| Nr. rooms | Integer | Jumlah total kamar hotel |
| User continent | String | Benua asal pengguna |
| Member years | Integer | Lama pengguna menjadi anggota TripAdvisor (tahun) |
| Review month | String | Bulan penulisan ulasan |
| Review weekday | String | Hari dalam minggu penulisan ulasan |

### 4.3 Kondisi Data
•	Missing Values: Tidak ditemukan

•	Duplicate Data: Tidak ditemukan

•	Outliers: Ada pada fitur jumlah ulasan dan helpful votes

•	Imbalanced Data: Distribusi kelas relatif seimbang

•	Noise: Ada variasi alami pada rating pengguna

•	Data Quality Issues: Tidak ada masalah signifikan

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

#### Visualisasi 1: [Judul Visualisasi]
[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]



---

## 5. DATA PREPARATION

Bagian ini menjelaskan **semua** proses transformasi dan preprocessing data yang dilakukan.
### 5.1 Data Cleaning
Aktivitas: Tidak ditemukan missing values dan data duplikat. Outliers ditangani secara implisit oleh model (Random Forest) dan melalui standardisasi fitur untuk model lain.
### 5.2 Feature Engineering
Aktivitas: Tidak dilakukan pembuatan fitur baru. Hanya fitur numerik yang digunakan untuk menjaga konsistensi input model.
### 5.3 Data Transformation
•	Scaling: StandardScaler digunakan untuk Logistic Regression dan MLP.
<img width="808" height="245" alt="image" src="https://github.com/user-attachments/assets/5923cfcb-ced7-4a36-9aae-46b85d7bda25" />
•	Encoding: Tidak diperlukan karena target berupa numerik ordinal.
### 5.4 Data Splitting
Strategi pembagian data:
Dataset Awal : ± 155 sampel
Dataset dibagi menjadi 2 bagian:
Dataset dibagi menjadi dua subset utama menggunakan teknik stratified train–test split untuk menjaga proporsi kelas target (Score) tetap konsisten pada setiap subset. Sebanyak 80% data digunakan sebagai training set, sedangkan 20% sisanya digunakan sebagai test set untuk evaluasi akhir model.
Pada tahap pelatihan model deep learning, sebagian data training kembali dibagi secara internal dengan validation split sebesar 20% untuk keperluan pemantauan performa dan pencegahan overfitting selama proses training. Pembagian data menggunakan random state sebesar 42 agar hasil eksperimen dapat direproduksi.

---

## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model

**Nama Model:** Logistic Regression
**Teori Singkat:**  
Logistic Regression merupakan algoritma klasifikasi yang memodelkan hubungan antara fitur input dan probabilitas kelas target menggunakan fungsi logistik (sigmoid). Model ini bekerja dengan mengestimasi parameter yang memaksimalkan kemungkinan prediksi kelas yang benar berdasarkan kombinasi linier dari fitur input.
**Alasan Pemilihan:**  
Logistic Regression dipilih sebagai model baseline karena kesederhanaannya, efisiensi komputasi yang tinggi, serta kemampuannya memberikan performa awal yang dapat dijadikan pembanding terhadap model yang lebih kompleks. Selain itu, model ini cocok untuk data tabular numerik dan mudah diinterpretasikan.

#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
Parameter yang digunakan pada model Logistic Regression adalah sebagai berikut:
•	solver: 'lbfgs'
•	max_iter: 1000
•	regularization: default (L2)
Pengaturan jumlah iterasi yang lebih besar digunakan untuk memastikan proses optimisasi konvergen dengan baik.
```

#### 6.1.3 Hasil Awal

Logistic Regression sebagai model baseline memperoleh nilai akurasi sebesar 46%. Hasil ini menunjukkan bahwa model baseline mampu memberikan performa prediksi yang cukup stabil pada dataset Las Vegas Strip, meskipun menggunakan pendekatan yang relatif sederhana. Nilai akurasi ini digunakan sebagai acuan awal (baseline) untuk membandingkan performa model machine learning lanjutan dan model deep learning pada tahap berikutnya. 

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:** Random Forest
**Teori Singkat:**  
Random Forest merupakan metode ensemble learning yang membangun banyak pohon keputusan (decision tree) secara independen dan menggabungkan hasil prediksinya melalui mekanisme majority voting. Setiap pohon dilatih menggunakan subset data dan fitur yang dipilih secara acak, sehingga mampu mengurangi risiko overfitting dan meningkatkan kemampuan generalisasi model.

**Alasan Pemilihan:**  
Random Forest dipilih sebagai model pembanding tingkat menengah karena kemampuannya dalam menangkap hubungan nonlinier antar fitur, ketahanannya terhadap noise, serta performanya yang umumnya lebih baik dibandingkan model baseline. Model ini juga tidak memerlukan asumsi distribusi data tertentu.

**Keunggulan:**
Model Random Forest memiliki kemampuan yang baik dalam menangkap hubungan non-linear antar fitur karena menggunakan pendekatan ensemble dari banyak decision tree. Model ini relatif robust terhadap noise dan outlier, serta tidak memerlukan proses feature scaling. Selain itu, Random Forest menyediakan informasi feature importance yang berguna untuk memahami kontribusi setiap fitur terhadap prediksi, sehingga meningkatkan interpretabilitas model pada data tabular.

**Kelemahan:**
Meskipun lebih kompleks, Random Forest pada dataset Las Vegas Strip tidak menunjukkan peningkatan performa yang signifikan dibandingkan model baseline. Model ini juga membutuhkan waktu training yang lebih lama dan sumber daya komputasi yang lebih besar. Selain itu, dengan jumlah data dan fitur yang terbatas, Random Forest berpotensi mengalami overfitting, sehingga kompleksitas tambahan tidak selalu sebanding dengan peningkatan akurasi yang diperoleh.

#### 6.2.2 Hyperparameter

**Parameter yang digunakan:**
```
Parameter yang digunakan pada model Random Forest adalah sebagai berikut:
•	n_estimators: 200
•	max_depth: None
•	random_state: 42
•	class_weight: None
Penggunaan jumlah pohon yang relatif besar bertujuan untuk meningkatkan stabilitas prediksi, sedangkan random_state digunakan untuk memastikan reprodusibilitas hasil.
```

#### 6.2.3 Hasil Model

Berdasarkan hasil pengujian pada data test, model Random Forest memperoleh nilai akurasi sebesar 43%. Nilai ini masih berada di bawah performa model baseline yang mencapai akurasi 46%. Hasil ini menunjukkan bahwa pada dataset yang digunakan, Random Forest belum mampu meningkatkan performa prediksi secara signifikan dibandingkan model baseline.

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** Multilayer Perceptron (MLP)

** (Centang) Jenis Deep Learning: **
- [✅] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
Multilayer Perceptron (MLP) dipilih karena dataset Las Vegas Strip merupakan data tabular numerik yang tidak memiliki struktur spasial maupun temporal. MLP mampu mempelajari hubungan non-linear antar fitur numerik secara efektif melalui beberapa hidden layer. Selain itu, MLP digunakan untuk mengevaluasi potensi peningkatan performa dibandingkan model machine learning tradisional dengan memanfaatkan kemampuan representasi kompleks dari deep learning.
#### 6.3.2 Arsitektur Model

**Deskripsi Layer:**

Arsitektur MLP yang digunakan terdiri dari beberapa fully connected layer dengan regularisasi dropout untuk mengurangi risiko overfitting.
Deskripsi Layer:
1.	Input Layer: shape (jumlah fitur numerik)
2.	Dense Layer: 128 unit, activation = ReLU
3.	Dropout: 0.3
4.	Dense Layer: 64 unit, activation = ReLU
5.	Dropout: 0.3
6.	Output Layer: jumlah kelas rating (5 kelas), activation = Softmax
Model ini dirancang untuk melakukan klasifikasi multikelas pada skor rating hotel dengan output berupa probabilitas untuk setiap kelas.
```

#### 6.3.3 Input & Preprocessing Khusus
 
**Preprocessing khusus untuk DL:**
Input shape: Jumlah fitur numerik hasil preprocessing (X_train_scaled.shape[1])

Preprocessing khusus untuk DL: 
•	Standardisasi fitur menggunakan StandardScaler
•	Seluruh fitur numerik dinormalisasi agar memiliki mean 0 dan standar deviasi 1
•	Target label disesuaikan menjadi 0-indexed (rating 1–5 diubah menjadi 0–4) untuk kompatibilitas dengan fungsi loss sparse_categorical_crossentropy


#### 6.3.4 Hyperparameter

**Training Configuration:**
```
Training Configuration:
•	Optimizer: Adam
•	Learning rate: Default (Adam)
•	Loss function: sparse_categorical_crossentropy
•	Metrics: accuracy
•	Batch size: 32
•	Epochs: 30
•	Validation split: 0.2
•	Callbacks: Tidak digunakan
```

#### 6.3.5 Training Process

**Training Time:**  
Relatif singkat (8 Detik) karena ukuran dataset dan arsitektur model yang sederhana.

**Computational Resource:**  
CPU – dijalankan pada lingkungan lokal / Google Colab.

**Training History Visualization:**

<img width="480" height="740" alt="image" src="https://github.com/user-attachments/assets/08a84637-a1a1-4ab0-a004-45f9c2058118" />

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi

Untuk Klasifikasi:
Accuracy: Proporsi prediksi yang benar
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
F1-Score: Harmonic mean dari precision dan recall
ROC-AUC: Area under ROC curve
Confusion Matrix: Visualisasi prediksi
Untuk Regresi:
MSE (Mean Squared Error): Rata-rata kuadrat error
RMSE (Root Mean Squared Error): Akar dari MSE
MAE (Mean Absolute Error): Rata-rata absolute error
R² Score: Koefisien determinasi
MAPE (Mean Absolute Percentage Error): Error dalam persentase
Untuk NLP (Text Classification):
Accuracy
F1-Score (terutama untuk imbalanced data)
Precision & Recall
Perplexity (untuk language models)
Untuk Computer Vision:
Accuracy
IoU (Intersection over Union) - untuk object detection/segmentation
Dice Coefficient - untuk segmentation
mAP (mean Average Precision) - untuk object detection
Untuk Clustering:
Silhouette Score
Davies-Bouldin Index
Calinski-Harabasz Index
Untuk Recommender System:
RMSE
Precision@K
Recall@K
NDCG (Normalized Discounted Cumulative Gain)

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
Metrik:
•	Accuracy: 0.465

•	Precision (weighted avg): 0.37

•	Recall (weighted avg): 0.47

•	F1-Score (weighted avg): 0.33

```

<img width="383" height="347" alt="image" src="https://github.com/user-attachments/assets/701b59e2-dc64-4e83-bfaa-857aed294ac1" />


#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
Metrik:
•	Accuracy: 0.436

•	Precision (weighted avg): 0.46

•	Recall (weighted avg): 0.44

•	F1-Score (weighted avg): 0.37

```

**Confusion Matrix / Visualization:**  
<img width="385" height="345" alt="image" src="https://github.com/user-attachments/assets/b9635dfc-7046-4d81-ae5c-afcff5e5ed81" />

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
Metrik:
•	Accuracy: 0.436

•	Precision (weighted avg): 0.32

•	Recall (weighted avg): 0.44

•	F1-Score (weighted avg): 0.35

```

**Confusion Matrix / Visualization:**  
<img width="380" height="345" alt="image" src="https://github.com/user-attachments/assets/85c81749-7a01-44c9-a1e4-14fc0e834b2c" />

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Opsional: tampilkan beberapa contoh prediksi]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | Recall |
|-------|----------|-----------|--------|--------|
| Baseline (Model 1) | 0.465 | 0.37 | 0.47 | 0.33 |
| Advanced (Model 2) | 0.436 | 0.46 | 0.44 | 0.37 |
| Deep Learning (Model 3) | 0.436 | 0.32 | 0.44 | 0.35 |

**Visualisasi Perbandingan:**  
<img width="561" height="384" alt="image" src="https://github.com/user-attachments/assets/f4147d64-1274-4d13-a343-5263d978a81b" />

### 7.4 Analisis Hasil

**Interpretasi:**
Model Terbaik:
Berdasarkan hasil evaluasi pada data uji, Logistic Regression dapat dianggap sebagai model dengan performa terbaik secara keseluruhan karena menghasilkan nilai accuracy tertinggi (0.465) dibandingkan Random Forest dan Deep Learning. Meskipun model ini sederhana, Logistic Regression mampu memberikan prediksi yang relatif stabil pada kelas mayoritas, sehingga lebih efektif pada kondisi distribusi data yang tidak merata.

Perbandingan dengan Baseline:
Dibandingkan dengan baseline Logistic Regression, Random Forest dan Deep Learning tidak menunjukkan peningkatan performa accuracy yang signifikan. Random Forest memang memberikan peningkatan pada precision dan F1-score, namun diikuti dengan penurunan accuracy. Sementara itu, model Deep Learning menghasilkan performa yang sebanding dengan Random Forest, tetapi tidak mampu melampaui baseline. Hal ini menunjukkan bahwa penggunaan model yang lebih kompleks tidak selalu menjamin peningkatan kinerja pada dataset ini.

Trade-off:
Terdapat trade-off yang jelas antara kompleksitas model dan performa. Logistic Regression memiliki kompleksitas rendah, waktu training yang cepat, serta interpretabilitas yang tinggi. Random Forest dan Deep Learning membutuhkan sumber daya komputasi yang lebih besar dan waktu training lebih lama, namun tidak memberikan peningkatan performa yang sepadan. Dengan demikian, dari perspektif efisiensi dan kemudahan interpretasi, model baseline lebih menguntungkan.

Error Analysis:
Kesalahan prediksi paling banyak terjadi pada kelas rating rendah (1–3), yang memiliki jumlah sampel sangat sedikit. Ketiga model cenderung salah mengklasifikasikan kelas-kelas tersebut sebagai rating yang lebih tinggi, terutama rating 4 dan 5. Hal ini menunjukkan bahwa model mengalami kesulitan dalam membedakan kelas minoritas akibat keterbatasan data dan dominasi kelas mayoritas.

Overfitting/Underfitting:
Tidak ditemukan indikasi overfitting yang signifikan, khususnya pada model Deep Learning, karena nilai training dan validation loss relatif stabil. Namun, performa yang rendah pada seluruh model mengindikasikan adanya underfitting, di mana model belum mampu menangkap pola yang cukup kompleks untuk memprediksi seluruh kelas rating dengan baik, terutama pada kelas minoritas.

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
Berdasarkan hasil evaluasi pada data uji, Logistic Regression ditetapkan sebagai model terbaik dalam penelitian ini.

**Alasan:**  
Logistic Regression menghasilkan nilai accuracy tertinggi (≈46,5%) dibandingkan Random Forest dan Deep Learning. Selain itu, model ini menunjukkan stabilitas hasil yang lebih konsisten, waktu training yang cepat, serta kompleksitas yang rendah, sehingga lebih sesuai dengan karakteristik dataset yang memiliki distribusi kelas tidak seimbang dan jumlah fitur terbatas.

**Pencapaian Goals:**  
Tujuan penelitian pada Section 3.2, yaitu membangun dan membandingkan model Machine Learning dan Deep Learning untuk memprediksi rating hotel di Las Vegas Strip, telah tercapai. Penelitian ini berhasil mengevaluasi performa tiga pendekatan berbeda serta menganalisis kelebihan dan keterbatasan masing-masing model.

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1 Dataset memiliki ketidakseimbangan kelas, dengan dominasi rating tinggi (4 dan 5) dibandingkan rating rendah.]
- [Insight 2 Fitur numerik seperti jumlah ulasan dan helpful votes memiliki kontribusi yang lebih dominan dibandingkan fitur lainnya.]
- [Insight 3 Keterbatasan jumlah data pada rating rendah menyulitkan model dalam melakukan prediksi yang akurat untuk kelas minoritas.]

**Insight dari Modeling:**
- [Insight 1 Model yang lebih kompleks seperti Random Forest dan Deep Learning tidak selalu menghasilkan performa yang lebih baik pada dataset tabular berukuran kecil hingga menengah.]
- [Insight 2 Model sederhana seperti Logistic Regression dapat memberikan hasil yang kompetitif ketika data tidak terlalu kompleks dan fitur sudah representatif.]

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
Hasil penelitian ini dapat dimanfaatkan sebagai sistem pendukung keputusan bagi pengelola hotel atau platform ulasan untuk memahami faktor-faktor yang memengaruhi rating hotel serta sebagai dasar awal dalam pengembangan sistem prediksi kepuasan pelanggan.

**Pembelajaran yang didapat:**  
Proyek ini memberikan pemahaman mengenai pentingnya eksplorasi data, pemilihan model yang sesuai dengan karakteristik dataset, serta evaluasi performa secara komprehensif. Selain itu, penelitian ini menegaskan bahwa kompleksitas model harus disesuaikan dengan kualitas dan distribusi data agar menghasilkan prediksi yang optimal.

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [✅] Mengumpulkan lebih banyak data
- [✅] Menambah variasi data
- [✅] Feature engineering lebih lanjut

**Model:**
- [✅] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [✅] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [✅] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
Main Libraries & Versions:

numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```
