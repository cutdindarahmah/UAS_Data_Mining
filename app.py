import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sidebar
st.sidebar.title("Main Page")
menu = st.sidebar.radio("Pilih Proyek", ["Classification", "Clustering"])

# Tampilan Halaman Utama
st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; 
                height: 100px; margin-top: -70px;'>
        <h1 style='color: white;'>Ujian Akhir Semester Data Mining</h1>
    </div>""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center;'>Nama: Cut Dinda Rahmah<br>NIM: 22146013</p>""", unsafe_allow_html=True)


#==================================
#============ KLASIFIKASI =========
#==================================
if menu == "Classification":
    st.header("Klasifikasi Diabetes dengan KNN")

    try:
        df = pd.read_csv('diabetes.csv')
        
        # Judul & Deskripsi
        st.subheader("Deskripsi Proyek")
        st.write("""
        Aplikasi ini menggunakan model klasifikasi K-Nearest Neighbors (KNN) untuk memprediksi apakah seorang pasien berisiko diabetes atau tidak.
        Dataset yang digunakan adalah Pima Indians Diabetes yang berisi fitur kesehatan seperti kadar glukosa, tekanan darah, dan usia.
        Model dilatih dan diuji untuk melihat akurasinya, ditampilkan melalui metrik dan confusion matrix, serta tersedia fitur input data baru untuk prediksi langsung.
        """)

        # Preprocessing
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training Model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Metrik Klasifikasi
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Metrik Klasifikasi")
        st.write(f"**Akurasi: {acc:.2f}**")

        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Input Data Baru
        st.subheader("Input Data Baru")
        pregnancies = st.number_input('Pregnancies', 0, 20, 1)
        glucose = st.number_input('Glucose', 0, 300, 120)
        blood_pressure = st.number_input('Blood Pressure', 0, 200, 70)
        skin_thickness = st.number_input('Skin Thickness', 0, 100, 20)
        insulin = st.number_input('Insulin', 0, 900, 79)
        bmi = st.number_input('BMI', 0.0, 70.0, 32.0)
        dpf = st.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.372)
        age = st.number_input('Age', 0, 100, 33)

        if st.button("Prediksi"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])
            prediction = knn.predict(input_data)
            hasil = 'Diabetes' if prediction[0] == 1 else 'Tidak Diabetes'
            st.success(f"Hasil Prediksi: **{hasil}**")

    except FileNotFoundError:
        st.error("File 'diabetes.csv' tidak ditemukan!")

#==================================
#=========== CLUSTERING ==========
#==================================
elif menu == "Clustering":
    st.header("Clustering Lokasi Gerai Kopi dengan KMeans")

    try:
        df = pd.read_csv('lokasi_gerai_kopi.csv')

        # Judul & Deskripsi
        st.subheader("Deskripsi Proyek")
        st.write("""
        Aplikasi ini menggunakan algoritma KMeans untuk mengelompokkan lokasi gerai kopi berdasarkan koordinat (x, y).
        Tujuan dari clustering ini adalah untuk menemukan area padat gerai, sebagai pertimbangan strategi lokasi.
        Aplikasi ini menampilkan hasil clustering data uji serta fitur input lokasi baru untuk menentukan termasuk cluster mana.
        """)

        # Clustering Model
        X = df[['x', 'y']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)

        # Visualisasi Hasil Clustering
        st.subheader("Visualisasi Hasil Clustering pada Data Testing")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='Set1', ax=ax)
        ax.set_title("Hasil Clustering Lokasi Gerai (KMeans)")
        st.pyplot(fig)

        # Input Lokasi Baru
        st.subheader("Input Lokasi Baru")
        x_new = st.number_input('x', 0.0, 200.0, 50.0)
        y_new = st.number_input('y', 0.0, 200.0, 50.0)

        if st.button("Clusterkan"):
            new_cluster = kmeans.predict(np.array([[x_new, y_new]]))[0]
            st.success(f"Lokasi baru termasuk dalam **cluster {new_cluster}**")

    except FileNotFoundError:
        st.error("File 'lokasi_gerai_kopi.csv' tidak ditemukan!")
