import streamlit as st 
import matplotlib.pyplot as plt 
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error , accuracy_score , confusion_matrix , ConfusionMatrixDisplay , classification_report
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
import warnings
from sklearn.exceptions import DataConversionWarning

# Menonaktifkan peringatan feature names
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.title('Clasifikasi Gizi Ibu Hamil')
optm = option_menu('Main_menu' , ['Klasifikasi', 'Latih_model', 'Visualisasi'] , 
                   orientation='horizontal')

df = pd.read_excel('databumil_clean.xlsx')
# Hapus baris yang mengandung missing values
df = df.dropna()

# Global 

# Global 

df_m = df[['BB', 'LILA', 'Hemoglobin', 'IMT', 'nutrition_label']]   

# normalisasi Data dan rubah Label
x = df_m.drop(columns=['nutrition_label'])
y = df_m['nutrition_label']
ss = StandardScaler().fit(x)
le = LabelEncoder().fit(y)
X = ss.transform(x)
Y = le.transform(y)

# Split data 
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.3 , random_state=123)
# Gausian NB
GNB = GaussianNB().fit(x_train , y_train)
prediksi_raw = GNB.predict(x_test)
prediksi = le.inverse_transform(prediksi_raw)
# KNN 
KNN = KNeighborsClassifier().fit(x_train , y_train)
prediksi_raw1 = KNN.predict(x_test)
prediksi1 = le.inverse_transform(prediksi_raw1)
        
if optm == 'Latih_model' : 
    
    #Load data nya
    df_m = df[['BB', 'LILA', 'Hemoglobin', 'IMT', 'nutrition_label']]   
    st.header('Dataset Klasifikasi')
    st.table(df_m.head())
    
    # normalisasi Data dan rubah Label
    x = df_m.drop(columns=['nutrition_label'])
    y = df_m['nutrition_label']
    ss = StandardScaler().fit(x)
    le = LabelEncoder().fit(y)
    X = ss.transform(x)
    Y = le.transform(y)
    
    # Input Paramater
    test_size = float(st.number_input('Test size'))
    butts = st.button('Latih!')
    if butts : 
        # Split data 
        x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=test_size , random_state=123)

        # Gausian NB

        GNB = GaussianNB().fit(x_train , y_train)
        prediksi_raw = GNB.predict(x_test)
        prediksi = le.inverse_transform(prediksi_raw)

        # KNN 
        KNN = KNeighborsClassifier(n_neighbors=5).fit(x_train , y_train)
        prediksi_raw1 = KNN.predict(x_test)
        prediksi1 = le.inverse_transform(prediksi_raw1)


        # Hasil 1 
        akurasi_NB = accuracy_score(y_test , prediksi_raw)
        loss_Nb = mean_squared_error(y_test , prediksi_raw)

        # Hasil 2
        akurasi_KNN = accuracy_score(y_test , prediksi_raw1)
        loss_KNN = mean_squared_error(y_test , prediksi_raw1)

        #tabels 
        st.header('Hasil Klasifikasi')
        akur = [akurasi_NB , akurasi_KNN , (akurasi_NB - akurasi_KNN) * -1]
        loss = [loss_Nb , loss_KNN , (loss_Nb - loss_KNN) * -1]
        name = ['Gausian Naive Bayes' , 'K Nearst Neigbore Clasifer' , 'Selisih']
        dataf = pd.DataFrame(data={
            'name' : name,
            'akurasi' : akur,
            'Mean Square Error' : loss
        })
        st.table(dataf)

        # hasil Visualisasi
        cols1 , cols2 = st.columns((5,5) , gap='medium')
        y_true = le.inverse_transform(y_test)
        with cols1 : 
            st.header('Gausian Naive bayes')
            
            dfport = pd.DataFrame(classification_report(y_test , prediksi_raw , output_dict=True)).transpose()
            st.dataframe(dfport)
            confuset = confusion_matrix(y_true , prediksi)
            dis = ConfusionMatrixDisplay(confuset , display_labels=['Gizi cukup'  , 'Gizi lebih' , 'Kurang gizi'])
            dis.plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        with cols2 : 
            st.header('K Nearst Neigbore Classifer')
            dfport = pd.DataFrame(classification_report(y_test , prediksi_raw1 , output_dict=True)).transpose()
            st.dataframe(dfport)
            confuset = confusion_matrix(y_true , prediksi1)
            dis = ConfusionMatrixDisplay(confuset , display_labels=['Gizi cukup'  , 'Gizi lebih' , 'Kurang gizi'])
            dis.plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

if optm == 'Klasifikasi' : 
    
    #Load data nya
    df_m = df[['BB', 'LILA', 'Hemoglobin', 'IMT', 'nutrition_label']]   
    st.write(len(df_m))     
    st.header('Dataset Klasifikasi')
    st.table(df_m.head())

    st.header('Cek Apakah Kamu Kekurangan / Kelebihan Gizi')

    BB = float(st.number_input('Berat Badan'))
    Lila = float(st.number_input('LILA'))
    HB = float(st.number_input('Hemoglobin'))
    IMT = float(st.number_input('Indeks Massa Tubuh (IMT)'))

    def categorize_LILA(lila):
        if lila < 23.5:
            return 'Kurang Gizi'
        elif 23.5 <= lila <= 33:
            return 'Gizi Cukup'
        else:
            return 'Gizi Lebih'

    cek_lila_button = st.button('Cek Kategori LILA Anda')

    if cek_lila_button:
        lila_category = categorize_LILA(Lila)
        st.write(f'Kategori LILA: {lila_category}')

    def categorize_hemoglobin(hb):
        if hb <= 8:
            return 'Anemia Berat'
        elif 8 < hb <= 10.9:
            return 'Anemia Ringan'
        else:
            return 'Normal'

    cek_hb_button = st.button('Cek Kategori Hemoglobin Anda')

    if cek_hb_button:
        hb_category = categorize_hemoglobin(HB)
        st.write(f'Kategori Hemoglobin: {hb_category}')

    def categorize_IMT(imt):
        if imt < 18.5:
            return 'Kurus'
        elif 18.5 <= imt < 24.9:
            return 'Normal'
        elif 25 <= imt < 29.9:
            return 'Overweight'
        else:
            return 'Obesitas'    

    cek_imt_button = st.button('Cek Kategori IMT Anda')

    if cek_imt_button:
        imt_category = categorize_IMT(IMT)
        st.write(f'Kategori IMT: {imt_category}')

    Mod = st.selectbox('Pilih Model' , ['Gausian Naive Bayes' , 'K Nearst Neigbore'])

    butt = st.button('Mari Sekarang Cek Gizi Anda')
    

    if butt : 
        
        inputs = np.array([BB , Lila , HB , IMT]).reshape(1, -1)
        norms = ss.transform(inputs)
        if Mod == 'Gausian Naive Bayes' : 
            prediksi = GNB.predict(norms)
            st.write(f'Anda Mengalami {le.inverse_transform(prediksi)[0]}')
            
        else : 
            prediksi = KNN.predict(norms)
            st.write(f'Anda Mengalami {le.inverse_transform(prediksi)[0]}')
    
    
    plt.show()

if optm == 'Visualisasi':
    st.header('Visualisasi Data Alamat dan nutrition_label')

    # Count occurrences of nutrition_label per Alamat
    count_df = df.groupby(['Alamat', 'nutrition_label']).size().reset_index(name='count')

    # Plot using Plotly
    fig = px.bar(count_df, x='Alamat', y='count', color='nutrition_label',
                 title='Jumlah Nutrisi Label per Alamat',
                 labels={'Alamat': 'Alamat', 'count': 'Count', 'nutrition_label': 'Nutrition Label'})

    # Customize layout
    fig.update_layout(xaxis_title='Alamat', yaxis_title='Count', barmode='group')

    # Show plot using Streamlit
    st.plotly_chart(fig)

    st.header('Visualisasi Rata-rata BB dan TB per Alamat')

    # Calculate average BB and TB per Alamat
    avg_df = df.groupby('Alamat').agg({'BB': 'mean', 'TB': 'mean'}).reset_index()

    # Plot BB using Plotly
    fig_bb = px.bar(avg_df, x='Alamat', y='BB',
                    title='Rata-rata Berat Badan per Alamat',
                    labels={'BB': 'Rata-rata Berat Badan', 'Alamat': 'Alamat'})

    # Customize layout
    fig_bb.update_layout(xaxis_title='Alamat', yaxis_title='Rata-rata Berat Badan')

    # Show BB plot using Streamlit
    st.plotly_chart(fig_bb)

    # Plot TB using Plotly
    fig_tb = px.bar(avg_df, x='Alamat', y='TB',
                    title='Rata-rata Tinggi Badan per Alamat',
                    labels={'TB': 'Rata-rata Tinggi Badan', 'Alamat': 'Alamat'})

    # Customize layout
    fig_tb.update_layout(xaxis_title='Alamat', yaxis_title='Rata-rata Tinggi Badan')

    # Show TB plot using Streamlit
    st.plotly_chart(fig_tb)

    st.header('Visualisasi Rata-rata UK (Usia Kandungan) per Alamat')

    # Calculate average UK per Alamat
    avg_df = df.groupby('Alamat')['UK'].mean().reset_index()

    # Plot UK using Plotly
    fig_uk = px.bar(avg_df, x='Alamat', y='UK',
                    title='Rata-rata Usia Kandungan per Alamat',
                    labels={'UK': 'Rata-rata Usia Kandungan', 'Alamat': 'Alamat'})

    # Customize layout
    fig_uk.update_layout(xaxis_title='Alamat', yaxis_title='Rata-rata Usia Kandungan')

    # Show UK plot using Streamlit
    st.plotly_chart(fig_uk)

    st.header('Visualisasi Variabel Kategorikal per Alamat')

    # Variabel kategorikal yang ingin divisualisasikan
    categorical_vars = ['hemoglobin_kat', 'LILA_kat', 'IMT_kat', 'nutrition_label']

    # Membuat visualisasi untuk setiap variabel kategorikal
    for var in categorical_vars:
        st.subheader(f'Visualisasi {var} per Alamat')

        # Hitung jumlah atau proporsi setiap nilai variabel kategorikal per Alamat
        var_counts = df.groupby(['Alamat', var]).size().reset_index(name='Count')

        # Plot dengan Plotly
        fig = px.bar(var_counts, x='Alamat', y='Count', color=var,
                     title=f'Jumlah {var} per Alamat',
                     labels={'Count': 'Jumlah', 'Alamat': 'Alamat'})

        # Tampilkan plot
        st.plotly_chart(fig)
