import streamlit as st 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error , accuracy_score , confusion_matrix , ConfusionMatrixDisplay , classification_report
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

st.title('Clasifikasi Gizi Ibu Hamil')
optm = option_menu('Main_menu' , ['Prediksi' , 'Visualisasi Data' , 'Latih_model'] , 
                   orientation='horizontal')

df = pd.read_csv('berlabel.csv')

if optm == 'Visualisasi Data' : 
    def select(dfs:pd.DataFrame , nama_desa): 
        cluster1 = 0
        cluster2 = 0
        cluster3 = 0
        for i in range(len(dfs)): 
            if dfs['lokasi'][i] == nama_desa: 
                if dfs['label'][i] == 'Kurang gizi' : 
                    cluster1 += 1 
                elif dfs['label'][i] == 'Gizi cukup': 
                    cluster2 += 1
                else : 
                    cluster3 += 1
        return [cluster1 , cluster2 , cluster3]

    loc = []
    for lc in df['lokasi'] : 
        if lc not in loc : 
            loc.append(lc)
    print(len(loc))
    label = ['Kurang gizi' , 'Gizi cukup' , 'Gizi lebih']
    data_plot = [select(df , loc[i]) for i in range(len(loc))]
    print(len(data_plot))

    st.header('Data Gizi Ibu Hamil pada 8 daerah')

    col1 , col2 = st.columns(2)

    plt.figure(figsize=(5 , 5))
    with col1 : 
        plt.title(loc[0])
        plt.bar(label , data_plot[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.title(loc[1])
        plt.bar(label , data_plot[1])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.title(loc[2])
        plt.bar(label , data_plot[2])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.title(loc[3])
        plt.bar(label , data_plot[3])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    with col2 : 
        plt.title(loc[4])
        plt.bar(label , data_plot[4])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.title(loc[5])
        plt.bar(label , data_plot[5])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.title(loc[6])
        plt.bar(label , data_plot[6])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        plt.title(loc[7])
        plt.bar(label , data_plot[7])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
        
# Global 

df_m = df.drop(columns=['lokasi' , 'Unnamed: 0.1' , 'Unnamed: 0'], axis=-1)       

# normalisasi Data dan rubah Label
x = df_m.drop(columns=['label'])
y = df_m['label']
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
    df_m = df.drop(columns=['lokasi' , 'Unnamed: 0.1' , 'Unnamed: 0'], axis=-1)       
    st.header('perviwe Ini adalah Data Yang di pakai')
    st.table(df_m.head())
    
    # normalisasi Data dan rubah Label
    x = df_m.drop(columns=['label'])
    y = df_m['label']
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
        st.header('Hasil Prediksi')
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

if optm == 'Prediksi' : 
    
    #Load data nya
    df_m = df.drop(columns=['lokasi' , 'Unnamed: 0.1' , 'Unnamed: 0'], axis=-1)  
    st.write(len(df_m))     
    st.header('perviwe Ini adalah Data Yang di pakai')
    st.table(df_m.head())

    st.header('Cek Apakah Kamu Kekurangan / Kelebihan Gizi')
    BB = float(st.number_input('Berat Badan'))
    Lila = float(st.number_input('LILA'))
    TB = float(st.number_input('Tinggi Badan'))
    UK = float(st.number_input('Umur Kandungan'))
    Umur = float(st.number_input('Umur'))
    Mod = st.selectbox('Pilih Model' , ['Gausian Naive Bayes' , 'K Nearst Neigbore'])
    butt = st.button('Cek!')
    
    if butt : 
        
        inputs = np.array([Umur , BB , TB , UK , Lila]).reshape(1, -1)
        norms = ss.transform(inputs)
        if Mod == 'Gausian Naive Bayes' : 
            prediksi = GNB.predict(norms)
            st.write(f'Anda Mengalami {le.inverse_transform(prediksi)[0]}')
            
        else : 
            prediksi = KNN.predict(norms)
            st.write(f'Anda Mengalami {le.inverse_transform(prediksi)[0]}')
    
    
    plt.show()