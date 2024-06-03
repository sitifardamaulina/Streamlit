import pandas as pd 

df = pd.read_csv('terpakai.csv').dropna() # Import data

def kategori_gizi(nilai):
    if nilai < 23.50:
        return "Kurang gizi"
    elif 23.50 <= nilai < 33.00:
        return "Gizi cukup"
    else:
        return "Gizi lebih"

df3 = pd.read_excel('Filebaru.xlsx')

df['lokasi'] = df3['Alamat']
df['label'] = df['LILA'].apply(kategori_gizi)

print(df)
df.to_csv('berlabel.csv')

