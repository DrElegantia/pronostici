import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import scikitplot as skplt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import streamlit as st
import seaborn as sns
from sklearn.metrics import confusion_matrix
import requests
from bs4 import BeautifulSoup
import pandas as pd

st.set_page_config(page_title="Pronostici Serie A con il Machine Learning")
st.title('Pronostici partite di Serie A con il Machine Learning')
st.header('Pronostici partite di Serie A per la stagione in corso')
st.write(
    "Benvenuto all'applicazione Pronostici partite di Serie A con il Machine Learning. "
    "Questa applicazione utilizza modelli di Machine Learning per predire gli esiti delle partite di calcio della Serie A italiana. "
    "Scorri verso il basso per vedere le previsioni delle prossime partite e la classifica prevista."
)
np.random.seed(2)

def converti_data(data_str):
    try:
        return pd.to_datetime(data_str, format='%d/%m/%Y')
    except ValueError:
        
        try:
            return pd.to_datetime(data_str, format='%d/%m/%y')
        except ValueError:
            print(f"Formato data non riconosciuto per: {data_str}")
            return None 


# Carica i dati da ciascun URL in un DataFrame
def load_dataframes():
    dataframes = {}
    for year in range(5, 24):
        url = f'https://www.football-data.co.uk/mmz4281/{year:02d}{year + 1:02d}/I1.csv'
        df_name = f'df{year:02d}'
        globals()[df_name] = pd.read_csv(url)
        dataframes[df_name] = globals()[df_name]

dataframes = load_dataframes()


dfs = [df23, df22, df21, df20, df19, df18, df17, df16, df15, df14, df13, df12, df11, df10, df09, df08, df07, df06, df05]  # Aggiungi tutti i tuoi DataFrame qui



for df in dfs:
    df['Date'] = df['Date'].apply(converti_data)
    df.sort_values(by='Date', inplace=True)
    df['x_FTHG'] = df.groupby('HomeTeam')['FTHG'].cumsum() - df['FTHG']
    df['x_FTAG'] = df.groupby('AwayTeam')['FTAG'].cumsum() - df['FTAG']
    df['x_FTHGS'] = df.groupby('HomeTeam')['FTAG'].cumsum() - df['FTAG']
    df['x_FTAGS'] = df.groupby('AwayTeam')['FTHG'].cumsum() - df['FTHG']
    df['x_FTHG_R'] = df.groupby('HomeTeam')['FTHG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['FTHG']
    df['x_FTAG_R'] = df.groupby('AwayTeam')['FTAG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['FTAG']
    df['x_FTHGS_R'] = df.groupby('HomeTeam')['FTAG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['FTAG']
    df['x_FTAGS_R'] = df.groupby('AwayTeam')['FTHG'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['FTHG']
    df['x_HS'] = df.groupby('HomeTeam')['HS'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HS']
    df['x_AS'] = df.groupby('AwayTeam')['AS'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AS']
    df['x_HST'] = df.groupby('HomeTeam')['HST'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HST']
    df['x_AST'] = df.groupby('AwayTeam')['AST'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AST']
    df['x_HC'] = df.groupby('HomeTeam')['HC'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HC']
    df['x_AC'] = df.groupby('AwayTeam')['AC'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AC']
    df['x_HF'] = df.groupby('HomeTeam')['HF'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['HF']
    df['x_AF'] = df.groupby('AwayTeam')['AF'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['AF']
    df['conta']=1
    df['PGH'] = df.groupby('HomeTeam')['conta'].cumsum() - df['conta']
    df['PGA'] = df.groupby('AwayTeam')['conta'].cumsum() - df['conta']
    # Ottieni le colonne univoche dalla colonna 'risultato'
    lettere_univoche = df['FTR'].unique()
    # Per ogni lettera unica, crea una nuova colonna nel DataFrame
    for lettera in lettere_univoche:
        df[f'{lettera}_home'] = (df['FTR'] == lettera).astype(int)

    df['Sconfitte_c'] = df.groupby('HomeTeam')['A_home'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)- df['A_home']
    df['Pareggi_c'] = df.groupby('HomeTeam')['D_home'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['D_home']
    df['Vittorie_c'] = df.groupby('HomeTeam')['H_home'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['H_home']
    for lettera in lettere_univoche:
        df[f'{lettera}_away'] = (df['FTR'] == lettera).astype(int)
    df['Pareggi_f'] = df.groupby('AwayTeam')['D_away'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['D_away']
    df['Vittorie_f'] = df.groupby('AwayTeam')['A_away'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['A_away']
    df['Sconfitte_f'] = df.groupby('AwayTeam')['H_away'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True) - df['H_away']
    mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
    df['PH'] = df['FTR'].map(mappatura_valori_h)
    df['PH']=df.groupby('HomeTeam')['PH'].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True) - df['PH']
    mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
    df['PA'] = df['FTR'].map(mappatura_valori_a)
    df['PA'] = df.groupby('AwayTeam')['PA'].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True) - df['PA']



# Unisci i DataFrame in un unico DataFrame
merged_df = pd.concat([df23,df22, df21, df20, df19, df18, df17, df16, df15, df14, df13, df12, df11, df10, df09, df08, df07, df06, df05], ignore_index=True)

merged_df = merged_df.dropna(subset=['AwayTeam'])

X = merged_df[['HomeTeam', 'AwayTeam']]
y = merged_df['FTR']




X_encoded_home = pd.get_dummies(X['HomeTeam'], prefix='Home')
X_encoded_away = pd.get_dummies(X['AwayTeam'], prefix='Away')


# Concatena i DataFrame vettorizzati
X = pd.concat([X.drop(['HomeTeam', 'AwayTeam'], axis=1), X_encoded_home, X_encoded_away, merged_df['x_FTHG'], merged_df['x_FTAG'],merged_df.Sconfitte_c, merged_df.Sconfitte_f, merged_df.Pareggi_c, merged_df.Pareggi_f, merged_df.Vittorie_c, merged_df.Vittorie_f, merged_df['PA'], merged_df['PH']], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Valutazione del modello sui dati di training
p_train = model.predict(X_train)


# Valutazione del modello sui dati di test
p_test = model.predict(X_test)

# Plot confusion matrix per i dati di test
# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, p_test)

# Normalizza la matrice di confusione
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Crea un heatmap della matrice di confusione
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Vittoria fuoricasa", "Pareggio", 'Vittoria casa'], yticklabels=["Vittoria fuoricasa", "Pareggio", 'Vittoria casa'])
plt.xlabel('Predizione')
plt.ylabel('Reale')
plt.title('Matrice di Confusione')
st.pyplot(fig)
st.markdown(
    """
    **Interpretazione della Matrice di Confusione:**

    Ogni cella della matrice rappresenta una combinazione specifica di previsioni e risultati reali. 
    Gli elementi sulla diagonale principale rappresentano le previsioni corrette. 
    Gli elementi al di fuori della diagonale principale rappresentano gli errori di previsione.
    
    Ad esempio:

    **Vittoria squadra casa (Predizione) - Vittoria squadra casa (Reale):** Questa cella mostra la percentuale di partite dove il modello ha correttamente previsto che la squadra di casa avrebbe vinto, e questa è effettivamente la situazione in cui la squadra di casa ha vinto.
    """
)



url = 'https://fbref.com/it/comp/11/calendario/Risultati-e-partite-di-Serie-A'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Trova la tabella con i dati desiderati
table = soup.find('table')

# Estrai i dati dalla tabella e crea un DataFrame
data = []
for row in table.find_all('tr'):
    cols = row.find_all(['td', 'th'])
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df_forecast= pd.DataFrame(data[1:], columns=data[0])


from datetime import datetime


# Converte la colonna 'Data' in formato datetime
df_forecast['Data'] = pd.to_datetime(df_forecast['Data'], format='%d-%m-%Y')

# Ottieni la data attuale
data_attuale = datetime.now()

# Filtra il DataFrame mantenendo solo le righe con data superiore a quella attuale
df_forecast= df_forecast[df_forecast['Data'] > data_attuale]
df_forecast = df_forecast.rename(columns={'Casa': 'HomeTeam', 'Ospiti': 'AwayTeam'})

df_forecast=df_forecast[['Data','HomeTeam', 'AwayTeam']]
df_forecast['HomeTeam'] = df_forecast['HomeTeam'].replace('Hellas Verona', 'Verona')
df_forecast['AwayTeam'] = df_forecast['AwayTeam'].replace('Hellas Verona', 'Verona')
df_forecast.reset_index(inplace=True)

X_forecast = df_forecast[['HomeTeam', 'AwayTeam']]

X_encoded_home_forecast = pd.get_dummies(X_forecast['HomeTeam'], prefix='Home')
X_encoded_away_forecast = pd.get_dummies(X_forecast['AwayTeam'], prefix='Away')
X_encoded_home_forecast = X_encoded_home_forecast.reindex(columns=X_encoded_home.columns, fill_value=0)
X_encoded_away_forecast = X_encoded_away_forecast.reindex(columns=X_encoded_away.columns, fill_value=0)


df_forecast.Data= pd.to_datetime(df_forecast.Data)
df_forecast.sort_values(by='Data', inplace=True)

# Lista delle colonne da aggiornare con gli ultimi valori
colonnes = ['x_FTHG', 'x_FTAG', 'x_FTHGS', 'x_FTAGS', 'x_FTHG_R', 'x_FTAG_R', 'x_FTHGS_R', 'x_FTAGS_R', 'x_HS','x_AS', 'x_HST', 'x_HC', 'x_AC', 'x_AST', 'x_HF', 'x_AF', 'conta', 'PGH', 'PGA', 'Sconfitte_c', 'Pareggi_c', 'Vittorie_c',
            'Pareggi_f', 'Vittorie_f', 'Sconfitte_f', 'PH', 'PA']

for col in colonnes:
    # Ottieni l'ultima riga di ciascun gruppo (squadra)
    last_values = df23.groupby('HomeTeam')[col].last()
    # Aggiorna i valori in df_forecast
    df_forecast[col] = df_forecast['HomeTeam'].map(last_values)

# Ottieni le colonne univoche dalla colonna 'risultato'
lettere_univoche = df23['FTR'].unique()

# Per ogni lettera unica, crea una nuova colonna nel DataFrame
for lettera in lettere_univoche:
    df_forecast[f'{lettera}_home'] = (df23['FTR'] == lettera).astype(int)

# Mappatura dei valori per le colonne PH e PA
mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
df_forecast['PH'] = df23['FTR'].map(mappatura_valori_h)

mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
df_forecast['PA'] = df23['FTR'].map(mappatura_valori_a)

# Aggiorna PH e PA con gli ultimi valori di df23
df_forecast['PH'] = df_forecast['HomeTeam'].map(df23.groupby('HomeTeam')['PH'].last())
df_forecast['PA'] = df_forecast['AwayTeam'].map(df23.groupby('AwayTeam')['PA'].last())




# Concatena i DataFrame vettorizzati
X_forecast = pd.concat([X_forecast.drop(['HomeTeam', 'AwayTeam'], axis=1), X_encoded_home_forecast, X_encoded_away_forecast, df_forecast['x_FTHG'], df_forecast['x_FTAG'],df_forecast.Sconfitte_c, df_forecast.Sconfitte_f, df_forecast.Pareggi_c, df_forecast.Pareggi_f, df_forecast.Vittorie_c, df_forecast.Vittorie_f, df_forecast['PA'], df_forecast['PH']], axis=1)




X_forecast.dropna(inplace=True)


p_forecast = model.predict(X_forecast)
# Converti l'array NumPy in un DataFrame o Series
p_forecast_df = pd.DataFrame({'Prediction': p_forecast})  # Sostituisci 'Prediction' con il nome desiderato

# Concatena X_forecast e p_forecast_df
forecast = pd.concat([df_forecast, p_forecast_df], axis=1)


mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
df23['Punti_home'] = df23['FTR'].map(mappatura_valori_h)
df23['Punti_home']= df23.groupby('HomeTeam')['Punti_home'].cumsum()
mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
df23['Punti_away'] = df23['FTR'].map(mappatura_valori_a)
df23['Punti_away'] = df23.groupby('AwayTeam')['Punti_away'].cumsum()
df23['Punti']=df23['Punti_home']+df23['Punti_away']
valori_unici = df23['HomeTeam'].unique()
ultimo_punto_per_team_home = df23.groupby('HomeTeam')['Punti_home'].last()
ultimo_punto_per_team_away = df23.groupby('AwayTeam')['Punti_away'].last()


mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
forecast['Punti_home'] = forecast['Prediction'].map(mappatura_valori_h)
forecast['Punti_home']= forecast.groupby('HomeTeam')['Punti_home'].cumsum()
mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
forecast['Punti_away'] = forecast['Prediction'].map(mappatura_valori_a)
forecast['Punti_away'] = forecast.groupby('AwayTeam')['Punti_away'].cumsum()
forecast['Punti']=forecast['Punti_home']+forecast['Punti_away']
valori_unici_forecast = forecast['HomeTeam'].unique()
ultimo_punto_per_team_home_forecast = forecast.groupby('HomeTeam')['Punti_home'].last()
ultimo_punto_per_team_away_forecast = forecast.groupby('AwayTeam')['Punti_away'].last()
a=pd.DataFrame((ultimo_punto_per_team_home_forecast + ultimo_punto_per_team_away_forecast+ultimo_punto_per_team_home + ultimo_punto_per_team_away))
st.write('Pronoscico Classifica finale Serie A:')
a = a.rename(columns={0: 'Punteggio'})
a = a.rename_axis('Squadra').sort_values(by='Punteggio', ascending=False)
st.table(a)



prossime_partite=forecast[['HomeTeam','AwayTeam','Prediction']].iloc[:10,:]
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace('A', 'Vittoria squadra fuoricasa')
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace('D', 'Pareggio')
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace('H', 'Vittoria squadra casa')
prossime_partite= prossime_partite.rename(columns={'HomeTeam': 'Casa','AwayTeam':'Fuori Casa','Prediction':'Risultato predetto dal modello' })
st.table(prossime_partite)

# Aggiungi elementi SEO
st.write("""
    <meta name="description" content="MOdello di Machine Learning per la predizione dei risultati delle partite di Serie A">
    <meta name="keywords" content="Serie A, predizioni, pronostici, Umberto Bertonelli, drelegantia, python">
    <meta name="author" content="Umberto Bertonelli">
    <link rel="canonical" href="https://umbertobertonelli.it">
""", unsafe_allow_html=True)

# Aggiungi il link al tuo sito
st.write("Questa app è stata creata da [Umberto Bertonelli](https://umbertobertonelli.it).")
st.write("La documentazione completa è disponibile [qui](https://github.com/DrElegantia/pronostici/tree/main).")

