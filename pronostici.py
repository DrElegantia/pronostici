import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from datetime import datetime

# Configurazione della pagina
st.set_page_config(page_title="Pronostici Serie A con il Machine Learning")
st.title('Pronostici partite di Serie A con il Machine Learning')
st.header('Pronostici partite di Serie A per la stagione in corso')
st.write(
    "Benvenuto all'applicazione Pronostici partite di Serie A con il Machine Learning. "
    "Questa applicazione utilizza modelli di Machine Learning per predire gli esiti delle partite di calcio della Serie A italiana. "
    "Scorri verso il basso per vedere le previsioni delle prossime partite e la classifica prevista."
)
np.random.seed(2)


# Funzione per convertire le date
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
    for year in range(5, 25):
        url = f'https://www.football-data.co.uk/mmz4281/{year:02d}{year + 1:02d}/I1.csv'
        df_name = f'df{year:02d}'
        globals()[df_name] = pd.read_csv(url)
        dataframes[df_name] = globals()[df_name]
    return dataframes


dataframes = load_dataframes()

# Lista dei DataFrame (in ordine decrescente per anno)
dfs = [df24, df23, df22, df21, df20, df19, df18, df17, df16, df15, df14, df13, df12, df11, df10, df09, df08, df07, df06,
       df05]

# Elaborazione di ciascun DataFrame per calcolare tutte le feature
for df in dfs:
    # Conversione della data e ordinamento
    df['Date'] = df['Date'].apply(converti_data)
    df.sort_values(by='Date', inplace=True)

    # STATISTICHE SUI GOAL
    df['x_FTHG'] = df.groupby('HomeTeam')['FTHG'].cumsum() - df['FTHG']
    df['x_FTAG'] = df.groupby('AwayTeam')['FTAG'].cumsum() - df['FTAG']
    df['x_FTHGS'] = df.groupby('HomeTeam')['FTAG'].cumsum() - df['FTAG']
    df['x_FTAGS'] = df.groupby('AwayTeam')['FTHG'].cumsum() - df['FTHG']

    # STATISTICHE ROLLING (ultime 3 partite) usando transform per mantenere l'indice
    df['x_FTHG_R'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df[
        'FTHG']
    df['x_FTAG_R'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df[
        'FTAG']
    df['x_FTHGS_R'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df[
        'FTAG']
    df['x_FTAGS_R'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df[
        'FTHG']

    # STATISTICHE DEGLI EVENTI DI GIOCO
    df['x_HS'] = df.groupby('HomeTeam')['HS'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df['HS']
    df['x_AS'] = df.groupby('AwayTeam')['AS'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df['AS']
    df['x_HST'] = df.groupby('HomeTeam')['HST'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df[
        'HST']
    df['x_AST'] = df.groupby('AwayTeam')['AST'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df[
        'AST']
    df['x_HC'] = df.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df['HC']
    df['x_AC'] = df.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df['AC']
    df['x_HF'] = df.groupby('HomeTeam')['HF'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df['HF']
    df['x_AF'] = df.groupby('AwayTeam')['AF'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - df['AF']

    # VARIABILI CUMULATIVE PER IL CONTEGGIO DELLE PARTITE
    df['conta'] = 1
    df['PGH'] = df.groupby('HomeTeam')['conta'].cumsum() - df['conta']
    df['PGA'] = df.groupby('AwayTeam')['conta'].cumsum() - df['conta']

    # Creazione delle colonne dummy per ogni esito (ad es. H_home, A_home, ecc.)
    lettere_univoche = df['FTR'].unique()
    for lettera in lettere_univoche:
        df[f'{lettera}_home'] = (df['FTR'] == lettera).astype(int)
    for lettera in lettere_univoche:
        df[f'{lettera}_away'] = (df['FTR'] == lettera).astype(int)

    # Ulteriori statistiche (ultime 3 partite per sconfitte, pareggi, vittorie)
    df['Sconfitte_c'] = df.groupby('HomeTeam')['A_home'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - \
                        df['A_home']
    df['Pareggi_c'] = df.groupby('HomeTeam')['D_home'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - \
                      df['D_home']
    df['Vittorie_c'] = df.groupby('HomeTeam')['H_home'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - \
                       df['H_home']
    df['Pareggi_f'] = df.groupby('AwayTeam')['D_away'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - \
                      df['D_away']
    df['Vittorie_f'] = df.groupby('AwayTeam')['A_away'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - \
                       df['A_away']
    df['Sconfitte_f'] = df.groupby('AwayTeam')['H_away'].transform(lambda x: x.rolling(window=3, min_periods=1).sum()) - \
                        df['H_away']

    # Punti cumulativi
    mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
    df['PH'] = df['FTR'].map(mappatura_valori_h)
    df['PH'] = df.groupby('HomeTeam')['PH'].transform(lambda x: x.rolling(window=6, min_periods=1).sum()) - df['PH']
    mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
    df['PA'] = df['FTR'].map(mappatura_valori_a)
    df['PA'] = df.groupby('AwayTeam')['PA'].transform(lambda x: x.rolling(window=6, min_periods=1).sum()) - df['PA']

# Unione di tutti i DataFrame in uno unico
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.dropna(subset=['AwayTeam'])

# Sostituisci i NaN con 0 (o usa dropna() se preferisci eliminare le righe con NaN)
merged_df = merged_df.fillna(0)


# Preparazione della variabile target e delle codifiche one-hot per i nomi delle squadre
y = merged_df['FTR']
X = merged_df[['HomeTeam', 'AwayTeam']]

X_encoded_home = pd.get_dummies(X['HomeTeam'], prefix='Home')
X_encoded_away = pd.get_dummies(X['AwayTeam'], prefix='Away')

# Elenco delle feature aggiuntive da includere
additional_features = [
    'x_FTHG', 'x_FTAG', 'x_FTHGS', 'x_FTAGS',
    'x_FTHG_R', 'x_FTAG_R', 'x_FTHGS_R', 'x_FTAGS_R',
    'x_HS', 'x_AS', 'x_HST', 'x_AST',
    'x_HC', 'x_AC', 'x_HF', 'x_AF',
    'conta', 'PGH', 'PGA',
    'Sconfitte_c', 'Pareggi_c', 'Vittorie_c',
    'Pareggi_f', 'Vittorie_f', 'Sconfitte_f',
    'PH', 'PA'
]

# Identifica le colonne dummy relative agli esiti (ad esempio: H_home, D_home, A_home e relative per away)
lettere_univoche = merged_df['FTR'].unique()
dummy_home_cols = [f"{letter}_home" for letter in lettere_univoche]
dummy_away_cols = [f"{letter}_away" for letter in lettere_univoche]

# Costruzione del dataset finale includendo:
# - Le codifiche one-hot per le squadre
# - Le feature aggiuntive calcolate
# - Le colonne dummy per ogni esito
X_final = pd.concat([
    X_encoded_home,
    X_encoded_away,
    merged_df[additional_features],
    merged_df[dummy_home_cols],
    merged_df[dummy_away_cols]
], axis=1)

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Addestramento del modello
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Valutazione sui dati di test
p_test = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, p_test)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot della matrice di confusione
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Vittoria fuoricasa", "Pareggio", "Vittoria casa"],
            yticklabels=["Vittoria fuoricasa", "Pareggio", "Vittoria casa"])
plt.xlabel('Predizione')
plt.ylabel('Reale')
plt.title('Matrice di Confusione')
st.pyplot(fig)
# --- Seconda parte: Forecast e allineamento delle feature ---

st.markdown(
    """
    **Interpretazione della Matrice di Confusione:**

    Ogni cella della matrice rappresenta una combinazione specifica di previsioni e risultati reali. 
    Gli elementi sulla diagonale principale rappresentano le previsioni corrette. 
    Gli elementi al di fuori della diagonale principale rappresentano gli errori di previsione.

    Ad esempio:

    **Vittoria squadra casa (Predizione) - Vittoria squadra casa (Reale):** Questa cella mostra la percentuale di partite in cui il modello ha correttamente previsto che la squadra di casa avrebbe vinto, e che effettivamente ha vinto.
    """
)

# Recupera i dati dal sito fbref
url = 'https://fbref.com/it/comp/11/calendario/Risultati-e-partite-di-Serie-A'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Trova la tabella e crea il DataFrame
data = []
for row in soup.find_all('tr'):
    cols = row.find_all(['td', 'th'])
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df_forecast = pd.DataFrame(data[1:], columns=data[0])



# Converte la colonna 'Data' in datetime e filtra le partite future
df_forecast['Data'] = pd.to_datetime(df_forecast['Data'], format='%d-%m-%Y')
data_attuale = datetime.now()
df_forecast = df_forecast[df_forecast['Data'] > data_attuale]

# Rinomina le colonne e seleziona quelle utili
df_forecast = df_forecast.rename(columns={'Casa': 'HomeTeam', 'Ospiti': 'AwayTeam'})
df_forecast = df_forecast[['Data', 'HomeTeam', 'AwayTeam']]
df_forecast['HomeTeam'] = df_forecast['HomeTeam'].replace('Hellas Verona', 'Verona')
df_forecast['AwayTeam'] = df_forecast['AwayTeam'].replace('Hellas Verona', 'Verona')
df_forecast.reset_index(drop=True, inplace=True)

# Codifiche one-hot per le squadre, allineate con quelle usate nel training
X_forecast_base = df_forecast[['HomeTeam', 'AwayTeam']]
X_encoded_home_forecast = pd.get_dummies(X_forecast_base['HomeTeam'], prefix='Home')
X_encoded_away_forecast = pd.get_dummies(X_forecast_base['AwayTeam'], prefix='Away')
X_encoded_home_forecast = X_encoded_home_forecast.reindex(columns=X_encoded_home.columns, fill_value=0)
X_encoded_away_forecast = X_encoded_away_forecast.reindex(columns=X_encoded_away.columns, fill_value=0)

df_forecast.sort_values(by='Data', inplace=True)

# Elenco delle feature aggiuntive come definite nel training
additional_features = ['x_FTHG', 'x_FTAG', 'x_FTHGS', 'x_FTAGS',
                       'x_FTHG_R', 'x_FTAG_R', 'x_FTHGS_R', 'x_FTAGS_R',
                       'x_HS', 'x_AS', 'x_HST', 'x_AST',
                       'x_HC', 'x_AC', 'x_HF', 'x_AF',
                       'conta', 'PGH', 'PGA',
                       'Sconfitte_c', 'Pareggi_c', 'Vittorie_c',
                       'Pareggi_f', 'Vittorie_f', 'Sconfitte_f',
                       'PH', 'PA']

# Dividiamo le feature in quelle relative alla squadra di casa e quelle relative alla squadra in trasferta
home_features = ['x_FTHG', 'x_FTHGS', 'x_FTHG_R', 'x_FTHGS_R',
                 'x_HS', 'x_HST', 'x_HC', 'x_HF',
                 'conta', 'PGH', 'Sconfitte_c', 'Pareggi_c', 'Vittorie_c', 'PH']
away_features = ['x_FTAG', 'x_FTAGS', 'x_FTAG_R', 'x_FTAGS_R',
                 'x_AS', 'x_AST', 'x_AC', 'x_AF',
                 'PGA', 'Sconfitte_f', 'Pareggi_f', 'Vittorie_f', 'PA']

# Per ciascuna feature home, mappa il valore storico (dall'ultimo match disponibile in df24)
for col in home_features:
    last_values = df24.groupby('HomeTeam')[col].last()
    df_forecast[col] = df_forecast['HomeTeam'].map(last_values)

# Per ciascuna feature away, mappa il valore storico (dall'ultimo match disponibile in df24)
for col in away_features:
    last_values = df24.groupby('AwayTeam')[col].last()
    df_forecast[col] = df_forecast['AwayTeam'].map(last_values)

# Creazione delle colonne dummy per gli esiti, come nel training (non disponibili per il forecast, perciò inizializzate a zero)
lettere_univoche = df24['FTR'].unique()
dummy_home_cols = [f"{letter}_home" for letter in lettere_univoche]
dummy_away_cols = [f"{letter}_away" for letter in lettere_univoche]

for col in dummy_home_cols:
    df_forecast[col] = 0
for col in dummy_away_cols:
    df_forecast[col] = 0

# Costruzione del DataFrame finale per il forecast, con le stesse colonne usate per il training
X_forecast = pd.concat([
    X_encoded_home_forecast,
    X_encoded_away_forecast,
    df_forecast[additional_features],
    df_forecast[dummy_home_cols],
    df_forecast[dummy_away_cols]
], axis=1)

X_forecast.fillna(0, inplace=True)

# Predizione del modello sulle partite future
p_forecast = model.predict(X_forecast)
p_forecast_df = pd.DataFrame({'Prediction': p_forecast})

# Combina le previsioni con il DataFrame di forecast
forecast = pd.concat([df_forecast, p_forecast_df], axis=1)

# Calcolo della classifica finale (basata sui dati storici e le previsioni)
mappatura_valori_h = {'A': 0, 'D': 1, 'H': 3}
df24['Punti_home'] = df24['FTR'].map(mappatura_valori_h)
df24['Punti_home'] = df24.groupby('HomeTeam')['Punti_home'].cumsum()
mappatura_valori_a = {'A': 3, 'D': 1, 'H': 0}
df24['Punti_away'] = df24['FTR'].map(mappatura_valori_a)
df24['Punti_away'] = df24.groupby('AwayTeam')['Punti_away'].cumsum()
df24['Punti'] = df24['Punti_home'] + df24['Punti_away']
ultimo_punto_per_team_home = df24.groupby('HomeTeam')['Punti_home'].last()
ultimo_punto_per_team_away = df24.groupby('AwayTeam')['Punti_away'].last()

forecast['Punti_home'] = forecast['Prediction'].map(mappatura_valori_h)
forecast['Punti_home'] = forecast.groupby('HomeTeam')['Punti_home'].cumsum()
forecast['Punti_away'] = forecast['Prediction'].map(mappatura_valori_a)
forecast['Punti_away'] = forecast.groupby('AwayTeam')['Punti_away'].cumsum()
forecast['Punti'] = forecast['Punti_home'] + forecast['Punti_away']
ultimo_punto_per_team_home_forecast = forecast.groupby('HomeTeam')['Punti_home'].last()
ultimo_punto_per_team_away_forecast = forecast.groupby('AwayTeam')['Punti_away'].last()

a = pd.DataFrame((
    ultimo_punto_per_team_home_forecast + ultimo_punto_per_team_away_forecast +
    ultimo_punto_per_team_home + ultimo_punto_per_team_away
))
st.write('Pronostico Classifica finale Serie A:')
a = a.rename(columns={0: 'Punteggio'})
a = a.rename_axis('Squadra').sort_values(by='Punteggio', ascending=False)


# Visualizzazione delle prossime partite e delle relative previsioni
prossime_partite = forecast[['HomeTeam', 'AwayTeam', 'Prediction']].iloc[:10, :]
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace('A', 'Vittoria squadra fuoricasa')
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace('D', 'Pareggio')
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace('H', 'Vittoria squadra casa')
prossime_partite = prossime_partite.rename(
    columns={'HomeTeam': 'Casa', 'AwayTeam': 'Fuori Casa', 'Prediction': 'Risultato predetto dal modello'})

# Calcolo dei punti storici e forecast (già presenti)
punti_storici = ultimo_punto_per_team_home + ultimo_punto_per_team_away
punti_forecast = ultimo_punto_per_team_home_forecast + ultimo_punto_per_team_away_forecast

# Allineiamo gli indici su tutte le squadre
tutte_le_squadre = sorted(set(punti_storici.index).union(set(punti_forecast.index)))
punti_storici = punti_storici.reindex(tutte_le_squadre, fill_value=0)
punti_forecast = punti_forecast.reindex(tutte_le_squadre, fill_value=0)
punti_totali = punti_storici + punti_forecast

# --- Calcolo dei risultati storici ---
# Per le partite giocate, usiamo df24 (o l'ultimo dataset disponibile) e la colonna FTR.
# Vittorie storiche: quando una squadra in casa ha vinto (FTR=='H') oppure in trasferta ha vinto (FTR=='A')
hist_home_wins = df24[df24['FTR'] == 'H'].groupby('HomeTeam').size()
hist_away_wins = df24[df24['FTR'] == 'A'].groupby('AwayTeam').size()
hist_wins = hist_home_wins.add(hist_away_wins, fill_value=0)

# Pareggi storici: partite in cui FTR == 'D'
hist_home_draws = df24[df24['FTR'] == 'D'].groupby('HomeTeam').size()
hist_away_draws = df24[df24['FTR'] == 'D'].groupby('AwayTeam').size()
hist_draws = hist_home_draws.add(hist_away_draws, fill_value=0)

# Sconfitte storiche: per la squadra di casa sono le partite perse (FTR=='A'), per quella in trasferta quelle perse (FTR=='H')
hist_home_losses = df24[df24['FTR'] == 'A'].groupby('HomeTeam').size()
hist_away_losses = df24[df24['FTR'] == 'H'].groupby('AwayTeam').size()
hist_losses = hist_home_losses.add(hist_away_losses, fill_value=0)

# --- Calcolo dei risultati forecast ---
# Nel forecast, la colonna "Prediction" contiene la previsione ('H', 'D' o 'A')
# Vittorie forecast: in casa se Prediction=='H', in trasferta se Prediction=='A'
fore_home_wins = forecast[forecast['Prediction'] == 'H'].groupby('HomeTeam').size()
fore_away_wins = forecast[forecast['Prediction'] == 'A'].groupby('AwayTeam').size()
fore_wins = fore_home_wins.add(fore_away_wins, fill_value=0)

# Pareggi forecast: per entrambe le situazioni se Prediction=='D'
fore_home_draws = forecast[forecast['Prediction'] == 'D'].groupby('HomeTeam').size()
fore_away_draws = forecast[forecast['Prediction'] == 'D'].groupby('AwayTeam').size()
fore_draws = fore_home_draws.add(fore_away_draws, fill_value=0)

# Sconfitte forecast: in casa se Prediction=='A' (perdita in casa), in trasferta se Prediction=='H' (perdita in trasferta)
fore_home_losses = forecast[forecast['Prediction'] == 'A'].groupby('HomeTeam').size()
fore_away_losses = forecast[forecast['Prediction'] == 'H'].groupby('AwayTeam').size()
fore_losses = fore_home_losses.add(fore_away_losses, fill_value=0)

# Allineamento degli indici per tutti i conteggi
all_teams = sorted(set(hist_wins.index)
                   .union(hist_draws.index)
                   .union(hist_losses.index)
                   .union(fore_wins.index)
                   .union(fore_draws.index)
                   .union(fore_losses.index))

hist_wins = hist_wins.reindex(all_teams, fill_value=0)
hist_draws = hist_draws.reindex(all_teams, fill_value=0)
hist_losses = hist_losses.reindex(all_teams, fill_value=0)
fore_wins = fore_wins.reindex(all_teams, fill_value=0)
fore_draws = fore_draws.reindex(all_teams, fill_value=0)
fore_losses = fore_losses.reindex(all_teams, fill_value=0)

# Totali risultati: somma di quelli storici e forecast
total_wins = hist_wins + fore_wins
total_draws = hist_draws + fore_draws
total_losses = hist_losses + fore_losses

# --- Creazione del DataFrame finale dettagliato ---
classifica_dettagliata = pd.DataFrame({
    'Punti storici': punti_storici.reindex(all_teams, fill_value=0),
    'Punti forecast': punti_forecast.reindex(all_teams, fill_value=0),
    'Totale punti': punti_totali.reindex(all_teams, fill_value=0),
    'Vittorie storiche': hist_wins,
    'Pareggi storici': hist_draws,
    'Sconfitte storiche': hist_losses,
    'Vittorie forecast': fore_wins,
    'Pareggi forecast': fore_draws,
    'Sconfitte forecast': fore_losses,
    'Totale Vittorie': total_wins,
    'Totale Pareggi': total_draws,
    'Totale Sconfitte': total_losses
})

# Calcola la posizione in classifica in base ai punti totali
classifica_dettagliata['Posizione'] = classifica_dettagliata['Totale punti'].rank(method='min', ascending=False).astype(int)
classifica_dettagliata = classifica_dettagliata.sort_values(by='Totale punti', ascending=False)

# Arrotonda e converte in interi le colonne numeriche della classifica dettagliata
classifica_dettagliata = classifica_dettagliata.round(0).astype(int)

st.write("### Pronostico Classifica Finale Dettagliata Serie A")
st.table(classifica_dettagliata)


# Visualizzazione delle prossime partite e delle relative previsioni
prossime_partite = forecast[['HomeTeam', 'AwayTeam', 'Prediction']].iloc[:10, :].copy()
prossime_partite['Prediction'] = prossime_partite['Prediction'].replace({'A': 'Vittoria squadra fuoricasa',
                                                                         'D': 'Pareggio',
                                                                         'H': 'Vittoria squadra casa'})
prossime_partite = prossime_partite.rename(
    columns={'HomeTeam': 'Casa', 'AwayTeam': 'Fuori Casa', 'Prediction': 'Risultato predetto dal modello'})
st.table(prossime_partite)

# Elenco di tutte le squadre (unione delle squadre di casa e in trasferta presenti in df24)
all_teams = sorted(set(df24['HomeTeam'].unique()).union(set(df24['AwayTeam'].unique())))

# Funzione per calcolare il trend storico dei punti per ciascuna squadra
def compute_historical_trend(df, teams):
    records = []
    for team in teams:
        # Filtra i match in cui la squadra ha giocato (sia in casa che in trasferta) e ordina per data
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date')
        total = 0
        for idx, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                pts = mappatura_valori_h.get(row['FTR'], 0)
            elif row['AwayTeam'] == team:
                pts = mappatura_valori_a.get(row['FTR'], 0)
            total += pts
            records.append({'Team': team, 'Date': row['Date'], 'CumulativePoints': total, 'Type': 'storico'})
    return pd.DataFrame(records)

# Funzione per calcolare il trend forecast (i punti previsti aggiunti a quelli storici)
def compute_forecast_trend(forecast_df, historical_points, teams):
    records = []
    for team in teams:
        # Seleziona i match di forecast in cui la squadra è coinvolta e ordina per data
        team_forecast = forecast_df[(forecast_df['HomeTeam'] == team) | (forecast_df['AwayTeam'] == team)].sort_values('Data')
        total = historical_points.get(team, 0)
        for idx, row in team_forecast.iterrows():
            if row['HomeTeam'] == team:
                pts = mappatura_valori_h.get(row['Prediction'], 0)
            elif row['AwayTeam'] == team:
                pts = mappatura_valori_a.get(row['Prediction'], 0)
            total += pts
            records.append({'Team': team, 'Date': row['Data'], 'CumulativePoints': total, 'Type': 'forecast'})
    return pd.DataFrame(records)

# Calcola il trend storico dei punti
df_trend_hist = compute_historical_trend(df24, all_teams)

# Calcola i punti finali storici per ogni squadra (base per il forecast)
punti_storici = {}
for team in all_teams:
    team_matches = df_trend_hist[df_trend_hist['Team'] == team]
    if not team_matches.empty:
        punti_storici[team] = team_matches.iloc[-1]['CumulativePoints']
    else:
        punti_storici[team] = 0

# Calcola il trend forecast
df_trend_forecast = compute_forecast_trend(forecast, punti_storici, all_teams)

# Unisce i dati storici e forecast in un unico DataFrame
df_trend = pd.concat([df_trend_hist, df_trend_forecast])
df_trend.sort_values(by='Date', inplace=True)

fig = go.Figure()

for team in all_teams:
    team_data = df_trend[df_trend['Team'] == team]
    if team_data.empty:
        continue
    # Dati storici (linea continua)
    storico = team_data[team_data['Type'] == 'storico']
    if not storico.empty:
        fig.add_trace(go.Scatter(
            x=storico['Date'],
            y=storico['CumulativePoints'],
            mode='lines+markers',
            name=f"{team} (storico)",
            line=dict(dash='solid')
        ))
    # Dati forecast (linea tratteggiata)
    forecast_data = team_data[team_data['Type'] == 'forecast']
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['CumulativePoints'],
            mode='lines+markers',
            name=f"{team} (forecast)",
            line=dict(dash='dash')
        ))

fig.update_layout(
    title="Andamento punti per ogni squadra",
    xaxis_title="Data",
    yaxis_title="Punti cumulativi",
    legend_title="Legenda",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Numero di simulazioni Monte Carlo
n_sim = 1000

# Assicurati che il DataFrame forecast sia ordinato per data
forecast_sorted = forecast.sort_values(by='Data')
unique_dates = forecast_sorted['Data'].unique()

# Precalcola le probabilità per le partite forecast
probs = model.predict_proba(X_forecast)
class_order = model.classes_  # ad esempio: array(['A', 'D', 'H'], dtype=object)

# Partiamo dai punti storici finali (già calcolati in precedenza, ad esempio in 'punti_storici')

historical_points = punti_storici

# Mappature dei punti in base all'esito


# Lista per salvare i DataFrame delle simulazioni
sim_results = []

sim_results = []
for sim in range(n_sim):
    current_points = historical_points.copy()  # Partenza dalla classifica storica
    sim_records = []
    for d in unique_dates:
        matches_on_date = forecast_sorted[forecast_sorted['Data'] == d]
        for idx, match in matches_on_date.iterrows():
            # Simula l'esito della partita con le probabilità ottenute
            p = probs[idx]
            outcome = np.random.choice(class_order, p=p)
            # Aggiorna i punti per la squadra di casa
            home_team = match['HomeTeam']
            current_points[home_team] = current_points.get(home_team, 0) + mappatura_valori_h.get(outcome, 0)
            # Aggiorna i punti per la squadra in trasferta
            away_team = match['AwayTeam']
            current_points[away_team] = current_points.get(away_team, 0) + mappatura_valori_a.get(outcome, 0)
        # Registra, per ogni squadra, i punti cumulativi alla data d con l'id della simulazione
        for team, pts in current_points.items():
            sim_records.append({'Team': team, 'Date': d, 'SimPoints': pts, 'SimID': sim})
    sim_results.append(pd.DataFrame(sim_records))

# Combina tutte le simulazioni in un unico DataFrame
all_sim_data = pd.concat(sim_results, ignore_index=True)

# --- Determina la data finale delle partite previste ---
final_date = forecast_sorted['Data'].max()

# Filtra i dati simulati per l'ultima giornata
final_sim = all_sim_data[all_sim_data['Date'] == final_date]

# Per ogni simulazione, determina la squadra campione (ossia quella con il punteggio massimo)
def get_champion(df):
    # Se ci sono eventuali parità, idxmax ne restituisce uno in modo arbitrario
    return df.loc[df['SimPoints'].idxmax()]

champions = final_sim.groupby('SimID').apply(get_champion).reset_index(drop=True)

# Conta quante simulazioni hanno visto ciascuna squadra campione
champion_counts = champions['Team'].value_counts().reset_index()
champion_counts.columns = ['Team', 'ChampionSimulations']
champion_counts['Percentuale'] = (champion_counts['ChampionSimulations'] / n_sim) * 100

# Ordina la classifica in base al numero di simulazioni vinte
champion_counts = champion_counts.sort_values(by='ChampionSimulations', ascending=False)

st.write("### Numero di Simulazioni in cui ciascuna squadra è campione")
st.table(champion_counts)

# Calcola la mediana dei punti cumulativi per ogni squadra e per ogni data
median_results = all_sim_data.groupby(['Team', 'Date'])['SimPoints'].median().reset_index()


fig = go.Figure()

teams = sorted(median_results['Team'].unique())
for team in teams:
    team_data = median_results[median_results['Team'] == team]
    fig.add_trace(go.Scatter(
        x=team_data['Date'],
        y=team_data['SimPoints'],
        mode='lines+markers',
        name=team
    ))

fig.update_layout(
    title="Evoluzione mediana dei punti per ogni squadra (Simulazione Monte Carlo)",
    xaxis_title="Data",
    yaxis_title="Punti cumulativi mediani",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

final_date = forecast_sorted['Data'].max()

# Filtra i dati simulati per la data finale
final_sim = all_sim_data[all_sim_data['Date'] == final_date]


final_stats = final_sim.groupby('Team')['SimPoints'].agg(['median', 'mean', 'std']).reset_index()
final_stats = final_stats.rename(columns={'median': 'Mediana', 'mean': 'Media', 'std': 'Deviazione'})
final_stats = final_stats.sort_values(by='Mediana', ascending=False)


final_stats['Posizione'] = range(1, len(final_stats) + 1)

final_stats = final_stats[['Posizione', 'Team', 'Mediana', 'Media', 'Deviazione']]

st.write("### Classifica Finale Monte Carlo")
st.dataframe(final_stats)

# Creazione di una tabella per ogni giornata con risultato effettivo e previsto

# Estrai le colonne di interesse
giornate = df_forecast[['Data', 'HomeTeam', 'AwayTeam']].copy()

# Aggiungi la previsione del modello
giornate['Risultato previsto'] = forecast['Prediction'].replace({
    'A': 'Vittoria fuori casa',
    'D': 'Pareggio',
    'H': 'Vittoria casa'
})

# Risultato effettivo
giornate['Risultato effettivo'] = 'N/A'  # Sostituirai questi valori con i risultati effettivi quando disponibili

# Popola la colonna 'Risultato effettivo' con i risultati reali storici
for i, row in giornate.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    # Verifica il risultato effettivo usando df24 (o il dataset che contiene i risultati storici)
    match_result = df24[((df24['HomeTeam'] == home_team) & (df24['AwayTeam'] == away_team)) |
                        ((df24['HomeTeam'] == away_team) & (df24['AwayTeam'] == home_team))]

    if not match_result.empty:
        ftr = match_result['FTR'].iloc[0]  # Ottieni il risultato finale (FTR: H, D, A)
        if ftr == 'H':
            giornate.at[i, 'Risultato effettivo'] = 'Vittoria casa'
        elif ftr == 'A':
            giornate.at[i, 'Risultato effettivo'] = 'Vittoria fuori casa'
        else:
            giornate.at[i, 'Risultato effettivo'] = 'Pareggio'

# Visualizzazione della tabella
st.write("### Risultato per Giornata e Pronostico")
st.table(giornate)

# Calibrazione
target_class = 'H'

# Converte y_test in un vettore binario: 1 se l'esito è target_class, altrimenti 0
y_test_binary = (y_test == target_class).astype(int)

# Estrai le probabilità per la classe target dal modello
target_index = list(model.classes_).index(target_class)
probs_target = model.predict_proba(X_test)[:, target_index]

# Usa il vettore binario per la calibrazione
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test_binary, probs_target, n_bins=10, strategy='quantile'
)

# Crea il grafico della curva di calibrazione
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Curva di calibrazione")
ax.plot([0, 1], [0, 1], "k:", label="Calibrazione perfetta")
ax.set_xlabel("Valore medio previsto")
ax.set_ylabel("Frazione di positivi")
ax.set_title("Curva di calibrazione per la classe '" + target_class + "'")
ax.legend()
st.pyplot(fig)

# Aggiunta degli elementi SEO e dei link
st.write("""
    <meta name="description" content="MOdello di Machine Learning per la predizione dei risultati delle partite di Serie A">
    <meta name="keywords" content="Serie A, predizioni, pronostici, Umberto Bertonelli, drelegantia, python">
    <meta name="author" content="Umberto Bertonelli">
    <link rel="canonical" href="https://umbertobertonelli.it">
""", unsafe_allow_html=True)

st.write("Questa app è stata creata da [Umberto Bertonelli](https://umbertobertonelli.it).")
st.write("La documentazione completa è disponibile [qui](https://github.com/DrElegantia/pronostici/tree/main).")

