import pandas as pd

# Chemin vers le fichier de données
DATA_SAVE_PATH = "data/minimax_data.csv"

# Charger les données depuis le fichier CSV
data = pd.read_csv(DATA_SAVE_PATH)

# Supprimer les duplications de lignes
# La méthode drop_duplicates() retire les doublons en tenant compte de toutes les colonnes par défaut
data_cleaned = data.drop_duplicates()

# Réécrire les données nettoyées dans le même fichier ou un nouveau fichier si tu veux conserver une version originale
data_cleaned.to_csv(DATA_SAVE_PATH, index=False)

print(f"Duplications supprimées. Nombre total de lignes après nettoyage : {len(data_cleaned)}")