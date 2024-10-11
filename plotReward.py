import matplotlib.pyplot as plt
import numpy as np

# Fonction pour lire les valeurs depuis un fichier
def lire_valeurs_fichier(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        valeurs = [float(ligne.strip()) for ligne in lignes]  # Utilisation de float au lieu de int
    return valeurs

# Fonction pour lisser les valeurs avec une moyenne glissante
def lisser_valeurs(valeurs, fenetre=5):
    return np.convolve(valeurs, np.ones(fenetre)/fenetre, mode='valid')

# Chemin vers ton fichier
nom_fichier = 'rewards.log'

# Lire les valeurs du fichier
valeurs = lire_valeurs_fichier(nom_fichier)

# Générer la liste des indices (1, 2, 3, ...)
indices = range(1, len(valeurs) + 1)

fenetre = int(len(valeurs)*0.05)
# Calculer les valeurs lissées avec une fenêtre de taille 5
valeurs_lissees = lisser_valeurs(valeurs, fenetre=fenetre)

# Générer les indices pour les valeurs lissées
indices_lissees = range(fenetre, len(valeurs) + 1)

# Tracer la courbe originale
plt.plot(indices, valeurs, marker='o', label='Originale')

# Tracer la courbe lissée
plt.plot(indices_lissees, valeurs_lissees, color='red', linestyle='-', label=f'Lissée (fenêtre={fenetre})')

# Ajouter un titre et des légendes
plt.title('Courbe des valeurs avec lissage')
plt.xlabel('Numéro de la valeur (indice)')
plt.ylabel('Valeur')
plt.grid(True)
plt.legend()

# Afficher le graphique
plt.show()
