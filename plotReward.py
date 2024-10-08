import matplotlib.pyplot as plt

# Fonction pour lire les valeurs depuis un fichier
def lire_valeurs_fichier(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        valeurs = [float(ligne.strip()) for ligne in lignes]  # Utilisation de float au lieu de int
    return valeurs

# Chemin vers ton fichier
nom_fichier = 'rewards.log'

# Lire les valeurs du fichier
valeurs = lire_valeurs_fichier(nom_fichier)

# Générer la liste des indices (1, 2, 3, ...)
indices = range(1, len(valeurs) + 1)

# Tracer la courbe
plt.plot(indices, valeurs, marker='o')
plt.title('Courbe des valeurs du fichier')
plt.xlabel('Numéro de la valeur (indice)')
plt.ylabel('Valeur')
plt.grid(True)
plt.show()
