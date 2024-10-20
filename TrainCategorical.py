import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Chemin vers le fichier de données
DATA_SAVE_PATH = "data/minimax_data.csv"

# Charger les données depuis le fichier CSV
data = pd.read_csv(DATA_SAVE_PATH)

# Mélanger les données de manière aléatoire
data_shuffled = data.sample(frac=1).reset_index(drop=True)

# Séparer les données (features) et les labels
# Supposons que la dernière colonne est l'action (le label), et les autres colonnes représentent l'état (les features)
features = data_shuffled.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière pour les features (états)
labels = data_shuffled.iloc[:, -1].values     # La dernière colonne pour les labels (actions)

# Convertir les labels en one-hot encoding (pour correspondre à la catégorical crossentropy)
num_classes = len(np.unique(labels))  # Nombre de classes (actions uniques)
labels_one_hot = to_categorical(labels, num_classes=num_classes)

# Séparer les données en ensembles d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, test_size=0.2, random_state=42)

# Normaliser les features (états)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire le modèle
model = tf.keras.Sequential()

# Ajout des couches du réseau de neurones
model.add(layers.Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Couche d'entrée + hidden layer 1
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(512, activation='relu'))  
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(num_classes, activation='softmax'))  # Couche de sortie avec softmax pour multi-classes

# Compiler le modèle avec la perte catégorielle (Categorical Crossentropy)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=64)

# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy:.4f}")

# Optionnel : Sauvegarder le modèle entraîné
model.save("Model/tetris_minimax_model.h5")

# Charger le modèle sauvegardé pour des prédictions futures
# model = tf.keras.models.load_model("tetris_minimax_model.h5")
