import cv2
import numpy as np
from Constante import *

def preprocess_frame(frame, levels=4):
    # Convertir l'image en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Redimensionner l'image à (84, 84)
    resized_frame = cv2.resize(gray_frame, (ImageSize, ImageSize), interpolation=cv2.INTER_AREA)

    # Normaliser l'image entre 0 et 1
    normalized_frame = resized_frame / 255.0

    # Quantifier l'image en niveaux de gris
    quantized_frame = np.floor(normalized_frame * levels) / levels  # Divise l'image en 'levels' nuances

    # Redimensionner la forme pour correspondre à l'entrée du CNN
    preprocessed_frame = quantized_frame.reshape(ImageSize, ImageSize, 1)
    
    return preprocessed_frame

