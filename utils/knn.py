import numpy as np
import pandas as pd
from collections import Counter

class KNN:
  # Inisialisasi model dengan nilai default k = 3
  def __init__(self, k=3):
    self.k = k

  # Fungsi untuk melatih model (menyimpan data latih)
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    
    return self

  # Fungsi untuk memprediksi label menggunakan jarak Euclidean dan Manhattan
  def predict(self, X_test):
    return {
        'Euclidean': np.array([self._euclidean_distance(patient) for patient in X_test]),
        'Manhattan': np.array([self._manhattan_distance(patient) for patient in X_test])
    }
    
  def dump(self, X_test:pd.DataFrame, y_test:pd.DataFrame):
    self.X_test = X_test
    self.y_test = y_test
    
    return self

  # Menghitung jarak Euclidean
  def _euclidean_distance(self, patient):
    distances = np.sqrt(np.sum((self.X_train - patient) ** 2, axis=1))
    return self._most_common_label(distances)

  # Menghitung jarak Manhattan
  def _manhattan_distance(self, patient):
    distances = np.sum(abs(self.X_train - patient), axis=1)
    return self._most_common_label(distances)

  # Mengambil label terbanyak dari k-tetangga terdekat
  def _most_common_label(self, distances):
    nearest_neighbors = np.argsort(distances)[:self.k]
    nearest_neighbors_labels = self.y_train.iloc[nearest_neighbors]
    most_common_label = Counter(nearest_neighbors_labels).most_common(1)[0][0]
    return most_common_label