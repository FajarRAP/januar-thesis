import numpy as np
import matplotlib.pyplot as plt

# Class untuk menghitung metrik evaluasi seperti Akurasi, Presisi, Recall, dan F1-Score
class MetricsEvaluator:
    def __init__(self, y_pred, y_test):
        # Menghitung True Positive, True Negative, False Positive, dan False Negative
        self.true_positive = np.sum((y_pred == 1) & (y_test == 1))
        self.true_negative = np.sum((y_pred == 0) & (y_test == 0))
        self.false_positive = np.sum((y_pred == 1) & (y_test == 0))
        self.false_negative = np.sum((y_pred == 0) & (y_test == 1))

    def confusion_matrix(self):
        return np.array([
            [self.true_negative, self.false_positive],
            [self.false_negative, self.true_positive]
        ])

    def accuracy(self):
        divisor = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return (self.true_positive + self.true_negative) / divisor if divisor > 0 else 0

    def precision(self):
        divisor = self.true_positive + self.false_positive
        return self.true_positive / divisor if divisor > 0 else 0

    def recall(self):
        divisor = self.true_positive + self.false_negative
        return self.true_positive / divisor if divisor > 0 else 0

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        divisor = precision + recall
        return 2 * (precision * recall) / divisor if divisor > 0 else 0

    def plot_confusion_matrix(self, title='Confusion Matrix'):
        # 1. Ambil nilai matriks dari class buatan Anda
        cm = self.confusion_matrix()

        # 2. Buat kanvas gambarnya
        fig, ax = plt.subplots(figsize=(6, 5))

        # 3. Warnai matriks menggunakan colormap (contoh: warna Biru)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)

        # Tambahkan bar warna di sebelah kanan untuk panduan skala
        plt.colorbar(cax)

        # 4. Tulis angka di tengah-tengah setiap kotak
        # Looping baris (i) dan kolom (j)
        batas_warna = cm.max() / 2  # Penentu kapan teks harus putih/hitam agar terbaca
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                warna_teks = "white" if cm[i, j] > batas_warna else "black"
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center',
                        size='xx-large', color=warna_teks)

        # 5. Atur Label dan Sumbu
        # Sumbu X (Tebakan / Predicted)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negatif (0)', 'Positif (1)'])
        ax.xaxis.set_ticks_position('bottom') # Pindahkan label sumbu X ke bawah

        # Sumbu Y (Asli / Actual)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Negatif (0)', 'Positif (1)'])

        # Beri judul
        plt.xlabel('Tebakan Model (Predicted)', fontsize=12)
        plt.ylabel('Nilai Asli (Actual)', fontsize=12)
        plt.title(title, fontsize=14, pad=20)

        return fig