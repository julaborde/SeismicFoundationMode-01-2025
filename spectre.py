import seisbench.datasets as sbd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os

# Étape 1 : Charger un jeu de données de SeisBench
# Exemple avec le dataset "STEAD"
dataset = sbd.STEAD()
data = dataset.get_dataframe()

# Étape 2 : Créer un dossier pour sauvegarder les spectrogrammes
output_dir = "spectrograms"
os.makedirs(output_dir, exist_ok=True)

# Étape 3 : Transformer les signaux en spectrogrammes
def plot_spectrogram(signal, sr, save_path):
    """Crée et sauvegarde un spectrogramme d'un signal."""
    plt.figure(figsize=(10, 4))
    # Transformer en spectrogramme
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=sr / 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Afficher et sauvegarder
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", fmax=sr / 2, cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Étape 4 : Processer les données du dataset
# Définir les paramètres de la fréquence d'échantillonnage
sampling_rate = 100  # Exemple : STEAD utilise 100 Hz
for idx, row in data.iterrows():
    # Charger le signal (exemple pour une seule composante)
    signal = row['trace_z']  # Vous pouvez choisir trace_z, trace_e ou trace_n selon vos besoins
    if len(signal) == 0:
        continue  # Sauter les signaux vides
    # Normaliser le signal
    signal = np.array(signal, dtype=np.float32)
    signal = signal / np.max(np.abs(signal))
    # Sauvegarder l'image du spectrogramme
    save_path = os.path.join(output_dir, f"spectrogram_{idx}.png")
    plot_spectrogram(signal, sr=sampling_rate, save_path=save_path)

    if idx >= 10:  # Limiter à 10 spectrogrammes pour l'exemple
        break

print(f"Spectrogrammes sauvegardés dans le dossier : {output_dir}")
