import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Paramètres du signal
fs = 44100  # Fréquence d'échantillonnage (Hz)
duration = 5  # Durée du signal (secondes)
f = 440.0  # Fréquence du signal (Hz)

# Génération du signal sinusoïdal
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * f * t)

# # Lecture du signal
# sd.play(signal, fs)
# sd.wait()  # Attendre que la lecture soit terminée

# Visualisation du signal complet
# plt.plot(t, signal)
# plt.title("Signal Sinusoïdal")
# plt.xlabel("Temps [s]")
# plt.ylabel("Amplitude")
# plt.show()

# # Visualisation d'une partie du signal
# tmin = 0.0
# tmax = 0.005
# plt.plot(t, signal)
# plt.title("Signal Sinusoïdal dans l'intervalle [0.0, 0.005]")
# plt.xlabel("Temps [s]")
# plt.ylabel("Amplitude")
# plt.xlim([tmin, tmax])
# plt.show()



# Générer du bruit
noise = 0.05 * np.random.randn(len(signal))

# Ajouter le bruit au signal
signal_noisy = signal + noise

# Lecture du signal
sd.play(signal_noisy, fs)
sd.wait()  # Attendre que la lecture soit terminée

# Visualisation du signal bruité
tmin = 0.0
tmax = 0.005
plt.plot(t, signal_noisy)
plt.title("Signal Sinusoïdal avec Bruit")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.xlim([tmin, tmax])
plt.show()

# Stocker le signal bruité dans un fichier WAV
from scipy.io.wavfile import write
write("signal_noisy.wav", fs, signal_noisy)
