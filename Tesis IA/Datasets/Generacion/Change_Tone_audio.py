import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os


AUDIO_PATH = "C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/Datasets IA Audio/fold_gunshot/145206-6-3-0_processed.wav"  # Cambia esto por tu archivo
PITCH_STEPS = 2             # Número de semitonos 
TIME_SHIFT_SECONDS = 0.2    # Tiempo de desplazamiento 


audio, sr = librosa.load(AUDIO_PATH, sr=None)
duration = librosa.get_duration(y=audio, sr=sr)

# --- AUGMENTACIÓN ---

# 1. Pitch Shifting
audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=PITCH_STEPS)

# 2. Time Shifting
samples_shift = int(TIME_SHIFT_SECONDS * sr)
audio_time = np.roll(audio, samples_shift)


sf.write("disparo_pitch.wav", audio_pitch, sr)
sf.write("disparo_time.wav", audio_time, sr)

# --- FUNCIÓN PARA MOSTRAR AUDIO + ESPECTROGRAMA ---
def plot_audio_and_spectrogram(y, sr, title):
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    fig.suptitle(title)

    # Forma de onda
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title("Forma de onda")

    # Espectrograma
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axs[1])
    axs[1].set_title("Espectrograma (escala logarítmica)")
    fig.colorbar(img, ax=axs[1], format="%+2.f dB")
    plt.tight_layout()
    plt.show()

# --- MOSTRAR RESULTADOS ---
plot_audio_and_spectrogram(audio, sr, "Audio Original")
plot_audio_and_spectrogram(audio_pitch, sr, f"Pitch Shifted (+{PITCH_STEPS} semitonos)")
plot_audio_and_spectrogram(audio_time, sr, f"Time Shifted (+{TIME_SHIFT_SECONDS}s)")
