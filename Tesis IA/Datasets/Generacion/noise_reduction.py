import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import random
import pywt
import pywt.data

AUDIO_PATH = "C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis"
audio_path = f"{AUDIO_PATH}/Audio_Datasets/PoligonoTiro"
carpeta_entrada = audio_path
carpeta_salida = f"{audio_path}/Noise reduction"
carpeta_graficas = carpeta_salida
informe_salida = f"{carpeta_salida}/inf.txt"
num_muestras_graficas = 20  

os.makedirs(carpeta_salida, exist_ok=True)
os.makedirs(carpeta_graficas, exist_ok=True)

def calcular_ruido(audio):
    return np.sqrt(np.mean(audio**2))

def procesar_audio(y, sr):
    longitud_ventana = int(5 * sr)
    solapamiento = int(longitud_ventana / 6)
    señal_procesada = np.zeros_like(y)
    paso = longitud_ventana - solapamiento

    for i in range(0, len(y) - longitud_ventana + 1, paso):
        segmento = y[i:i + longitud_ventana]
        segmento_limpio = signal.wiener(segmento, mysize=8000)
        ventana = np.hanning(longitud_ventana)
        segmento_limpio *= ventana
        señal_procesada[i:i + longitud_ventana] += segmento_limpio

    y_wiener = señal_procesada / np.max(np.abs(señal_procesada))
    sos = signal.butter(10, [1, 7500], btype='bandpass', fs=sr, output='sos')
    y_filtrado = signal.sosfilt(sos, y_wiener)

    def eliminar_ruido_wavelet(señal, wavelet='db8', nivel=10, umbral_factor=2.5):
        """
        Elimina ruido usando descomposición wavelet con umbralización
        """
        # Realizar la descomposición wavelet multi-nivel
        coeficientes = pywt.wavedec(señal, wavelet, level=nivel)
        
        # Calcular umbral adaptativo basado en la estimación del ruido
        sigma = (np.median(np.abs(coeficientes[-1])) / 0.6745)
        umbral = umbral_factor * sigma * np.sqrt(2 * np.log(len(señal)))
        
        # Aplicar umbralización suave (soft thresholding) a los coeficientes de detalle
        coefs_nuevos = [coeficientes[0]]  # Mantener coeficientes de aproximación
        
        for i in range(1, len(coeficientes)):
            # Umbralización suave
            coefs_nuevos.append(pywt.threshold(coeficientes[i], umbral, mode='soft'))
        
        # Reconstruir la señal con los coeficientes modificados
        señal_limpia = pywt.waverec(coefs_nuevos, wavelet)
        
        # Asegurar que la longitud de la señal reconstruida sea igual a la original
        señal_limpia = señal_limpia[:len(señal)]
        
        return señal_limpia
    # 2. Descomposición Wavelet
    señal_wavelet = eliminar_ruido_wavelet(y, wavelet='db8', nivel=10, umbral_factor=6)

    return señal_wavelet

def graficar_amplitud(y1, y2, sr, nombre_archivo, carpeta):
    tiempo = np.linspace(0, len(y1)/sr, len(y1))
    plt.figure(figsize=(12, 5))
    plt.subplot(2,1,1)
    plt.plot(tiempo, y1)
    plt.title('Amplitud - Original')
    plt.xlabel('Tiempo (s)')
    plt.subplot(2,1,2)
    plt.plot(tiempo, y2)
    plt.title('Amplitud - Filtrado')
    plt.xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"{nombre_archivo}_amplitud.png"))
    plt.close()

def graficar_espectrograma(y1, y2, sr, nombre_archivo, carpeta):
    plt.figure(figsize=(12, 5))
    plt.subplot(2,1,1)
    plt.specgram(y1, Fs=sr, NFFT=1024, noverlap=512, cmap='magma')
    plt.title('Espectrograma - Original')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(label='dB')
    
    plt.subplot(2,1,2)
    plt.specgram(y2, Fs=sr, NFFT=1024, noverlap=512, cmap='magma')
    plt.title('Espectrograma - Filtrado')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(label='dB')

    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"{nombre_archivo}_espectrograma.png"))
    plt.close()

# ========== PROCESAMIENTO GENERAL ==========
archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith('.wav')]
muestras_graficas = random.sample(archivos, min(num_muestras_graficas, len(archivos)))

with open(informe_salida, "w") as f:
    f.write(f"Informe de procesamiento de audio\nFecha: {datetime.now()}\n\n")
    for archivo in archivos:
        path_audio = os.path.join(carpeta_entrada, archivo)
        print(f"Procesando: {archivo}")
        y_original, sr = sf.read(path_audio)
        y_filtrado = procesar_audio(y_original, sr)

        # Guardar audio procesado
        path_salida = os.path.join(carpeta_salida, archivo)
        sf.write(path_salida, y_filtrado, sr)

        # Calcular métricas
        rms_original = calcular_ruido(y_original)
        rms_filtrado = calcular_ruido(y_filtrado)
        f.write(f"Archivo: {archivo}\n")
        f.write(f"  Ruido original (RMS): {rms_original:.6f}\n")
        f.write(f"  Ruido filtrado (RMS): {rms_filtrado:.6f}\n")
        f.write(f"  Reducción: {rms_original - rms_filtrado:.6f}\n\n")

        # Gráficas si el archivo fue elegido aleatoriamente
        if archivo in muestras_graficas:
            nombre_base = os.path.splitext(archivo)[0]
            graficar_amplitud(y_original, y_filtrado, sr, nombre_base, carpeta_graficas)
            graficar_espectrograma(y_original, y_filtrado, sr, nombre_base, carpeta_graficas)

print("✅ Procesamiento completo. Informe y gráficas generadas.")
