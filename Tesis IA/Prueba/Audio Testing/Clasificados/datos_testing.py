import os
import shutil
import time
import csv

# CONFIGURACIÓN
source_folder = r"Tesis IA\Prueba\Audio Testing\Clasificados\no_gunshot"     
dest_folder = r"Tesis IA\Prueba\Audio Testing"    
csv_filename = r"Tesis IA\Prueba\Audio Testing\audios_etiquetadosNG.csv"
is_gunshot = False  

# Crea la carpeta de destino si no existe
os.makedirs(dest_folder, exist_ok=True)

# Lista de datos para el CSV
csv_data = [("filename", "is_gunshot")]
i = 0
# Procesa cada archivo .wav en la carpeta de origen
for filename in os.listdir(source_folder):
    if filename.lower().endswith(".wav"):
        
        if i == 80:
            break
        new_filename = f"audioPoligonoGuns2_{i}.wav"
        time.sleep(0.1)  
        i += 1
      
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(dest_folder, new_filename)

   
        shutil.copy(src_path, dst_path)

        
        csv_data.append((new_filename, int(is_gunshot)))

# Guarda el CSV
with open(csv_filename, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"Procesados {len(csv_data)-1} audios (copiados). CSV guardado como '{csv_filename}'.")
