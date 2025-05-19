import os
import csv
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import shutil
from pathlib import Path
from PIL import Image, ImageTk
import threading

class AudioProcessorApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Procesador de Audios para Red Neuronal")
        self.root.geometry("900x700")  # Tamaño inicial
        self.root.minsize(600, 500)  # Tamaño mínimo para evitar colapsos
        
        # Configurar tema Forest
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")  # Usar tema verde que se parece a "Forest"
        
        self.mode_var = ctk.StringVar(value="folder")
        self.target_duration = 4  # segundos
        self.target_sr = 22050    # frecuencia de muestreo
        
        self.input_path = ctk.StringVar()
        self.csv_path = ctk.StringVar()
        self.root_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.audio_type = ctk.StringVar()
        self.audio_id = ctk.StringVar()
        
        # Contadores para el procesamiento
        self.total_files = 0
        self.processed_count = 0
        self.current_progress = 0
        
        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        # Frame principal con scroll
        main_frame = ctk.CTkScrollableFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Título
        title_label = ctk.CTkLabel(main_frame, text="Procesador de Audios para Red Neuronal", 
                                   font=ctk.CTkFont(size=22, weight="bold"))
        title_label.pack(pady=15)

        # Panel izquierdo (similar al panel de checkbuttons en la imagen)
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side="left", fill="y", padx=(0, 10), pady=10)
        
        # Modo de procesamiento - Estilo similar a "Checkbuttons" en la imagen
        mode_frame = ctk.CTkFrame(left_panel)
        mode_frame.pack(fill="x", pady=10)

        mode_label = ctk.CTkLabel(mode_frame, text="Modo de procesamiento:", anchor="w", 
                                  font=ctk.CTkFont(weight="bold"))
        mode_label.pack(padx=10, pady=5, anchor="w")

        folder_radio = ctk.CTkRadioButton(mode_frame, text="Carpeta única", 
                                         variable=self.mode_var, value="folder", 
                                         command=self.update_ui,
                                         fg_color="#2D8C4E",  # Color verde Forest
                                         hover_color="#1E5631")  # Verde más oscuro para hover
        folder_radio.pack(anchor="w", padx=10, pady=5)

        csv_radio = ctk.CTkRadioButton(mode_frame, text="CSV con múltiples tipos", 
                                      variable=self.mode_var, value="csv", 
                                      command=self.update_ui,
                                      fg_color="#2D8C4E",  # Color verde Forest
                                      hover_color="#1E5631")  # Verde más oscuro para hover
        csv_radio.pack(anchor="w", padx=10, pady=5)
        
        # Panel derecho (principal)
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, pady=10)

        # Frame contenedor de inputs
        self.input_frame = ctk.CTkFrame(right_panel)
        self.input_frame.pack(fill="x", pady=10)

        # Frame modo carpeta
        self.folder_frame = ctk.CTkFrame(self.input_frame)
        
        folder_label = ctk.CTkLabel(self.folder_frame, text="Carpeta de audios:", anchor="w")
        folder_label.pack(anchor="w", padx=10, pady=5)

        folder_entry_frame = ctk.CTkFrame(self.folder_frame)
        folder_entry_frame.pack(fill="x", padx=10)

        folder_entry = ctk.CTkEntry(folder_entry_frame, textvariable=self.input_path,
                                   border_color="#2D8C4E")  # Borde verde Forest
        folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        folder_button = ctk.CTkButton(folder_entry_frame, text="Explorar", 
                                     command=self.browse_folder,
                                     fg_color="#2D8C4E",  # Color verde Forest
                                     hover_color="#1E5631")  # Verde más oscuro para hover
        folder_button.pack(side="right")

        # Tipo de audio e ID
        self.type_frame = ctk.CTkFrame(self.folder_frame)
        self.type_frame.pack(fill="x", pady=10)

        type_label = ctk.CTkLabel(self.type_frame, text="Tipo de audio:", anchor="w")
        type_label.pack(anchor="w", padx=10, pady=5)

        type_entry = ctk.CTkEntry(self.type_frame, textvariable=self.audio_type,
                                 border_color="#2D8C4E")  # Borde verde Forest
        type_entry.pack(fill="x", padx=10)

        id_label = ctk.CTkLabel(self.type_frame, text="ID del tipo:", anchor="w")
        id_label.pack(anchor="w", padx=10, pady=5)

        id_entry = ctk.CTkEntry(self.type_frame, textvariable=self.audio_id,
                               border_color="#2D8C4E")  # Borde verde Forest
        id_entry.pack(fill="x", padx=10)

        # CSV Frame (oculto inicialmente)
        self.csv_frame = ctk.CTkFrame(self.input_frame)

        csv_label = ctk.CTkLabel(self.csv_frame, text="Archivo CSV:", anchor="w")
        csv_label.pack(anchor="w", padx=10, pady=5)

        csv_entry_frame = ctk.CTkFrame(self.csv_frame)
        csv_entry_frame.pack(fill="x", padx=10)

        csv_entry = ctk.CTkEntry(csv_entry_frame, textvariable=self.csv_path,
                                border_color="#2D8C4E")  # Borde verde Forest
        csv_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        csv_button = ctk.CTkButton(csv_entry_frame, text="Explorar", 
                                  command=self.browse_csv,
                                  fg_color="#2D8C4E",  # Color verde Forest
                                  hover_color="#1E5631")  # Verde más oscuro para hover
        csv_button.pack(side="right")

        root_label = ctk.CTkLabel(self.csv_frame, text="Carpeta raíz (donde están los audios):", anchor="w")
        root_label.pack(anchor="w", padx=10, pady=5)

        root_entry_frame = ctk.CTkFrame(self.csv_frame)
        root_entry_frame.pack(fill="x", padx=10)

        root_entry = ctk.CTkEntry(root_entry_frame, textvariable=self.root_path,
                                 border_color="#2D8C4E")  # Borde verde Forest
        root_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        root_button = ctk.CTkButton(root_entry_frame, text="Explorar", 
                                   command=self.browse_root,
                                   fg_color="#2D8C4E",  # Color verde Forest
                                   hover_color="#1E5631")  # Verde más oscuro para hover
        root_button.pack(side="right")

        # Salida
        output_frame = ctk.CTkFrame(right_panel)
        output_frame.pack(fill="x", pady=10)

        output_label = ctk.CTkLabel(output_frame, text="Carpeta de salida:", anchor="w")
        output_label.pack(anchor="w", padx=10, pady=5)

        output_entry_frame = ctk.CTkFrame(output_frame)
        output_entry_frame.pack(fill="x", padx=10)

        output_entry = ctk.CTkEntry(output_entry_frame, textvariable=self.output_path,
                                   border_color="#2D8C4E")  # Borde verde Forest
        output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        output_button = ctk.CTkButton(output_entry_frame, text="Explorar", 
                                     command=self.browse_output,
                                     fg_color="#2D8C4E",  # Color verde Forest
                                     hover_color="#1E5631")  # Verde más oscuro para hover
        output_button.pack(side="right")

        # Barra de progreso y contador
        progress_frame = ctk.CTkFrame(right_panel)
        progress_frame.pack(fill="x", pady=10, padx=10)
        
        self.files_label = ctk.CTkLabel(progress_frame, text="Files: 0/0", anchor="w")
        self.files_label.pack(side="left", padx=(5, 10))
        
        self.global_progress_bar = ctk.CTkProgressBar(progress_frame)
        self.global_progress_bar.pack(side="right", fill="x", expand=True, padx=5)
        self.global_progress_bar.set(0)
        
        # Botón de procesar (estilo AccentButton verde de la imagen)
        process_button = ctk.CTkButton(right_panel, text="Procesar Audios", 
                                      font=ctk.CTkFont(size=16),
                                      command=self.start_processing,  # Usar el método que inicia el hilo
                                      fg_color="#2D8C4E",  # Color verde Forest
                                      hover_color="#1E5631",  # Verde más oscuro para hover
                                      height=40)  # Altura similar al AccentButton de la imagen
        process_button.pack(pady=20, padx=20, fill="x")

        # Log - Estilo similar a la tabla en la imagen con borde
        log_frame = ctk.CTkFrame(right_panel, border_width=1, border_color="#2D8C4E")
        log_frame.pack(fill="both", expand=True, pady=10, padx=10)

        log_label = ctk.CTkLabel(log_frame, text="Registro:", font=ctk.CTkFont(weight="bold"))
        log_label.pack(anchor="w", pady=5, padx=5)

        self.log_text = ctk.CTkTextbox(log_frame, height=150, border_color="#2D8C4E")
        self.log_text.configure(bg_color="#1E1E1E", fg_color="#252525")  # Colores oscuros similares a la tabla
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Pestaña similar a la imagen
        tab_frame = ctk.CTkFrame(right_panel)
        tab_frame.pack(fill="x", pady=(10, 0))
        
        tab_view = ctk.CTkTabview(tab_frame, fg_color="#2D8C4E", segmented_button_fg_color="#1E1E1E",
                                  segmented_button_selected_color="#2D8C4E",
                                  segmented_button_unselected_color="#1E1E1E")
        tab_view.pack(fill="x")
        
        tab_1 = tab_view.add("Configuración")
        tab_2 = tab_view.add("Resultado")
        
        # Inicializa el modo visible
        self.update_ui()

    def update_ui(self):
        # Limpiar los frames existentes
        self.folder_frame.pack_forget()
        self.csv_frame.pack_forget()
        
        # Mostrar el frame correspondiente según el modo seleccionado
        if self.mode_var.get() == "folder":
            self.folder_frame.pack(fill="x")
        else:
            self.csv_frame.pack(fill="x")
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Audio Folder")
        if folder:
            self.input_path.set(folder)
    
    def browse_csv(self):
        file = filedialog.askopenfilename(title="Select CSV File", 
                                         filetypes=[("CSV files", "*.csv")])
        if file:
            self.csv_path.set(file)
    
    def browse_root(self):
        folder = filedialog.askdirectory(title="Select Root Folder")
        if folder:
            self.root_path.set(folder)
    
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)
    
    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()
    
    def update_progress(self, current, total):
        self.current_progress = current / total if total > 0 else 0
        self.global_progress_bar.set(self.current_progress)
        self.files_label.configure(text=f"Files: {current}/{total}")
        self.root.update_idletasks()
    
    def process_audio_file(self, file_path, output_folder, audio_type, audio_id, csv_data):
        try:
            # Cargar el audio
            y, sr = librosa.load(file_path, sr=None)
            
            # Extraer características básicas antes de procesar
            filename = os.path.basename(file_path)
            original_duration = librosa.get_duration(y=y, sr=sr)
            rms = np.sqrt(np.mean(y**2))
            
            # Remuestrear si es necesario
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Calcular la duración actual en segundos
            current_duration = len(y) / sr
            
            processed_segments = []
            
            if current_duration < self.target_duration:
                # Si es más corto, extender repitiendo el audio
                repetitions = int(np.ceil(self.target_duration / current_duration))
                y_extended = np.tile(y, repetitions)
                y_processed = y_extended[:int(self.target_duration * sr)]
                processed_segments.append(y_processed)
                
            elif current_duration > self.target_duration:
                # Si es más largo, dividir en segmentos de 4 segundos
                num_segments = int(current_duration // self.target_duration)
                
                for seg_idx in range(num_segments):
                    start_sample = int(seg_idx * self.target_duration * sr)
                    end_sample = int((seg_idx + 1) * self.target_duration * sr)
                    segment = y[start_sample:end_sample]
                    processed_segments.append(segment)
                
                # Si queda un fragmento que no completa los 4 segundos, extenderlo
                remaining_duration = current_duration % self.target_duration
                if remaining_duration > 0.5:  # Solo procesar si dura más de 0.5 segundos
                    start_sample = int(num_segments * self.target_duration * sr)
                    remaining_segment = y[start_sample:]
                    
                    # Extender el segmento restante a 4 segundos
                    repetitions = int(np.ceil(self.target_duration / remaining_duration))
                    extended_segment = np.tile(remaining_segment, repetitions)
                    extended_segment = extended_segment[:int(self.target_duration * sr)]
                    processed_segments.append(extended_segment)
            else:
                # Si ya tiene exactamente 4 segundos, usarlo tal cual
                processed_segments.append(y)
            
            # Guardar cada segmento procesado
            for seg_idx, segment in enumerate(processed_segments):
                # Crear nuevo nombre de archivo
                base_name = os.path.splitext(filename)[0]
                if len(processed_segments) > 1:
                    output_filename = f"{audio_type}_{base_name[2:8]}_A{seg_idx+1}.wav"
                else:
                    output_filename = f"{audio_type}_{base_name[2:8]}_A.wav"
                
                # Asegurar que la carpeta de salida existe
                os.makedirs(output_folder, exist_ok=True)

                fold_out = os.path.join(output_folder, f'Fold {audio_type}')
                if not os.path.exists(fold_out):
                    os.makedirs(fold_out)

                output_path = os.path.join(fold_out, output_filename)
                
                # Calcular características adicionales del segmento procesado
                segment_rms = np.sqrt(np.mean(segment**2))
                segment_zero_crossings = librosa.feature.zero_crossing_rate(segment)[0].mean()
                segment_spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.target_sr)[0].mean()

                if segment_rms < 0.001 and segment_zero_crossings < 0.001 and segment_spectral_centroid < 0.001:
                    self.log(f"Not Processed (Silence): {output_filename}")
                    return True

                # Calcular SNR (Signal-to-Noise Ratio) aproximado
                signal_power = np.mean(segment**2)
                noise_estimate = np.var(segment - np.mean(segment))
                snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 100
                
                # Intensidad (usamos RMS como aproximación)
                intensidad = segment_rms * 100  # Escalamos para tener un valor más legible
                
                # Bandwidth (aproximado usando la desviación estándar del centroide espectral)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.target_sr)[0].mean()

                # Añadir información al CSV con los campos requeridos
                csv_data.append({
                    'filename': output_filename,
                    #'fsID': os.path.splitext(base_name)[0],
                    'start': 0.0,
                    'end': self.target_duration,
                    #'salience': 1,
                    'fold': f'Fold {audio_type}',
                    'classID': audio_id,
                    'class': audio_type,
                    'SNR': round(snr, 2),
                    'Intensidad': round(intensidad, 2),
                    'bandwiht': round(spectral_bandwidth, 2)
                })
                
                # Guardar el audio procesado
                
                sf.write(output_path, segment, self.target_sr)
                
                self.log(f"Processed: {output_filename}")
            
            return True
        except Exception as e:
            self.log(f"Error processing {file_path}: {str(e)}")
            return False
    
    def start_processing(self):
        # Reiniciar contadores
        self.total_files = 0
        self.processed_count = 0
        
        # Inicia el procesamiento en un hilo separado para no bloquear la UI
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.daemon = True
        processing_thread.start()
    
    def process_audio(self):
        # Verificar que se ha seleccionado la carpeta de salida
        if not self.output_path.get():
            messagebox.showerror("Error", "You must select an output folder")
            return
        
        output_folder = self.output_path.get()
        
        # Lista para almacenar datos del CSV
        csv_data = []
        
        # Procesar según el modo seleccionado
        mode = self.mode_var.get()
        
        if mode == "folder":
            # Verificar que se ha seleccionado la carpeta de entrada
            if not self.input_path.get():
                messagebox.showerror("Error", "You must select an input folder")
                return
                
            # Verificar que se ha seleccionado el tipo de audio
            if not self.audio_type.get() or not self.audio_id.get():
                messagebox.showerror("Error", "You must specify audio type and ID")
                return
            
            input_folder = self.input_path.get()
            audio_type = self.audio_type.get()
            audio_id = self.audio_id.get()
            
            # Crear carpeta de salida si no existe
            os.makedirs(output_folder, exist_ok=True)
            
            # Obtener lista de archivos de audio
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                audio_files.extend(list(Path(input_folder).glob(f'**/*{ext}')))
                
            self.total_files = len(audio_files)
            self.log(f"Found {self.total_files} audio files to process")
            
            # Procesar cada archivo
            for i, file_path in enumerate(audio_files):
                self.update_progress(i + 1, self.total_files)
                self.log(f"Processing {file_path.name}...")
                success = self.process_audio_file(
                    str(file_path), 
                    output_folder, 
                    audio_type, 
                    audio_id, 
                    csv_data
                )
                if success:
                    self.processed_count += 1
            
        else:  # Modo CSV
            # Verificar que se ha seleccionado el archivo CSV y la carpeta raíz
            if not self.csv_path.get() or not self.root_path.get():
                messagebox.showerror("Error", "You must select both a CSV file and a root folder")
                return
            
            csv_file = self.csv_path.get()
            root_folder = self.root_path.get()
            
            try:
                # Leer el CSV
                df = pd.read_csv(csv_file)
                self.total_files = len(df)
                self.log(f"Found {self.total_files} entries in CSV file")
                
                # Procesar cada entrada del CSV
                for i, row in df.iterrows():
                    self.update_progress(i + 1, self.total_files)
                    
                    # Obtener información de la fila
                    file_path = os.path.join(root_folder, row['filename'])
                    audio_type = row['class']
                    audio_id = str(row['classID'])
                    
                    if not os.path.exists(file_path):
                        self.log(f"File not found: {file_path}")
                        continue
                    
                    # Procesar el archivo
                    self.log(f"Processing {os.path.basename(file_path)}...")
                    success = self.process_audio_file(
                        file_path, 
                        output_folder, 
                        audio_type, 
                        audio_id, 
                        csv_data
                    )
                    if success:
                        self.processed_count += 1
                    
            except Exception as e:
                self.log(f"Error processing CSV file: {str(e)}")
                messagebox.showerror("Error", f"Error processing CSV file: {str(e)}")
                return
        
        # Guardar los datos de procesamiento en un CSV
        if csv_data:
            output_csv_path = os.path.join(output_folder, "audio_metadata.csv")
            try:
                if os.path.isfile(output_csv_path):
                    pd.DataFrame(csv_data).to_csv(output_csv_path, mode='a', index=False, header=False)
                else:
                    pd.DataFrame(csv_data).to_csv(output_csv_path, index=False)
                self.log(f"Metadata saved to {output_csv_path}")
            except Exception as e:
                self.log(f"Error saving metadata CSV: {str(e)}")
        
        # Actualizar la barra de progreso al 100%
        self.update_progress(self.total_files, self.total_files)
        
        # Mostrar mensaje de finalización
        self.log(f"Processing complete! {self.processed_count} files processed successfully.")
        messagebox.showinfo("Success", f"Processing complete!\n{self.processed_count} files processed successfully.")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AudioProcessorApp()
    app.run()