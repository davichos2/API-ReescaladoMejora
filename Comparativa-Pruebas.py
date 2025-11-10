import cv2
import os
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURACIÓN ---
FRAME_A_EXTRAER = 10  # El fotograma exacto que quieres guardar (0-indexado)
# --- /CONFIGURACIÓN ---

# 1. Definir dónde guardar las imágenes
# Se crea una carpeta "frames_extraidos" en el mismo lugar que el script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, f"frames_extraidos_frame_{FRAME_A_EXTRAER}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extraer_frame_especifico():
    # 2. Abrir el explorador de archivos para seleccionar videos
    root = tk.Tk()
    root.withdraw()
    print(f"Abriendo explorador para seleccionar los videos (18-20)...")
    
    video_paths = filedialog.askopenfilenames(
        title=f"Selecciona todos los videos a los que quieras extraer el frame {FRAME_A_EXTRAER}",
        filetypes=[
            ("Archivos de video", "*.mp4 *.avi *.mov *.mkv"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    if not video_paths:
        print("No se seleccionó ningún video. Saliendo.")
        return

    print(f"Se seleccionaron {len(video_paths)} videos. Empezando...")
    print(f"Los frames se guardarán en: {OUTPUT_DIR}\n")

    # 3. Iterar sobre cada video seleccionado
    videos_procesados = 0
    errores = 0
    
    for i, video_path in enumerate(video_paths):
        print(f"Procesando video {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        
        try:
            # Abrir el video
            cap = cv2.VideoCapture(video_path)
            
            # Mover el "cabezal" del video al fotograma exacto
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_A_EXTRAER)
            
            # Leer ese fotograma
            ret, frame = cap.read()
            
            if ret:
                # 4. Si se leyó bien, guardarlo como PNG
                
                # Crear un nombre de archivo único
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_name = f"{base_name}_frame_{FRAME_A_EXTRAER}.png"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                
                # Guardar la imagen (PNG es sin pérdida = máxima calidad)
                cv2.imwrite(output_path, frame)
                print(f"  ✅ ¡Frame {FRAME_A_EXTRAER} guardado como: {output_name}!")
                videos_procesados += 1
            else:
                # 5. Si 'ret' es False, es que el video no tiene un frame 10
                print(f"  ❌ Error: No se pudo leer el frame {FRAME_A_EXTRAER}. ¿El video es muy corto?")
                errores += 1
                
            # Liberar el archivo de video
            cap.release()
        
        except Exception as e:
            print(f"  ❌ Error fatal al procesar el video: {e}")
            errores += 1

    print("\n" + "="*80)
    print("¡Proceso de extracción completado!")
    print(f"Videos procesados exitosamente: {videos_procesados}")
    print(f"Videos con errores: {errores}")
    print("="*80)

if __name__ == "__main__":
    extraer_frame_especifico()