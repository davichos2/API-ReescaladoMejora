import os
import subprocess
import tkinter as tk
from tkinter import filedialog
import time

# --- ¡NUEVO! 10 CONJUNTOS DE PERFILES (CPU + GPU) ---
# Cada conjunto tiene un objetivo (ej. "Calidad Alta") y 
# su implementación para CPU (libx264) y GPU (h264_nvenc).

PROFILE_SETS = [
    {
        "name": "Set-01_MaxCalidad",
        "cpu": {"codec": "libx264", "params": ["-preset", "slow", "-crf", "18"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p7", "-cq", "18"]} # p7 = más lento
    },
    {
        "name": "Set-02_AltaCalidad",
        "cpu": {"codec": "libx264", "params": ["-preset", "medium", "-crf", "20"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p6", "-cq", "20"]} # p6 = lento
    },
    {
        "name": "Set-03_Balanceado-Calidad",
        "cpu": {"codec": "libx264", "params": ["-preset", "medium", "-crf", "23"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p5", "-cq", "23"]} # p5 = medio
    },
    {
        "name": "Set-04_Balanceado-Rapido",
        "cpu": {"codec": "libx264", "params": ["-preset", "fast", "-crf", "23"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p4", "-cq", "23"]} # p4 = rápido
    },
    {
        "name": "Set-05_Rapido-MediaCalidad",
        "cpu": {"codec": "libx264", "params": ["-preset", "fast", "-crf", "25"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p3", "-cq", "25"]} # p3 = más rápido
    },
    {
        "name": "Set-06_Rapido-BajaCalidad",
        "cpu": {"codec": "libx264", "params": ["-preset", "faster", "-crf", "28"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p2", "-cq", "28"]} # p2 = muy rápido
    },
    {
        "name": "Set-07_MuyRapido",
        "cpu": {"codec": "libx264", "params": ["-preset", "veryfast", "-crf", "28"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p1", "-cq", "28"]} # p1 = el más rápido
    },
    {
        "name": "Set-08_UltraRapido-CPU", # Enfocado en CPU ultrarrápido
        "cpu": {"codec": "libx264", "params": ["-preset", "ultrafast", "-crf", "26"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p1", "-cq", "26"]}
    },
    {
        "name": "Set-09_MaxCompresion-CPU", # Enfocado en CPU muy lento
        "cpu": {"codec": "libx264", "params": ["-preset", "veryslow", "-crf", "24"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p7", "-cq", "24"]}
    },
    {
        "name": "Set-10_BajaLatencia-GPU", # Perfil especial de GPU
        "cpu": {"codec": "libx264", "params": ["-preset", "medium", "-crf", "22"]},
        "gpu": {"codec": "h264_nvenc", "params": ["-preset", "p5", "-cq", "22", "-tune", "ll"]} # ll = low latency
    }
]


# --- RUTA DE SALIDA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "videos_comprimidos_10x2_perfiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def execute_ffmpeg_command(input_path, output_path, profile, profile_type):
    """
    Ejecuta un comando de FFmpeg usando un perfil específico (CPU o GPU).
    """
    print(f"  Iniciando Tarea [{profile_type}] -> {os.path.basename(output_path)}")
    start_time = time.time()
    
    # Construir el comando de FFmpeg
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', profile['codec'], # 'libx264' o 'h264_nvenc'
    ]
    
    # Añadir los parámetros específicos del perfil (ej: -crf 23)
    command.extend(profile['params'])
    
    # Añadir parámetros finales
    command.extend([
        '-c:a', 'copy', # Copiar audio sin recomprimir
        '-y',           # Sobrescribir sin preguntar
        output_path
    ])
    
    try:
        # Ejecutar el comando
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            encoding='utf-8'
        )
        
        end_time = time.time()
        print(f"  ✅ Éxito [{profile_type}] en {end_time - start_time:.2f}s.")
        
    except subprocess.CalledProcessError as e:
        # ¡Error! El comando de FFmpeg falló
        print(f"  ❌ ERROR [{profile_type}] al comprimir {os.path.basename(output_path)}")
        print(f"  Razón: El códec '{profile['codec']}' falló.")
        print("  (¿Estás seguro de que 'h264_nvenc' está instalado y disponible?)")
        print("  --- Salida de FFmpeg ---")
        print(e.stderr)
        print("  -------------------------")
    except FileNotFoundError:
        print("  ❌ ERROR FATAL: No se encontró 'ffmpeg'.")
        print("  Asegúrate de que FFmpeg esté instalado y añadido al PATH del sistema.")
        return False # Detener el script si FFmpeg no se encuentra
        
    return True


def main():
    # 1. Abrir el explorador de archivos para seleccionar videos
    root = tk.Tk()
    root.withdraw()
    print("Abriendo explorador para seleccionar los 10 videos (o más)...")
    
    video_paths = filedialog.askopenfilenames(
        title="Selecciona tus videos (idealmente 10)",
        filetypes=[
            ("Archivos de video", "*.mp4 *.avi *.mov *.mkv"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    if not video_paths:
        print("No se seleccionó ningún video. Saliendo.")
        return

    print(f"Se seleccionaron {len(video_paths)} videos. Empezando...")
    print(f"Los archivos de salida se guardarán en: {OUTPUT_DIR}\n")
    
    num_sets = len(PROFILE_SETS) # Siempre será 10

    # 2. Iterar sobre cada video seleccionado
    for i, input_path in enumerate(video_paths):
        print("="*80)
        print(f"Procesando Video {i+1} de {len(video_paths)}: {os.path.basename(input_path)}")
        
        # --- LÓGICA PRINCIPAL ---
        # Usamos el operador módulo (%) para ciclar a través de los 10 conjuntos.
        current_set = PROFILE_SETS[i % num_sets]
        print(f"Aplicando Conjunto de Perfil: {current_set['name']}")

        # Obtener nombre base y extensión
        base_name, extension = os.path.splitext(os.path.basename(input_path))
        
        # --- TAREA 1: COMPRESIÓN CPU ---
        profile_cpu = current_set['cpu']
        output_name_cpu = f"{base_name}_SET-{i+1:02d}_{current_set['name']}_CPU{extension}"
        output_path_cpu = os.path.join(OUTPUT_DIR, output_name_cpu)
        
        if not execute_ffmpeg_command(input_path, output_path_cpu, profile_cpu, "CPU"):
            break # Abortar si ffmpeg no existe

        # --- TAREA 2: COMPRESIÓN GPU ---
        profile_gpu = current_set['gpu']
        output_name_gpu = f"{base_name}_SET-{i+1:02d}_{current_set['name']}_GPU{extension}"
        output_path_gpu = os.path.join(OUTPUT_DIR, output_name_gpu)

        if not execute_ffmpeg_command(input_path, output_path_gpu, profile_gpu, "GPU"):
            break # Abortar si ffmpeg no existe
        
        print(f"Video {i+1} completado (CPU y GPU).")

    print("\n" + "="*80)
    print("¡Proceso de compresión completado para todos los videos!")
    print("="*80)

if __name__ == "__main__":
    main()