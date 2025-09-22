# test_compression.py

import os
import subprocess

def comprimir_gpu(input_path: str, output_path: str):
    """Llama a FFmpeg para reescalar a 4K y comprimir usando el codificador de la GPU (NVENC)."""
    # Usamos los mismos parámetros que en tu app principal
    PRESET_NVENC = 'p5'
    CQ_VALUE = '31'
    
    # Este es el mismo comando que probamos antes
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', 'scale=1920:-1', # Reescalamos a 1080p para esta prueba simple
        '-c:v', 'h264_nvenc',
        '-preset', PRESET_NVENC,
        '-cq', CQ_VALUE,
        '-c:a', 'copy',
        output_path
    ]
    
    print(f"Ejecutando comando:\n{' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=600)
        return result
    except FileNotFoundError:
        print("--- ERROR ---")
        print("Comando 'ffmpeg' no encontrado. Asegúrate de que FFmpeg esté instalado y en el PATH del sistema.")
        return None
    except subprocess.CalledProcessError as e:
        print("--- Error FFMPEG ---")
        print(f"El comando de FFmpeg falló. Esta fue la salida de error:")
        print(e.stderr)
        print("--------------------")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None

# --- Bloque principal para ejecutar la prueba ---
if __name__ == "__main__":
    input_file = "video_de_prueba.mp4"
    output_file = "video_comprimido_prueba.mp4"
    
    print(f"--- Iniciando prueba de compresión ---")
    print(f"Video de entrada: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"\n[ERROR] No se encontró el archivo '{input_file}'.")
        print("Por favor, asegúrate de tener un video con ese nombre en la misma carpeta.")
    else:
        resultado = comprimir_gpu(input_path=input_file, output_path=output_file)
        
        if resultado:
            ruta_absoluta = os.path.abspath(output_file)
            print(f"\n[ÉXITO] ¡Compresión finalizada correctamente!")
            print(f"Puedes encontrar el video de prueba en: {ruta_absoluta}")