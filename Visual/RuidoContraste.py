import cv2
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from scipy import stats as sp_stats
from scipy.stats import entropy as scipy_entropy
import mahotas
# --- ¡NUEVA IMPORTACIÓN! ---
import matplotlib.pyplot as plt

# --- ¡NUEVO! OBTENER RUTA DEL SCRIPT ---
# Esto nos da la ruta absoluta de la carpeta donde se está ejecutando este script.
# Todos los archivos (Excel, PNGs) se guardarán aquí.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- ¡NUEVA FUNCIÓN PARA GUARDAR HISTOGRAMAS! ---
def guardar_histograma_comparativo(gris_o, gris_m, frame_num, ruta_salida_plot):
    """
    Calcula, dibuja y guarda un gráfico comparativo de los histogramas
    de las dos imágenes en escala de grises.
    """
    try:
        # 1. Calcular histogramas para cada imagen
        # Parámetros: [imagen], [canal], [máscara], [bins], [rango]
        hist_o = cv2.calcHist([gris_o], [0], None, [256], [0, 256])
        hist_m = cv2.calcHist([gris_m], [0], None, [256], [0, 256])
        
        # 2. Crear la figura del gráfico
        plt.figure(figsize=(10, 6))
        
        # 3. Dibujar ambos histogramas
        plt.plot(hist_o, color='blue', label=f'Original (Frame {frame_num})', alpha=0.7)
        plt.plot(hist_m, color='red', label=f'Mejorado (Frame {frame_num})', alpha=0.7, linestyle='--')
        
        # 4. Añadir títulos y etiquetas
        plt.title(f'Histograma Comparativo - Frame {frame_num}')
        plt.xlabel('Nivel de Gris (0 = Negro, 255 = Blanco)')
        plt.ylabel('Cantidad de Píxeles')
        plt.legend()  # Muestra las etiquetas 'Original' y 'Mejorado'
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim([0, 256]) # Asegura que el eje X vaya de 0 a 256
        plt.gca().get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ','))
        ) # Formatea el eje Y con comas (ej: 1,000)

        # 5. Guardar la figura en un archivo PNG
        plt.savefig(ruta_salida_plot)
        
        # 6. Cerrar la figura para liberar memoria (importante en bucles)
        plt.close()
        
    except Exception as e:
        print(f"  ❌ Error al generar histograma para frame {frame_num}: {e}")

# --- Funciones de Métricas (Sin cambios) ---

def calcular_media(imagen_gris):
    return np.mean(imagen_gris)

def calcular_varianza(imagen_gris):
    return np.var(imagen_gris)

def calcular_curtosis(imagen_gris):
    return sp_stats.kurtosis(imagen_gris, axis=None)

def calcular_entropia(imagen_gris):
    _, counts = np.unique(imagen_gris.ravel(), return_counts=True)
    return scipy_entropy(counts, base=2)

def calcular_energia_pixeles(imagen_gris):
    return np.sum(imagen_gris.astype('float64')**2)

def calcular_densidad_bordes(imagen_gris):
    bordes = cv2.Canny(imagen_gris, 100, 200)
    return (np.sum(bordes > 0) / imagen_gris.size) * 100

def calcular_contraste_textura(imagen_gris):
    try:
        features = mahotas.features.haralick(imagen_gris).mean(axis=0)        
        return features[1] # Contraste
    except ValueError as e:
        print(f"  Advertencia en Mahotas (textura): {e}. Devolviendo 0.")
        return 0

# --- FUNCIÓN PRINCIPAL (MODIFICADA) ---
def analizar_videos(path_original, path_mejorado, frame_numbers, output_dir):
    """
    Analiza y compara fotogramas, guardando métricas y plots de histogramas
    en la carpeta 'output_dir'.
    """
    
    if not os.path.exists(path_original):
        print(f"Error: No se encontró el video original en {path_original}")
        return None
    if not os.path.exists(path_mejorado):
        print(f"Error: No se encontró el video mejorado en {path_mejorado}")
        return None

    cap_orig = cv2.VideoCapture(path_original)
    cap_mej = cv2.VideoCapture(path_mejorado)

    if not cap_orig.isOpened() or not cap_mej.isOpened():
        print("Error al abrir los archivos de video.")
        return None

    resultados = []
    
    try:
        for frame_num in frame_numbers:
            print(f"Procesando fotograma {frame_num}...")
            
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            cap_mej.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            ret_o, frame_o = cap_orig.read()
            ret_m, frame_m = cap_mej.read()
            
            if not ret_o or not ret_m:
                print(f"Advertencia: No se pudo leer el fotograma {frame_num} de uno o ambos videos.")
                continue
                
            gris_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
            gris_m = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
            
            if gris_o.shape != gris_m.shape:
                print(f"Error: Los fotogramas {frame_num} tienen dimensiones diferentes. "
                      f"Original: {gris_o.shape}, Mejorado: {gris_m.shape}")
                print("Las métricas no son comparables. Abortando.")
                return None
            
            # --- ¡NUEVO! GENERAR Y GUARDAR HISTOGRAMA ---
            # 1. Definir el nombre del archivo de salida para el plot
            plot_filename = f"histograma_comparativo_frame_{frame_num}.png"
            # 2. Crear la ruta completa usando el directorio base
            ruta_salida_plot = os.path.join(output_dir, plot_filename)
            # 3. Llamar a la función que crea y guarda el gráfico
            guardar_histograma_comparativo(gris_o, gris_m, frame_num, ruta_salida_plot)
            print(f"  ... Histograma guardado en: {plot_filename}")
            
            # --- Cálculo de métricas (sin cambios) ---
            metricas_o = {
                'video': 'original', 'frame': frame_num, 'media': calcular_media(gris_o),
                'varianza': calcular_varianza(gris_o), 'curtosis': calcular_curtosis(gris_o),
                'entropia': calcular_entropia(gris_o), 'energia': calcular_energia_pixeles(gris_o),
                'densidad_bordes': calcular_densidad_bordes(gris_o),
                'contraste_textura': calcular_contraste_textura(gris_o),
            }
            resultados.append(metricas_o)
            
            metricas_m = {
                'video': 'mejorado', 'frame': frame_num, 'media': calcular_media(gris_m),
                'varianza': calcular_varianza(gris_m), 'curtosis': calcular_curtosis(gris_m),
                'entropia': calcular_entropia(gris_m), 'energia': calcular_energia_pixeles(gris_m),
                'densidad_bordes': calcular_densidad_bordes(gris_m),
                'contraste_textura': calcular_contraste_textura(gris_m),
            }
            resultados.append(metricas_m)
            
    finally:
        cap_orig.release()
        cap_mej.release()
        print("Análisis completado. Liberando videos.")

    df = pd.DataFrame(resultados)
    df_pivot = df.pivot(index='frame', columns='video')
    df_pivot = df_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    
    return df_pivot

# --- EJECUCIÓN DEL SCRIPT (MODIFICADO) ---

if __name__ == "__main__":
    
    root = tk.Tk()
    root.withdraw()
    
    print("Abriendo explorador para seleccionar el VIDEO ORIGINAL...")
    PATH_VIDEO_ORIGINAL = filedialog.askopenfilename(
        title="Selecciona el VIDEO ORIGINAL",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
    )
    
    if not PATH_VIDEO_ORIGINAL:
        print("Operación cancelada. No se seleccionó video original.")
    else:
        print(f"Video original seleccionado: {PATH_VIDEO_ORIGINAL}")
        
        print("Abriendo explorador para seleccionar el VIDEO MEJORADO...")
        PATH_VIDEO_MEJORADO = filedialog.askopenfilename(
            title="Selecciona el VIDEO MEJORADO",
            filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
        )
        
        if not PATH_VIDEO_MEJORADO:
            print("Operación cancelada. No se seleccionó video mejorado.")
        else:
            print(f"Video mejorado seleccionado: {PATH_VIDEO_MEJORADO}")
            
            FRAMES_A_EVALUAR = [10, 15, 20, 25, 30]
            
            print("\nIniciando análisis de métricas de video...")
            
            pd.set_option('display.float_format', '{:.4f}'.format)
            pd.set_option('display.width', 1000)

            # ¡MODIFICADO! Pasamos el BASE_DIR a la función
            tabla_resultados = analizar_videos(
                PATH_VIDEO_ORIGINAL, 
                PATH_VIDEO_MEJORADO, 
                FRAMES_A_EVALUAR, 
                BASE_DIR  # <--- ¡Aquí está la magia!
            )
            
            # --- ¡MODIFICADO! GUARDAR SOLO EXCEL EN LA RUTA CORRECTA ---
            if tabla_resultados is not None:
                
                # 1. Definir la ruta de salida completa para el Excel
                ruta_salida_excel = os.path.join(BASE_DIR, "comparativa_videos_metricas.xlsx")
                
                # 2. Guardar en Excel
                try:
                    tabla_resultados.to_excel(ruta_salida_excel)
                    print("\n" + "="*80)
                    print(f"✅ ¡Resultados de métricas guardados en: '{ruta_salida_excel}'!")
                    print("="*80)
                except Exception as e:
                    print(f"\n❌ Error al guardar el archivo Excel: {e}")
                
                # 3. (Eliminamos la parte de CSV)
                
                # 4. Imprimir en la terminal
                print("\n         Resultados del Análisis (Terminal)")
                print("-" * 80)
                print(tabla_resultados)
                print("\n" + "="*80)