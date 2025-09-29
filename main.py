# main.py (Versión Final Definitiva - Corregido UnboundLocalError en contraste)

# --- IMPORTS ---
import os
import uuid
import shutil
import json
import subprocess
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from moviepy import VideoFileClip
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
from RealESRGAN import RealESRGAN
from utils import convert_ycbcr_to_rgb

# --- CONFIGURACIÓN DE DIRECTORIOS Y COMPRESIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEODIR = os.path.join(BASE_DIR, "Uploads/")
PROCESSEDDIR = os.path.join(BASE_DIR, "Process/")
AUXDIR_INPUT = os.path.join(BASE_DIR, "auxiliar/input/")
AUXDIR_OUTPUT = os.path.join(BASE_DIR, "auxiliar/output/")
FINALDIR = os.path.join(BASE_DIR, "FinalCompressed/")
WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights/RealESRGAN_x2.pth')

PRESET_NVENC = 'p5'
CQ_VALUE = '31'

app = FastAPI()

# --- FUNCIÓN DE INICIO: CREAR DIRECTORIOS ---
@app.on_event("startup")
async def startup_event():
    os.makedirs(VIDEODIR, exist_ok=True)
    os.makedirs(PROCESSEDDIR, exist_ok=True)
    os.makedirs(AUXDIR_INPUT, exist_ok=True)
    os.makedirs(AUXDIR_OUTPUT, exist_ok=True)
    os.makedirs(FINALDIR, exist_ok=True)

# --- FUNCIONES AUXILIARES ---

def eliminateNoise(videoEntrada: str, name: str):
    video_output = os.path.join(PROCESSEDDIR, name)
    cap = cv2.VideoCapture(videoEntrada)
    if not cap.isOpened(): raise IOError("No se pudo abrir el video de entrada para eliminar ruido.")
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Eliminando ruido (CPU)"):
        ret, frame = cap.read()
        if not ret: break
        processed_frame = cv2.bilateralFilter(frame, 9, 75, 75)
        out.write(processed_frame)
    cap.release(); out.release()

def contraste(path: str, noise: bool, video: str):
    ruta_entrada = os.path.join(PROCESSEDDIR, video) if noise else path
    video_output = os.path.join(PROCESSEDDIR, f"pre_{video}")
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened(): raise IOError("No se pudo abrir el video de entrada para ajustar contraste.")
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Ajustando contraste (CPU)"):
        ret, frame = cap.read()
        if not ret: break
        
        # <--- AQUÍ ESTÁ LA CORRECCIÓN ---
        # Primero convertimos y asignamos a 'lab'
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # Luego usamos 'lab' para dividir los canales
        l, a, b = cv2.split(lab)
        
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        processed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        out.write(processed_frame)
    cap.release(); out.release()

def comprimir_gpu(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cmd = [
        'ffmpeg', '-y', '-i', input_path, '-vf', 'scale=3840:-1', '-c:v', 'h264_nvenc',
        '-preset', PRESET_NVENC, '-cq', CQ_VALUE, '-c:a', 'copy', output_path
    ]
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    with tqdm(total=total_frames, desc="Comprimiendo video (GPU)") as pbar:
        for line in process.stderr:
            match = re.search(r"frame=\s*(\d+)", line)
            if match:
                current_frame = int(match.group(1))
                pbar.update(current_frame - pbar.n)
    process.wait()
    if process.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Error durante la compresión con FFmpeg. Código de salida: {process.returncode}")

def cleanup_files(files_to_delete: list):
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)

# --- ENDPOINT PRINCIPAL ---
@app.post("/uploadfile/")
async def SubirVideo(background_tasks: BackgroundTasks, noise: bool = True, contrast: bool = True, file: UploadFile = File(...)):
    
    ############################### Validaciones ################################################################
    if file.content_type != 'video/mp4': raise HTTPException(status_code=400, detail="Error: formato inválido. Solo se aceptan archivos MP4.")
    uniqueName = f"{uuid.uuid4()}_{file.filename}"
    original_filePath = os.path.join(VIDEODIR, uniqueName)
    with open(original_filePath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    videoDetails = VideoFileClip(original_filePath)
    
    if videoDetails.duration > 180:
        cleanup_files([original_filePath])
        videoDetails.close()
        raise HTTPException(status_code=400, detail="Duración máxima de 180 segundos excedida.")
    
    videoDetails.close()
    processed_filePath = original_filePath
    
    ################################# Filtros ####################################################################
    if noise:
        eliminateNoise(processed_filePath, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, uniqueName)
    
    if contrast:
        contraste(processed_filePath, noise, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")
    

    ############################# División del Video ###########################################################
    uncompressed_output_video = os.path.join(PROCESSEDDIR, f"rescaled_{uniqueName}")
    
    if os.path.exists(AUXDIR_INPUT): shutil.rmtree(AUXDIR_INPUT)
    if os.path.exists(AUXDIR_OUTPUT): shutil.rmtree(AUXDIR_OUTPUT)
    os.makedirs(AUXDIR_INPUT, exist_ok=True)
    os.makedirs(AUXDIR_OUTPUT, exist_ok=True)
    
    vidcap = cv2.VideoCapture(processed_filePath)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    for i in range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))):
        success, image = vidcap.read()
        if not success: break
        cv2.imwrite(os.path.join(AUXDIR_INPUT, f'{i:04d}.png'), image)
    vidcap.release()


######################################### Parte de Reescalado ###########################################


# Preparar el modelo y comprobar CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"modelo cargado en {device}")
    model = RealESRGAN(device, scale=2)
    # carga los pesos (usa download=True solo si no los tienes)
    model.load_weights('weights/RealESRGAN_x2.pth', download=True)

    # Procesar frames usando model.predict (entrada: PIL.Image RGB)
    input_images = sorted(os.listdir(AUXDIR_INPUT))
    from tqdm import tqdm

    for image_name in tqdm(input_images, desc="Procesando frames"):
        if not image_name.lower().endswith('.png'):
            continue

        input_image_path = os.path.join(AUXDIR_INPUT, image_name)
        output_image_path = os.path.join(AUXDIR_OUTPUT, image_name)

        try:
            # Abrir como PIL RGB
            img = Image.open(input_image_path).convert('RGB')

            # Opción A (recomendado): procesar todo el RGB con model.predict
            sr_img = model.predict(img)

            # Si usas GPU, sincronizar para asegurarte de que el trabajo termine antes de continuar
            if device.type == 'cuda':
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass

            # Guardar resultado (mismo nombre de archivo)
            sr_img.save(output_image_path)

        except Exception as e:
            # Si falla un frame, lo anotamos y seguimos con el resto
            print(f"ERROR procesando {input_image_path}: {e}")
            import traceback
            traceback.print_exc()
            # opcional: copiar el frame original al directorio de salida para no romper la secuencia
            try:
                Image.open(input_image_path).save(output_image_path)
            except Exception:
                pass

    # Reconstruir el video
    file_list = sorted(os.listdir(AUXDIR_OUTPUT))
    first_image = None

    # Determinar el tamaño a partir del primer frame válido
    for filename in file_list:
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(AUXDIR_OUTPUT, filename))
            if img is not None:
                height, width, layers = img.shape
                size = (width, height)
                first_image = filename
                break

    if first_image is None:
        print("[ERROR] No se encontró ningún frame válido.")
        exit(1)

    out = cv2.VideoWriter(uncompressed_output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for filename in file_list:
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(AUXDIR_OUTPUT, filename))
            if img is None:
                print(f"[WARNING] Frame inválido: {filename}")
                continue
            out.write(img)

    out.release()


################################### Parte de compresión ##################################################
    print(f"\n--- INICIANDO FASE 4: COMPRESIÓN FINAL CON FFMPEG ---")
    final_video_path = os.path.join(FINALDIR, f"compressed_{uniqueName}")
    comprimir_gpu(input_path=uncompressed_output_video, output_path=final_video_path)
    files_to_clean = [original_filePath, uncompressed_output_video]
    if os.path.exists(os.path.join(PROCESSEDDIR, uniqueName)): files_to_clean.append(os.path.join(PROCESSEDDIR, uniqueName))
    if os.path.exists(os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")): files_to_clean.append(os.path.join(PROCESSEDDIR, f"pre_{uniqueName}"))
    cleanup_files(files_to_clean)
    background_tasks.add_task(cleanup_files, [final_video_path])
    return FileResponse(path=final_video_path, media_type='video/mp4', filename=f"compressed_{file.filename}")