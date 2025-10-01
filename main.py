# main.py (Versión Final Completa: Reescalado IA + Compresión Condicional CPU/GPU)

# --- IMPORTS ---
import os
import uuid
import shutil
import json
import subprocess
import re
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from RealESRGAN import RealESRGAN

# --- CONFIGURACIÓN DE DIRECTORIOS Y COMPRESIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEODIR = os.path.join(BASE_DIR, "Uploads/")
PROCESSEDDIR = os.path.join(BASE_DIR, "Process/")
AUXDIR_INPUT = os.path.join(BASE_DIR, "auxiliar/input/")
AUXDIR_OUTPUT = os.path.join(BASE_DIR, "auxiliar/output/")
FINALDIR = os.path.join(BASE_DIR, "FinalCompressed/")

# --- PARÁMETROS DE COMPRESIÓN ---
# GPU (NVENC)
PRESET_NVENC = 'p5'
CQ_VALUE = '31'
# CPU (libx264)
PRESET_CPU = 'veryfast'
CRF_VALUE = '28'

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
    # ... (sin cambios)
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
    # ... (sin cambios)
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
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        processed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        out.write(processed_frame)
    cap.release(); out.release()

def comprimir_video(input_path: str, output_path: str):
    cmd = []
    
    if torch.cuda.is_available():
        print("✅ GPU detectada. Usando compresión acelerada con NVENC.")
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-i', input_path,
            '-c:v', 'h264_nvenc',
            '-preset', PRESET_NVENC,
            '-cq', CQ_VALUE,
            '-c:a', 'copy',
            output_path
        ]
    else:
        print("🐢 No se detectó GPU. Usando compresión estándar con CPU (puede ser lento).")
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', PRESET_CPU,
            '-crf', CRF_VALUE,
            '-c:a', 'copy',
            output_path
        ]

    print("Iniciando compresión con FFmpeg...")
    start_time = time.time()
    
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()

    end_time = time.time()
    duration = end_time - start_time
    
    print("Compresión con FFmpeg finalizada.")
    print(f"Tiempo de compresión: {duration:.2f} segundos")
    
    if process.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Error durante la compresión con FFmpeg. Código de salida: {process.returncode}")

def cleanup_files(files_to_delete: list):
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)

# --- ENDPOINT PRINCIPAL ---
@app.post("/uploadfile/")
async def SubirVideo(
    background_tasks: BackgroundTasks,
    noise: bool = True,
    contrast: bool = True,
    rescale: bool = True,  # <-- NUEVO PARÁMETRO
    file: UploadFile = File(...)
):
    
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
    
    video_para_comprimir = original_filePath # Por defecto, el video original
    
    ################################# Filtros de CPU ##############################################################
    processed_filePath = original_filePath
    if noise:
        eliminateNoise(processed_filePath, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, uniqueName)
    
    if contrast:
        contraste(processed_filePath, noise, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")
        
    video_para_comprimir = processed_filePath
    
    ################################# Reescalado con IA (Solo GPU) ################################################
    uncompressed_output_video = ""
    if rescale:
        if not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="El reescalado con IA solo está disponible en modo GPU.")
        
        print("\n--- INICIANDO FASE 3: REESCALADO CON IA (GPU) ---")
        
        # --- Limpieza de directorios auxiliares ---
        if os.path.exists(AUXDIR_INPUT): shutil.rmtree(AUXDIR_INPUT)
        if os.path.exists(AUXDIR_OUTPUT): shutil.rmtree(AUXDIR_OUTPUT)
        os.makedirs(AUXDIR_INPUT, exist_ok=True)
        os.makedirs(AUXDIR_OUTPUT, exist_ok=True)
        
        # --- División del Video en Frames ---
        vidcap = cv2.VideoCapture(processed_filePath)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        for i in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Dividiendo video en frames"):
            success, image = vidcap.read()
            if not success: break
            cv2.imwrite(os.path.join(AUXDIR_INPUT, f'{i:08d}.png'), image)
        vidcap.release()

        # --- Proceso de Reescalado ---
        device = torch.device('cuda')
        model = RealESRGAN(device, scale=2)
        model.load_weights('weights/RealESRGAN_x2.pth', download=True)

        for image_name in tqdm(sorted(os.listdir(AUXDIR_INPUT)), desc="Reescalando frames (IA)"):
            if not image_name.lower().endswith('.png'): continue
            
            input_image_path = os.path.join(AUXDIR_INPUT, image_name)
            output_image_path = os.path.join(AUXDIR_OUTPUT, image_name)
            img = Image.open(input_image_path).convert('RGB')
            sr_img = model.predict(img)
            sr_img.save(output_image_path)

        # --- Reconstrucción del Video Reescalado ---
        uncompressed_output_video = os.path.join(PROCESSEDDIR, f"rescaled_{uniqueName}")
        first_image_path = os.path.join(AUXDIR_OUTPUT, sorted(os.listdir(AUXDIR_OUTPUT))[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape
        size = (width, height)
        
        out = cv2.VideoWriter(uncompressed_output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for filename in tqdm(sorted(os.listdir(AUXDIR_OUTPUT)), desc="Reconstruyendo video reescalado"):
            img = cv2.imread(os.path.join(AUXDIR_OUTPUT, filename))
            out.write(img)
        out.release()
        
        video_para_comprimir = uncompressed_output_video # Actualizamos el video que se va a comprimir

    ################################### Parte de compresión ##################################################
    print(f"\n--- INICIANDO FASE 4: COMPRESIÓN FINAL CON FFMPEG ---")
    final_video_path = os.path.join(FINALDIR, f"compressed_{uniqueName}")
    
    comprimir_video(input_path=video_para_comprimir, output_path=final_video_path)
    
    # Limpiamos todos los archivos temporales generados
    files_to_clean = [original_filePath]
    if os.path.exists(os.path.join(PROCESSEDDIR, uniqueName)): files_to_clean.append(os.path.join(PROCESSEDDIR, uniqueName))
    if os.path.exists(os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")): files_to_clean.append(os.path.join(PROCESSEDDIR, f"pre_{uniqueName}"))
    if uncompressed_output_video and os.path.exists(uncompressed_output_video): files_to_clean.append(uncompressed_output_video)
    cleanup_files(files_to_clean)
    
    return FileResponse(path=final_video_path, media_type='video/mp4', filename=f"compressed_{file.filename}")