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
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

# Asumiendo que estos archivos existen en tu proyecto y están en el mismo directorio
from models import ESPCN
from utils import convert_ycbcr_to_rgb

# --- CONFIGURACIÓN DE DIRECTORIOS Y COMPRESIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEODIR = os.path.join(BASE_DIR, "Uploads/")
PROCESSEDDIR = os.path.join(BASE_DIR, "Process/")
AUXDIR_INPUT = os.path.join(BASE_DIR, "auxiliar/input/")
AUXDIR_OUTPUT = os.path.join(BASE_DIR, "auxiliar/output/")
FINALDIR = os.path.join(BASE_DIR, "FinalCompressed/")
WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights/espcn_x3.pth')

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
    if noise:
        eliminateNoise(processed_filePath, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, uniqueName)
    if contrast:
        contraste(processed_filePath, noise, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo para PyTorch: {device}")
    cudnn.benchmark = True
    model = ESPCN(scale_factor=3).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    input_images = sorted(os.listdir(AUXDIR_INPUT))
    for image_name in tqdm(input_images, desc="Reescalando frames (GPU)"):
        if not image_name.lower().endswith('.png'): continue
        img = Image.open(os.path.join(AUXDIR_INPUT, image_name)).convert('RGB')
        y, cb, cr = img.convert('YCbCr').split()
        inp = torch.from_numpy(np.array(y).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(): pred = model(inp).clamp(0.0, 1.0)
        pred_np = pred.mul(255.0).cpu().numpy().squeeze().astype(np.uint8)
        H, W = pred_np.shape
        cb_up, cr_up = cb.resize((W, H), Image.BICUBIC), cr.resize((W, H), Image.BICUBIC)
        rgb_out = convert_ycbcr_to_rgb(np.stack([pred_np, np.array(cb_up), np.array(cr_up)], axis=2))
        out_img = Image.fromarray(np.clip(rgb_out, 0, 255).astype(np.uint8))
        out_img.save(os.path.join(AUXDIR_OUTPUT, image_name))
    first_image = cv2.imread(os.path.join(AUXDIR_OUTPUT, sorted(os.listdir(AUXDIR_OUTPUT))[0]))
    height, width, _ = first_image.shape
    out = cv2.VideoWriter(uncompressed_output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for filename in sorted(os.listdir(AUXDIR_OUTPUT)):
        if filename.lower().endswith('.png'): out.write(cv2.imread(os.path.join(AUXDIR_OUTPUT, filename)))
    out.release()
    print(f"\n--- INICIANDO FASE 4: COMPRESIÓN FINAL CON FFMPEG ---")
    final_video_path = os.path.join(FINALDIR, f"compressed_{uniqueName}")
    comprimir_gpu(input_path=uncompressed_output_video, output_path=final_video_path)
    files_to_clean = [original_filePath, uncompressed_output_video]
    if os.path.exists(os.path.join(PROCESSEDDIR, uniqueName)): files_to_clean.append(os.path.join(PROCESSEDDIR, uniqueName))
    if os.path.exists(os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")): files_to_clean.append(os.path.join(PROCESSEDDIR, f"pre_{uniqueName}"))
    cleanup_files(files_to_clean)
    background_tasks.add_task(cleanup_files, [final_video_path])
    return FileResponse(path=final_video_path, media_type='video/mp4', filename=f"compressed_{file.filename}")