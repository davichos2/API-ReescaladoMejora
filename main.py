# main.py (Optimizado: reescalado IA en FP16 in-memory + compresión NVENC con audio original)
import os, uuid, shutil, subprocess, time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from moviepy.editor import VideoFileClip
import cv2, numpy as np
from tqdm import tqdm
import torch, torch.backends.cudnn as cudnn
from PIL import Image
from RealESRGAN import RealESRGAN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEODIR = os.path.join(BASE_DIR, "Uploads/")
PROCESSEDDIR = os.path.join(BASE_DIR, "Process/")
FINALDIR = os.path.join(BASE_DIR, "FinalCompressed/")

PRESET_NVENC = 'medium'
CQ_VALUE = '31'
PRESET_CPU = 'veryfast'
CRF_VALUE = '28'

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    os.makedirs(VIDEODIR, exist_ok=True)
    os.makedirs(PROCESSEDDIR, exist_ok=True)
    os.makedirs(FINALDIR, exist_ok=True)

def eliminateNoise(videoEntrada: str, name: str):
    video_output = os.path.join(PROCESSEDDIR, name)
    cap = cv2.VideoCapture(videoEntrada)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video de entrada para eliminar ruido.")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total), desc="Eliminando ruido (CPU)"):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = cv2.bilateralFilter(frame, 9, 75, 75)
        out.write(processed_frame)
    cap.release(); out.release()

def contraste(path: str, noise: bool, video: str):
    ruta_entrada = os.path.join(PROCESSEDDIR, video) if noise else path
    video_output = os.path.join(PROCESSEDDIR, f"pre_{video}")
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video de entrada para ajustar contraste.")
    width, height = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5)
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total), desc="Ajustando contraste (CPU)"):
        ret, frame = cap.read()
        if not ret:
            break
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        out.write(processed_frame)
    cap.release(); out.release()

def _video_has_audio(path: str) -> bool:
    try:
        clip = VideoFileClip(path)
        has_audio = clip.audio is not None
        clip.close()
        return has_audio
    except Exception:
        return False

def comprimir_video(input_path: str, output_path: str, audio_source: str | None = None):
    use_gpu = torch.cuda.is_available()
    has_audio = False
    if audio_source and os.path.exists(audio_source):
        has_audio = _video_has_audio(audio_source)

    common_flags = ['-y', '-loglevel', 'error']  # muestra solo errores
    if use_gpu:
        if has_audio:
            cmd = ['ffmpeg', *common_flags, '-hwaccel', 'cuda',
                   '-i', input_path, '-i', audio_source,
                   '-map', '0:v:0', '-map', '1:a:0',
                   '-c:v', 'h264_nvenc', '-preset', PRESET_NVENC, '-cq', CQ_VALUE,
                   '-pix_fmt', 'yuv420p',
                   '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                   '-color_range', 'tv',
                   '-c:a', 'aac', '-b:a', '192k',
                   '-shortest', output_path]
        else:
            cmd = ['ffmpeg', *common_flags, '-hwaccel', 'cuda',
                   '-i', input_path,
                   '-c:v', 'h264_nvenc', '-preset', PRESET_NVENC, '-cq', CQ_VALUE,
                   '-pix_fmt', 'yuv420p',
                   '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                   '-color_range', 'tv',
                   '-an', output_path]
    else:
        if has_audio:
            cmd = ['ffmpeg', *common_flags,
                   '-i', input_path, '-i', audio_source,
                   '-map', '0:v:0', '-map', '1:a:0',
                   '-c:v', 'libx264', '-preset', PRESET_CPU, '-crf', CRF_VALUE,
                   '-pix_fmt', 'yuv420p',
                   '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                   '-color_range', 'tv',
                   '-c:a', 'aac', '-b:a', '192k',
                   '-shortest', output_path]
        else:
            cmd = ['ffmpeg', *common_flags,
                   '-i', input_path,
                   '-c:v', 'libx264', '-preset', PRESET_CPU, '-crf', CRF_VALUE,
                   '-pix_fmt', 'yuv420p',
                   '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                   '-color_range', 'tv',
                   '-an', output_path]

    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dur = time.time() - start
    print(f"FFmpeg terminó en {dur:.2f}s, rc={proc.returncode}")
    if proc.returncode != 0:
        msg = proc.stderr.decode('utf-8', errors='ignore') or 'FFmpeg falló sin mensaje'
        raise HTTPException(status_code=500, detail=f"FFmpeg error (rc={proc.returncode}): {msg}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        msg = proc.stderr.decode('utf-8', errors='ignore')
        raise HTTPException(status_code=500, detail=f"Archivo de salida no generado o vacío: {output_path}. FFmpeg: {msg}")


def cleanup_files(files_to_delete: list):
    for f in files_to_delete:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

@app.post("/uploadfile/")
async def SubirVideo(
    background_tasks: BackgroundTasks,
    noise: bool = True,
    contrast: bool = True,
    rescale: bool = True,
    file: UploadFile = File(...)
):
    if file.content_type != 'video/mp4':
        raise HTTPException(status_code=400, detail="Error: formato inválido. Solo se aceptan archivos MP4.")

    uniqueName = f"{uuid.uuid4()}_{file.filename}"
    original_filePath = os.path.join(VIDEODIR, uniqueName)
    with open(original_filePath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    videoDetails = VideoFileClip(original_filePath)
    if videoDetails.duration > 180:
        cleanup_files([original_filePath]); videoDetails.close()
        raise HTTPException(status_code=400, detail="Duración máxima de 180 segundos excedida.")
    videoDetails.close()

    processed_filePath = original_filePath
    video_para_comprimir = original_filePath

    if noise:
        eliminateNoise(processed_filePath, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, uniqueName)
        video_para_comprimir = processed_filePath

    if contrast:
        contraste(processed_filePath, noise, uniqueName)
        processed_filePath = os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")
        video_para_comprimir = processed_filePath

    uncompressed_output_video = ""
    if rescale:
        if not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="El reescalado con IA solo está disponible en modo GPU.")

        print("\n--- INICIANDO FASE 3: REESCALADO CON IA (GPU, FP16, in-memory, batch) ---")
        device = torch.device('cuda')
        cudnn.benchmark = True

        model = RealESRGAN(device, scale=2)
        model.load_weights('weights/RealESRGAN_x2.pth', download=True)
        model.model.eval(); model.model.to(device)

        cap = cv2.VideoCapture(processed_filePath)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="No se pudo abrir el video para reescalar.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w, out_h = in_w * 2, in_h * 2
        uncompressed_output_video = os.path.join(PROCESSEDDIR, f"rescaled_{uniqueName}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(uncompressed_output_video, fourcc, fps, (out_w, out_h))

        batch_size = 8
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in tqdm(range(total_frames), desc="Leyendo frames para batch"):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                # CORRECCIÓN DE COLOR: BGR -> RGB antes de crear el tensor
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float().div(255.0)
                frames.append(frame_tensor)

                if len(frames) == batch_size:
                    batch_tensor = torch.stack(frames).to(device)  # BxCxHxW (RGB)
                    output_batch = model.model(batch_tensor)        # salida en RGB normalizado
                    for i in range(output_batch.size(0)):
                        sr_img = output_batch[i].clamp(0,1).permute(1,2,0).cpu().numpy()
                        sr_img = (sr_img * 255.0).astype(np.uint8)   # RGB uint8
                        # Volver a BGR para VideoWriter de OpenCV
                        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                        out.write(sr_img_bgr)
                    frames = []

            if len(frames) > 0:
                batch_tensor = torch.stack(frames).to(device)
                output_batch = model.model(batch_tensor)
                for i in range(output_batch.size(0)):
                    sr_img = output_batch[i].clamp(0,1).permute(1,2,0).cpu().numpy()
                    sr_img = (sr_img * 255.0).astype(np.uint8)
                    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                    out.write(sr_img_bgr)

        cap.release(); out.release()
        video_para_comprimir = uncompressed_output_video

    print("\n--- INICIANDO FASE 4: COMPRESIÓN FINAL CON FFMPEG ---")
    final_video_path = os.path.join(FINALDIR, f"compressed_{uniqueName}")
    audio_source = original_filePath
    comprimir_video(input_path=video_para_comprimir, output_path=final_video_path, audio_source=audio_source)

    files_to_clean = [original_filePath]
    p1 = os.path.join(PROCESSEDDIR, uniqueName)
    p2 = os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")
    if os.path.exists(p1): files_to_clean.append(p1)
    if os.path.exists(p2): files_to_clean.append(p2)
    if uncompressed_output_video and os.path.exists(uncompressed_output_video): files_to_clean.append(uncompressed_output_video)
    for f in files_to_clean:
        try: os.remove(f)
        except Exception: pass

    if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
        raise HTTPException(status_code=500, detail="La compresión no produjo un archivo válido.")
    return FileResponse(path=final_video_path, media_type='video/mp4', filename=f"compressed_{file.filename}")
