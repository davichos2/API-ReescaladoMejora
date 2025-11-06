# main.py (SRCNN x2 CORREGIDO - GRAYSCALE)
import os, uuid, shutil, subprocess, time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from moviepy.editor import VideoFileClip
import cv2, numpy as np
from tqdm import tqdm
import torch, torch.backends.cudnn as cudnn
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEODIR = os.path.join(BASE_DIR, "Uploads/")
PROCESSEDDIR = os.path.join(BASE_DIR, "Process/")
FINALDIR = os.path.join(BASE_DIR, "FinalCompressed/")

PRESET_NVENC = 'medium'
CQ_VALUE = '31'
PRESET_CPU = 'veryfast'
CRF_VALUE = '28'

app = FastAPI()

# ==================== MODELO SRCNN (GRAYSCALE) ====================
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4)
        )
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2)
        )
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.features(x))
        x = self.relu(self.map(x))
        x = self.reconstruction(x)
        return x

class SRCNNModel:
    def __init__(self, device):
        self.device = device
        self.model = SRCNN().to(device)

    def load_weights(self, weights_path):
        if os.path.exists(weights_path):
            print(f"✅ Cargando pesos SRCNN desde {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Pesos SRCNN cargados exitosamente")
        else:
            raise FileNotFoundError(f"❌ No se encontró: {weights_path}")

# ======================================================

@app.on_event("startup")
async def startup_event():
    os.makedirs(VIDEODIR, exist_ok=True)
    os.makedirs(PROCESSEDDIR, exist_ok=True)
    os.makedirs(FINALDIR, exist_ok=True)

def eliminateNoise(videoEntrada: str, name: str, d: int = 9, sigmaColor: int = 75, sigmaSpace: int = 75):
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
        processed_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
        out.write(processed_frame)
    cap.release(); out.release()


def contraste(path: str, noise: bool, video: str, clipLimit: float = 2.0, tileGridSize: tuple = (8, 8)):
    ruta_entrada = os.path.join(PROCESSEDDIR, video) if noise else path
    video_output = os.path.join(PROCESSEDDIR, f"pre_{video}")
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video de entrada para ajustar contraste.")
    width, height = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5)
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
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

def resize_video_ffmpeg(input_path, output_path, target_w, target_h):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)


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

    common_flags = ['-y', '-loglevel', 'error']
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
    print("--- INICIANDO LIMPIEZA DE ARCHIVOS (DESACTIVADA PARA PRUEBAS) ---")

@app.post("/uploadfile/")
async def SubirVideo(
    background_tasks: BackgroundTasks,
    noise: bool = True,
    noise_d: int = 9,
    noise_sigmaColor: int = 75,
    noise_sigmaSpace: int = 75,
    contrast: bool = True,
    contrast_clipLimit: float = 2.0,
    contrast_tileGridSize: str = "8,8",
    rescale: bool = True,
    file: UploadFile = File(...)
):
    
    tileGrid = tuple(map(int, contrast_tileGridSize.split(",")))
    if file.content_type != 'video/mp4':
        raise HTTPException(status_code=400, detail="Error: formato inválido. Solo se aceptan archivos MP4.")

    uniqueName = f"{uuid.uuid4()}_{file.filename}"
    original_filePath = os.path.join(VIDEODIR, uniqueName)
    with open(original_filePath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    videoDetails = VideoFileClip(original_filePath)
    if videoDetails.duration > 180:
        try: os.remove(original_filePath) 
        except: pass
        videoDetails.close()
        raise HTTPException(status_code=400, detail="Duración máxima de 180 segundos excedida.")
    videoDetails.close()

    temp_audio = os.path.join(PROCESSEDDIR, f"audio_{uniqueName}.aac")
    subprocess.run(['ffmpeg', '-y', '-i', original_filePath, '-vn', '-acodec', 'aac', temp_audio], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    audio_source = temp_audio

    processed_filePath = original_filePath
    video_para_comprimir = original_filePath

    cap = cv2.VideoCapture(processed_filePath)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir el video para reescalar.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    standard_resolutions = [360, 480, 720, 1080, 1440, 2160]
    target_h = next((h for h in standard_resolutions if h > in_h), None)
    if target_h is None:
        target_h = in_h

    scale = 2
    input_h_for_model = target_h // scale
    input_w_for_model = int(input_h_for_model * (in_w / in_h))
    input_w_for_model += input_w_for_model % 2
    input_h_for_model += input_h_for_model % 2
    
    resized_file = os.path.join(PROCESSEDDIR, f"resized_{uniqueName}")
    resize_video_ffmpeg(processed_filePath, resized_file, input_w_for_model, input_h_for_model)
    processed_filePath = resized_file
    video_para_comprimir = processed_filePath

    try:
        cap.release()
        os.remove(original_filePath)
    except FileNotFoundError:
        pass
    cap = cv2.VideoCapture(processed_filePath)

    if noise:
        eliminateNoise(processed_filePath, uniqueName, d=noise_d, sigmaColor=noise_sigmaColor, sigmaSpace=noise_sigmaSpace)
        processed_filePath = os.path.join(PROCESSEDDIR, uniqueName)
        video_para_comprimir = processed_filePath

    if contrast:
        contraste(processed_filePath, noise, uniqueName, clipLimit=contrast_clipLimit, tileGridSize=tileGrid)
        processed_filePath = os.path.join(PROCESSEDDIR, f"pre_{uniqueName}")
        video_para_comprimir = processed_filePath

    uncompressed_output_video = ""
    if rescale:
        if not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="El reescalado con IA solo está disponible en modo GPU.")

        print("\n--- INICIANDO FASE 3: REESCALADO CON SRCNN (GPU, FP16, YCrCb, batch) ---")
        device = torch.device('cuda')
        cudnn.benchmark = True

        model = SRCNNModel(device)
        model.load_weights('weights/srcnn_x2-T91-7d6e0623.pth.tar')
        model.model.eval()
        model.model.to(device)

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

        batch_size = 2
        frames_y = []
        frames_cr = []
        frames_cb = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in tqdm(range(total_frames), desc="Procesando con SRCNN (YCrCb)"):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                
                # Bicubic upscale primero
                frame_upscaled = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                
                # BGR -> YCrCb
                frame_ycrcb = cv2.cvtColor(frame_upscaled, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(frame_ycrcb)
                
                # Procesar solo canal Y con el modelo
                y_tensor = torch.from_numpy(y).unsqueeze(0).float().div(255.0)  # 1xHxW
                frames_y.append(y_tensor)
                frames_cr.append(cr)
                frames_cb.append(cb)

                if len(frames_y) == batch_size:
                    batch_y = torch.stack(frames_y).to(device)  # Bx1xHxW
                    output_y = model.model(batch_y)
                    
                    for i in range(output_y.size(0)):
                        y_sr = output_y[i].clamp(0,1).squeeze(0).cpu().numpy()
                        y_sr = (y_sr * 255.0).astype(np.uint8)
                        
                        # Recombinar Y procesado con Cr y Cb originales
                        ycrcb_sr = cv2.merge([y_sr, frames_cr[i], frames_cb[i]])
                        bgr_sr = cv2.cvtColor(ycrcb_sr, cv2.COLOR_YCrCb2BGR)
                        out.write(bgr_sr)
                    
                    frames_y = []
                    frames_cr = []
                    frames_cb = []

            if len(frames_y) > 0:
                batch_y = torch.stack(frames_y).to(device)
                output_y = model.model(batch_y)
                
                for i in range(output_y.size(0)):
                    y_sr = output_y[i].clamp(0,1).squeeze(0).cpu().numpy()
                    y_sr = (y_sr * 255.0).astype(np.uint8)
                    ycrcb_sr = cv2.merge([y_sr, frames_cr[i], frames_cb[i]])
                    bgr_sr = cv2.cvtColor(ycrcb_sr, cv2.COLOR_YCrCb2BGR)
                    out.write(bgr_sr)

        cap.release(); out.release()
        video_para_comprimir = uncompressed_output_video

    print("\n--- INICIANDO FASE 4: COMPRESIÓN FINAL CON FFMPEG ---")
    final_video_path = os.path.join(FINALDIR, f"compressed_{uniqueName}")
    comprimir_video(input_path=video_para_comprimir, output_path=final_video_path, audio_source=audio_source)

    if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
        raise HTTPException(status_code=500, detail="La compresión no produjo un archivo válido.")
    
    return FileResponse(path=final_video_path, media_type='video/mp4', filename=f"compressed_{file.filename}")
