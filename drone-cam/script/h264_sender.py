import subprocess
import sys

import cv2

# GCS_IP = "192.168.137.1"
GCS_IP = "192.168.137.1"
GCS_PORT = 5001

FPS = 20
WIDTH = 640
HEIGHT = 480

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Gagal membuka camera")
    sys.exit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = camera.read()

if not ret:
    print("Gagal membaca frame dari camera")
    camera.release()
    sys.exit(1)

height, width = frame.shape[:2]

print(f"Camera berhasil dibuka")
print(f"Resolution: {width}x{height}")
print(f"Target FPS: {FPS}")
print(f"Streaming H.264 ke {GCS_IP}:{GCS_PORT}")

ffmpeg_command = [
    "ffmpeg",
    # Input raw BGR frames dari Python
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    f"{width}x{height}",
    "-r",
    str(FPS),
    "-i",
    "-",
    # H.264 encoder (CPU)
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-bf",
    "0",
    "-g",
    "20",
    "-rc",
    "cbr",
    # Video format
    "-pix_fmt",
    "yuv420p",
    # Bitrate
    "-b:v",
    "1.5M",
    "-maxrate",
    "1.5M",
    # MPEG-TS melalui UDP, flush tiap paket agar delay kecil
    "-flush_packets",
    "1",
    "-f",
    "mpegts",
    f"udp://{GCS_IP}:{GCS_PORT}?pkt_size=1316",
]

try:
    encoder = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

except FileNotFoundError:
    print("ERROR: FFmpeg tidak ditemukan.")
    print("Pastikan command 'ffmpeg' tersedia di terminal.")
    camera.release()
    sys.exit(1)

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Gagal membaca frame")
            break

        # Pastikan ukuran frame sesuai
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        # Kirim raw frame ke FFmpeg
        encoder.stdin.write(frame.tobytes())
        encoder.stdin.flush()

        # Preview lokal dimatikan agar sender tidak lag
        # (tekan Ctrl+C untuk berhenti)

except BrokenPipeError:
    print("FFmpeg encoder berhenti.")

except KeyboardInterrupt:
    print("\nStreaming dihentikan.")

finally:
    if encoder.stdin:
        encoder.stdin.close()

    encoder.wait()

    camera.release()
    cv2.destroyAllWindows()

    print("H.264 sender berhenti.")
