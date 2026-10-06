import subprocess
import sys

import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5001

WIDTH = 640
HEIGHT = 480
FPS = 20

FRAME_SIZE = WIDTH * HEIGHT * 3

ffmpeg_command = [
    "ffmpeg",
    # UDP MPEG-TS input
    "-fflags",
    "nobuffer",
    "-flags",
    "low_delay",
    "-i",
    f"udp://{HOST}:{PORT}?fifo_size=1000000&overrun_nonfatal=1",
    # Output raw BGR untuk Python
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "pipe:1",
]

print(f"Listening H.264 UDP on {HOST}:{PORT}")

try:
    decoder = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )

except FileNotFoundError:
    print("ERROR: FFmpeg tidak ditemukan.")
    print("Pastikan command 'ffmpeg' tersedia di terminal.")
    sys.exit(1)

cv2.namedWindow("GCS - H264 Drone Camera", cv2.WINDOW_NORMAL)

cv2.resizeWindow("GCS - H264 Drone Camera", WIDTH, HEIGHT)

try:
    while True:
        raw_frame = decoder.stdout.read(FRAME_SIZE)

        if len(raw_frame) != FRAME_SIZE:
            print("Frame tidak lengkap atau stream berhenti.")
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

        cv2.imshow("GCS - H264 Drone Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nReceiver dihentikan.")

finally:
    decoder.terminate()
    decoder.wait()

    cv2.destroyAllWindows()

    print("H.264 receiver berhenti.")
