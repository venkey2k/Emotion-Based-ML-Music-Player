import eel
import os
import base64
import time
import cv2
import numpy as np
import threading
import collections

os.environ['TF_USE_LEGACY_KERAS'] = '1'

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1

# ┌──────────────────────────────────────────────┐
# │  IMPORT CUSTOM MODULES                        │
# └──────────────────────────────────────────────┘
from watcher import SongScanner, LibraryWatcher
from emotion_detector import EmotionDetector, DetectorConfig

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CONFIG = {
    "camera_index": 0,
    "max_fps": 30,
    "stream_quality": 60,
    "model_path": "keras/Emotion_Detection.h5",
    "cascade_path": "keras/haarcascade_frontalface_default.xml",
}

CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry':    (0, 0, 255),
    'happy':    (0, 255, 255),
    'neutral':  (0, 255, 0),
    'sad':      (255, 0, 0),
    'surprise': (255, 0, 255),
}

# ──────────────────────────────────────────────
# INITIALIZE EEL
# ──────────────────────────────────────────────
eel.init('.')

# ──────────────────────────────────────────────
# EMOTION DETECTOR (v7 with all fixes)
# ──────────────────────────────────────────────
print("=" * 55)
print("  Moodify — Emotion Music Player")
print("  Using Emotion Detector v7")
print("=" * 55)

detector = None


def init_models():
    """Initialize the v7 emotion detector."""
    global detector

    if detector is not None:
        return

    print("⏳ Loading emotion detection models...")

    # Optional: customize detector config
    cfg = DetectorConfig(
        tta_enabled=True,
        clahe_enabled=True,
        class_boost_enabled=True,
        geo_boost_enabled=True,
        multi_crop_enabled=True,
        ema_alpha=0.50,
    )

    detector = EmotionDetector(
        model_path=CONFIG["model_path"],
        cascade_path=CONFIG["cascade_path"],
        config=cfg,
    )

    print("✔  Emotion Detector v7 ready")
    print("   Fixes: TTA + CLAHE + ClassBoost + GeoBoost")
    print("=" * 55)


# ──────────────────────────────────────────────
# CAMERA MANAGER — Uses v7 Detector
# ──────────────────────────────────────────────
class CameraManager:
    """
    Manages webcam access and emotion detection.
    Uses EmotionDetector v7 for all predictions.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.cap = None
        self._streaming = False
        self._thread = None

        # Current state (exposed to eel)
        self._current_emotion = "neutral"
        self._current_confidence = 0.0
        self._current_all_scores = {}
        self._fps = 0.0

    def open(self):
        """Open webcam if not already open."""
        if self.cap is None or not self.cap.isOpened():
            print("📷 Opening Camera...")
            self.cap = cv2.VideoCapture(CONFIG["camera_index"])
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            # Warmup frames
            for _ in range(5):
                self.cap.read()

    def close(self):
        """Release webcam."""
        self.stop_stream()
        if self.cap is not None:
            print("📷 Releasing Camera...")
            self.cap.release()
            self.cap = None

    def read_frame(self):
        """Read and mirror a single frame."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # mirror

    # ══════════════════════════════════════════════
    # SINGLE-SHOT DETECTION (for getEmotion)
    # Uses v7 detector's batch mode
    # ══════════════════════════════════════════════

    def detect_emotion_once(self, num_samples=7):
        """
        Capture multiple frames, run v7 detector,
        return the smoothed emotion.
        """
        init_models()
        self.open()

        # Collect frames
        frames = []
        for _ in range(num_samples):
            frame = self.read_frame()
            if frame is not None:
                frames.append(frame)

        if not frames:
            return "neutral", 0.0, {}

        # Use detector's batch mode (resets smoother internally)
        emotion, confidence, all_scores = detector.detect_batch(frames)

        print(
            f"🎭  Detected: {emotion} ({confidence * 100:.1f}%) "
            f"— {all_scores}"
        )

        return emotion, confidence, all_scores

    # ══════════════════════════════════════════════
    # LIVE STREAM (for camera feed)
    # Uses v7 detector per frame
    # ══════════════════════════════════════════════

    def _stream_loop(self):
        """Background thread: continuously process + stream frames."""
        init_models()
        detector.reset()  # fresh smoother for stream

        prev_time = time.time()

        while self._streaming:
            frame = self.read_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # FPS calculation
            now = time.time()
            self._fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── Run v7 detector ──────────────────
            result = detector.detect(frame)

            self._current_emotion = result['emotion']
            self._current_confidence = result['confidence']
            self._current_all_scores = result['all_scores']

            # ── Draw on frame ────────────────────
            bbox = result.get('bbox')
            if bbox:
                x, y, w, h = bbox
                emotion = result['emotion']
                conf = result['confidence']
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))

                # Corner-style bounding box
                cl = int(min(w, h) * 0.2)
                corners = [
                    (x, y, cl, 0, 0, cl),
                    (x+w, y, -cl, 0, 0, cl),
                    (x, y+h, cl, 0, 0, -cl),
                    (x+w, y+h, -cl, 0, 0, -cl),
                ]
                for cx, cy, dx, dy, ex, ey in corners:
                    cv2.line(
                        frame, (cx, cy), (cx+dx, cy+dy),
                        color, 3
                    )
                    cv2.line(
                        frame, (cx, cy), (cx+ex, cy+ey),
                        color, 3
                    )

                # Emotion label
                cv2.putText(
                    frame,
                    f"{emotion.upper()} {conf*100:.0f}%",
                    (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2,
                )

                # Score bars
                self._draw_score_bars(
                    frame, result['all_scores'],
                    x + w + 15, y
                )

            # FPS overlay
            cv2.putText(
                frame, f"FPS: {self._fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            # Calibration indicator
            cal_text = "CAL: OK" if detector.is_calibrated else "CAL: --"
            cal_color = (0, 255, 0) if detector.is_calibrated else (0, 0, 255)
            cv2.putText(
                frame, cal_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1,
            )

            # No face warning
            if result['faces_count'] == 0:
                fh, fw = frame.shape[:2]
                cv2.putText(
                    frame, "No face detected",
                    (fw // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            # ── Encode + send to frontend ────────
            _, buf = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, CONFIG["stream_quality"]]
            )
            b64 = base64.b64encode(buf).decode('utf-8')
            try:
                eel.updateFrame(
                    f"data:image/jpeg;base64,{b64}"
                )()
            except Exception:
                pass

            # Cap FPS
            elapsed = time.time() - now
            sleep_time = max(
                0, (1.0 / CONFIG["max_fps"]) - elapsed
            )
            time.sleep(sleep_time)

    @staticmethod
    def _draw_score_bars(frame, scores, x, y):
        """Draw emotion score bars beside the face."""
        fh, fw = frame.shape[:2]
        if x + 170 > fw:
            return  # no space

        for i, label in enumerate(CLASS_LABELS):
            by = y + i * 22
            val = scores.get(label, 0) / 100.0
            color = EMOTION_COLORS.get(label, (180, 180, 180))

            # Label
            cv2.putText(
                frame, label[:3].upper(), (x, by + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
            )

            # Bar background
            bx = x + 40
            cv2.rectangle(
                frame, (bx, by), (bx + 100, by + 14),
                (40, 40, 40), -1,
            )

            # Bar fill
            fill = int(100 * val)
            if fill > 0:
                cv2.rectangle(
                    frame, (bx, by), (bx + fill, by + 14),
                    color, -1,
                )

            # Percentage
            cv2.putText(
                frame, f"{val*100:.0f}%", (bx + 104, by + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
            )

    def start_stream(self):
        """Start background streaming thread."""
        if self._streaming:
            return
        self.open()
        self._streaming = True
        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True
        )
        self._thread.start()
        print("▶ Stream started (v7 detector)")

    def stop_stream(self):
        """Stop streaming."""
        if not self._streaming:
            return
        self._streaming = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self.close()
        print("⏹ Stream stopped")


# Global singleton
cam_mgr = CameraManager()


# ──────────────────────────────────────────────
# EEL EXPOSED — Camera / Emotion
# ──────────────────────────────────────────────
@eel.expose
def getEmotion():
    """Single-shot emotion detection with v7 accuracy."""
    emotion, conf, _ = cam_mgr.detect_emotion_once()
    return emotion


@eel.expose
def startCamera():
    cam_mgr.start_stream()


@eel.expose
def stopCamera():
    cam_mgr.stop_stream()


@eel.expose
def getAvailableEmotions():
    return CLASS_LABELS


@eel.expose
def calibrateNeutral():
    """
    Calibrate neutral baseline from current webcam frame.
    Call when user shows neutral face.
    Returns True if successful.
    """
    init_models()
    cam_mgr.open()
    frame = cam_mgr.read_frame()
    if frame is None:
        return False
    return detector.calibrate_neutral(frame)


# ┌──────────────────────────────────────────────┐
# │  SONG MANAGEMENT — Uses watcher module        │
# └──────────────────────────────────────────────┘
scanner = SongScanner(songs_dir='songs', images_dir='images')
watcher = None


def notify_frontend(songs):
    """Push updated song list to JavaScript."""
    try:
        eel.onLibraryChanged(songs)()
    except Exception as e:
        print(f"⚠️  Could not push to frontend: {e}")


@eel.expose
def get_all_songs():
    return scanner.get_songs()


@eel.expose
def upload_song(file_data_b64, metadata):
    """Upload handler — watcher auto-detects the new file."""
    try:
        songs_dir = scanner.songs_dir

        mood = metadata['moods'][0] if metadata['moods'] else 'neutral'
        mood_folder = os.path.join(songs_dir, mood)
        os.makedirs(mood_folder, exist_ok=True)

        header, encoded = file_data_b64.split(',', 1)
        data = base64.b64decode(encoded)

        safe_title = "".join(
            c for c in metadata['title']
            if c.isalnum() or c in (' ', '-', '_')
        ).strip()
        filename = f"{safe_title}.mp3"
        file_path = os.path.join(mood_folder, filename)

        if os.path.exists(file_path):
            filename = f"{safe_title}_{int(time.time())}.mp3"
            file_path = os.path.join(mood_folder, filename)

        with open(file_path, 'wb') as f:
            f.write(data)

        print(f"✔ File saved: {file_path}")

        try:
            audio = MP3(file_path, ID3=ID3)
            try:
                audio.add_tags()
            except Exception:
                pass
            audio.tags.add(TIT2(encoding=3, text=metadata['title']))
            audio.tags.add(
                TPE1(encoding=3, text=metadata['description'])
            )
            audio.save()
            print("✔ Tags updated")
        except Exception as e:
            print(f"⚠ Tag error: {e}")

        scanner.mark_dirty()
        return {'success': True}

    except Exception as e:
        print(f"❌ Upload Error: {e}")
        return {'success': False, 'message': str(e)}


# ┌──────────────────────────────────────────────┐
# │  YOUTUBE DOWNLOADER INTEGRATION              │
# └──────────────────────────────────────────────┘
from youtube_dl_module import YouTubeMP3Downloader

@eel.expose
def get_youtube_info(url):
    """Fetch video metadata (Title, Thumbnail) without downloading."""
    try:
        downloader = YouTubeMP3Downloader()
        info = downloader.get_video_info(url)
        if info:
            return {'success': True, 'data': info}
        return {'success': False, 'message': 'Could not fetch info'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

@eel.expose
def download_youtube_song(url, mood):
    """
    Downloads audio from a YouTube URL, tags it, and saves it
    to the specific mood folder.
    """
    try:
        print(f"📥 YouTube Download Request: {url} [{mood}]")
        
        # 1. Determine Output Directory
        target_dir = os.path.join(scanner.songs_dir, mood)
        os.makedirs(target_dir, exist_ok=True)
        
        # 2. Initialize Downloader
        downloader = YouTubeMP3Downloader(output_dir=target_dir)
        
        # 3. Process
        result = downloader.process(url)
        
        if result.startswith("Success"):
            scanner.mark_dirty()  # refreshing the library
            return {'success': True, 'message': result}
        else:
            return {'success': False, 'message': result}
            
    except Exception as e:
        print(f"❌ YouTube Error: {e}")
        return {'success': False, 'message': str(e)}


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    init_models()

    # Start file watcher
    watcher = LibraryWatcher(
        scanner=scanner,
        notify_callback=notify_frontend,
        debounce_seconds=1.5,
    )
    watcher.start()

    try:
        eel.start('main.html', size=(1200, 800))
    except (SystemExit, MemoryError, KeyboardInterrupt):
        pass
    finally:
        watcher.stop()
        cam_mgr.close()
        if detector:
            detector.close()
        print("👋 Moodify closed")