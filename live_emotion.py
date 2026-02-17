#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════╗
║   EMOTION DETECTOR v7 — ANGRY/SAD ACCURACY FIX               ║
║                                                                ║
║   Problem: Model under-predicts angry & sad (FER2013 bias)    ║
║                                                                ║
║   3 Targeted Fixes:                                            ║
║   ✔ Test-Time Augmentation (original + flip averaged)         ║
║   ✔ CLAHE contrast boost (makes subtle muscles visible)       ║
║   ✔ Geometric boosting (brow furrow → angry, frown → sad)    ║
║                                                                ║
║   Controls:                                                    ║
║   q — Quit         s — Screenshot    c — Calibrate neutral    ║
║   d — Debug toggle  g — Geo boost    r — Reset buffers        ║
║   t — TTA toggle    e — CLAHE toggle  v — Verbose             ║
║   1 — Raw CNN       2 — Smoothed      3 — Full (TTA+CLAHE)   ║
║   +/- — Smoothing   p — Freeze/analyze frame                  ║
║                                                                ║
║   Model: 48×48 grayscale, 5-class                             ║
║   Classes: angry, happy, neutral, sad, surprise               ║
╚════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import sys
import time
import math
import logging
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('EmotionV7')

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tf_keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        print("ERROR: pip install tf-keras")
        sys.exit(1)

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    log.info("MediaPipe not available — geometric boosting disabled")


# ════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
IDX_ANGRY    = 0
IDX_HAPPY    = 1
IDX_NEUTRAL  = 2
IDX_SAD      = 3
IDX_SURPRISE = 4
NUM_CLASSES  = 5

EMOTION_COLORS = {
    'angry':    (0,   0,   255),
    'happy':    (0,   220, 0),
    'neutral':  (200, 200, 0),
    'sad':      (255, 100, 0),
    'surprise': (0,   200, 220),
}

EMOTION_DISPLAY = {
    'angry':    'ANGRY  >:(',
    'happy':    'HAPPY  :D',
    'neutral':  'NEUTRAL :|',
    'sad':      'SAD    :(',
    'surprise': 'SURPRISE :O',
}


class PipelineMode(Enum):
    RAW_CNN  = "raw"
    SMOOTHED = "smoothed"
    FULL     = "full"


# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
@dataclass
class Config:
    # Camera
    camera_index:   int   = 0
    frame_width:    int   = 720
    frame_height:   int   = 540
    mirror:         bool  = True

    # Paths
    model_path:     str   = 'keras/Emotion_Detection.h5'
    cascade_path:   str   = 'keras/haarcascade_frontalface_default.xml'

    # ══════════════════════════════════════════════════
    # FIX 1: Test-Time Augmentation (TTA)
    # Average prediction of original + horizontally flipped face
    # Proven +3-5% accuracy on subtle emotions
    # ══════════════════════════════════════════════════
    tta_enabled:    bool  = True
    tta_weight_original: float = 0.55   # slightly favor original
    tta_weight_flipped:  float = 0.45

    # ══════════════════════════════════════════════════
    # FIX 2: CLAHE contrast enhancement
    # Makes subtle muscle movements (angry squint, sad droop)
    # visible to the CNN. Applied LIGHTLY so it helps not hurts.
    # ══════════════════════════════════════════════════
    clahe_enabled:  bool  = True
    clahe_clip:     float = 1.5     # LIGHT — not 2.5+
    clahe_grid:     int   = 4       # smaller grid = subtler effect

    # ══════════════════════════════════════════════════
    # FIX 3: Per-class calibration for FER2013 bias
    # The model systematically under-predicts angry & sad
    # These multipliers compensate (applied before argmax)
    # ══════════════════════════════════════════════════
    class_boost_enabled: bool = True
    class_boost: dict = field(default_factory=lambda: {
        'angry':    1.50,    # INCREASED from 1.35
        'happy':    1.00,    # no change
        'neutral':  0.80,    # DECREASED from 0.88 to reduce false neutral
        'sad':      1.55,    # INCREASED from 1.40
        'surprise': 1.10,    # ADDED slight boost for surprise
    })

    # ══════════════════════════════════════════════════
    # FIX 4: Geometric boosting for angry/sad
    # When MediaPipe landmarks show clear anger/sadness signals,
    # boost those CNN scores. Only activates on strong signals.
    # ══════════════════════════════════════════════════
    geo_boost_enabled:  bool  = True
    geo_boost_strength: float = 0.20  # max score boost from geometry
    geo_angry_min_confidence: float = 0.50  # min geo signal to activate
    geo_sad_min_confidence:   float = 0.50

    # Multi-crop ensemble
    multi_crop_enabled: bool = True   # try 3 different crop sizes

    # Haar cascade
    haar_scale:     float = 1.15
    haar_neighbors: int   = 5
    haar_min_size:  int   = 60

    # Smoothing
    ema_alpha:      float = 0.50
    vote_window:    int   = 5
    min_confidence: float = 0.25

    # Face persistence
    face_persist:   int   = 3

    # Auto-calibrate geometry baseline
    auto_calibrate:     bool  = True
    auto_cal_frames:    int   = 20
    auto_cal_neutral:   float = 0.45

    # Display
    show_debug:     bool  = True
    show_raw:       bool  = True
    verbose:        bool  = True
    verbose_interval: int = 15
    screenshot_dir: str   = './screenshots'


CFG = Config()


# ════════════════════════════════════════════════════════
# MEDIAPIPE LANDMARKS
# ════════════════════════════════════════════════════════
class LM:
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_INNER  = 133;  LEFT_EYE_OUTER  = 33
    RIGHT_EYE_INNER = 362;  RIGHT_EYE_OUTER = 263
    LEFT_EYEBROW_TOP    = 105; RIGHT_EYEBROW_TOP    = 334
    LEFT_EYEBROW_INNER  = 107; RIGHT_EYEBROW_INNER  = 336
    LEFT_EYEBROW  = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_LEFT   = 61;  MOUTH_RIGHT  = 291
    MOUTH_TOP    = 13;  MOUTH_BOTTOM = 14
    INNER_TOP    = 82;  INNER_BOTTOM = 87
    NOSE_TIP     = 1;   CHIN         = 152
    JAW_LEFT     = 234; JAW_RIGHT    = 454
    FOREHEAD_TOP = 10
    DISPLAY_LANDMARKS = list(set(
        LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW +
        [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM, NOSE_TIP]
    ))


# ════════════════════════════════════════════════════════
# GEOMETRIC ANALYSIS — Focused on angry/sad detection
# ════════════════════════════════════════════════════════
@dataclass
class GeoFeatures:
    # Raw measurements
    avg_ear: float = 0.0
    mar: float = 0.0
    mouth_width_ratio: float = 0.0
    smile_curvature: float = 0.0
    avg_brow_height: float = 0.0
    brow_furrow: float = 0.0
    face_width: float = 0.0
    quality: float = 0.0
    is_valid: bool = False

    # Angry indicators
    angry_brow_furrow: float = 0.0      # brows pulled together
    angry_eye_squint: float = 0.0       # eyes narrowed
    angry_lip_press: float = 0.0        # lips pressed tight
    angry_score: float = 0.0            # combined angry signal

    # Sad indicators
    sad_inner_brow_raise: float = 0.0   # inner brows up
    sad_lip_corner_down: float = 0.0    # mouth corners droop
    sad_mouth_narrow: float = 0.0       # mouth gets narrower
    sad_score: float = 0.0              # combined sad signal

    # Other
    is_duchenne: bool = False


@dataclass
class Baseline:
    avg_ear: float = 0.25
    mar: float = 0.08
    mouth_width_ratio: float = 0.40
    avg_brow_height: float = 0.20
    brow_furrow: float = 0.09


class GeoAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.baseline = Baseline()
        self.cal_buffer = []
        self.calibrated = False

    @staticmethod
    def _d(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    @staticmethod
    def _mid(a, b):
        return ((a[0]+b[0])/2, (a[1]+b[1])/2)

    def _ear(self, pts):
        if len(pts) < 6:
            return 0.0
        return ((self._d(pts[1], pts[5]) + self._d(pts[2], pts[4])) /
                (2 * max(self._d(pts[0], pts[3]), 1e-6)))

    def extract(self, lm_px, face_lms=None) -> GeoFeatures:
        f = GeoFeatures()

        def p(i):
            return (float(lm_px[i][0]), float(lm_px[i][1]))

        # Quality check
        if face_lms:
            good = 0
            for idx in [1, 33, 133, 362, 263, 61, 291, 13, 14, 105, 334, 152]:
                lk = face_lms.landmark[idx]
                if 0.0 <= lk.x <= 1.0 and 0.0 <= lk.y <= 1.0:
                    good += 1
            f.quality = good / 12.0
            if f.quality < 0.5:
                return f

        f.is_valid = True
        b = self.baseline

        # ── EAR (Eye Aspect Ratio) ──────────────────
        left_ear  = self._ear([p(i) for i in LM.LEFT_EYE])
        right_ear = self._ear([p(i) for i in LM.RIGHT_EYE])
        f.avg_ear = (left_ear + right_ear) / 2.0

        # ── Mouth ────────────────────────────────────
        mt, mb = p(LM.MOUTH_TOP), p(LM.MOUTH_BOTTOM)
        ml, mr = p(LM.MOUTH_LEFT), p(LM.MOUTH_RIGHT)
        mw = self._d(ml, mr)
        mh = self._d(mt, mb)
        f.mar = mh / max(mw, 1e-6)

        # ── Face width ───────────────────────────────
        f.face_width = self._d(p(LM.JAW_LEFT), p(LM.JAW_RIGHT))
        if f.face_width > 1e-6:
            f.mouth_width_ratio = mw / f.face_width

        # ── Smile curvature ──────────────────────────
        mouth_center = self._mid(mt, mb)
        corner_avg_y = (ml[1] + mr[1]) / 2.0
        f.smile_curvature = (mouth_center[1] - corner_avg_y) / max(mw, 1e-6)

        # ── Brows ────────────────────────────────────
        lbt = p(LM.LEFT_EYEBROW_TOP)
        rbt = p(LM.RIGHT_EYEBROW_TOP)
        lec = self._mid(p(LM.LEFT_EYE_INNER), p(LM.LEFT_EYE_OUTER))
        rec = self._mid(p(LM.RIGHT_EYE_INNER), p(LM.RIGHT_EYE_OUTER))
        if f.face_width > 1e-6:
            lbh = self._d(lbt, lec) / f.face_width
            rbh = self._d(rbt, rec) / f.face_width
            f.avg_brow_height = (lbh + rbh) / 2.0
            f.brow_furrow = self._d(
                p(LM.LEFT_EYEBROW_INNER),
                p(LM.RIGHT_EYEBROW_INNER)) / f.face_width

        # ══════════════════════════════════════════════
        # ANGRY DETECTION FEATURES
        # Key signals: brow furrow + eye squint + lip press
        # ══════════════════════════════════════════════

        # Brow furrow: inner brows pulled TOGETHER
        # When angry: brow_furrow DECREASES (brows move inward)
        if b.brow_furrow > 0.01:
            f.angry_brow_furrow = float(np.clip(
                (b.brow_furrow - f.brow_furrow) / (b.brow_furrow * 0.3),
                0, 1))

        # Eye squint: EAR DECREASES when eyes narrow
        if b.avg_ear > 0.01:
            f.angry_eye_squint = float(np.clip(
                (b.avg_ear - f.avg_ear) / (b.avg_ear * 0.25),
                0, 1))

        # Lip press: MAR DECREASES when lips press together
        if b.mar > 0.01:
            f.angry_lip_press = float(np.clip(
                (b.mar - f.mar) / (b.mar * 0.4),
                0, 1))

        # Combined angry (need at least brow OR squint + press)
        f.angry_score = (0.50 * f.angry_brow_furrow +
                         0.25 * f.angry_eye_squint +
                         0.25 * f.angry_lip_press)

        # ══════════════════════════════════════════════
        # SAD DETECTION FEATURES
        # Key signals: inner brow raise + lip corners down + narrow mouth
        # ══════════════════════════════════════════════

        # Inner brow raise: brow height INCREASES
        if b.avg_brow_height > 0.01:
            f.sad_inner_brow_raise = float(np.clip(
                (f.avg_brow_height - b.avg_brow_height) / (b.avg_brow_height * 0.25),
                0, 1))

        # Lip corners down: smile_curvature becomes NEGATIVE
        f.sad_lip_corner_down = float(np.clip(
            -f.smile_curvature / 0.10, 0, 1))

        # Mouth narrows when sad
        if b.mouth_width_ratio > 0.01:
            f.sad_mouth_narrow = float(np.clip(
                (b.mouth_width_ratio - f.mouth_width_ratio) / (b.mouth_width_ratio * 0.15),
                0, 1))

        # Combined sad
        f.sad_score = (0.40 * f.sad_inner_brow_raise +
                       0.40 * f.sad_lip_corner_down +
                       0.20 * f.sad_mouth_narrow)

        # Duchenne smile
        cheek_raise = float(np.clip((b.avg_ear - f.avg_ear) / 0.08, 0, 1))
        lip_pull = float(np.clip(
            (f.mouth_width_ratio - b.mouth_width_ratio) / 0.10, 0, 1))
        f.is_duchenne = cheek_raise > 0.2 and lip_pull > 0.3

        return f

    def calibrate(self, f):
        self.baseline = Baseline(
            avg_ear=f.avg_ear, mar=f.mar,
            mouth_width_ratio=f.mouth_width_ratio,
            avg_brow_height=f.avg_brow_height,
            brow_furrow=f.brow_furrow,
        )
        self.calibrated = True
        log.info(f"Calibrated: EAR={f.avg_ear:.3f} MAR={f.mar:.3f} "
                 f"BF={f.brow_furrow:.3f} BH={f.avg_brow_height:.3f}")

    def try_auto_cal(self, f, cnn_neutral):
        if self.calibrated or not self.cfg.auto_calibrate:
            return
        if not f.is_valid or cnn_neutral < self.cfg.auto_cal_neutral:
            return
        self.cal_buffer.append(f)
        if len(self.cal_buffer) >= self.cfg.auto_cal_frames:
            self.baseline = Baseline(
                avg_ear=float(np.mean([g.avg_ear for g in self.cal_buffer])),
                mar=float(np.mean([g.mar for g in self.cal_buffer])),
                mouth_width_ratio=float(np.mean([g.mouth_width_ratio for g in self.cal_buffer])),
                avg_brow_height=float(np.mean([g.avg_brow_height for g in self.cal_buffer])),
                brow_furrow=float(np.mean([g.brow_furrow for g in self.cal_buffer])),
            )
            self.calibrated = True
            self.cal_buffer.clear()
            log.info(f"Auto-calibrated: EAR={self.baseline.avg_ear:.3f} "
                     f"MAR={self.baseline.mar:.3f} "
                     f"BF={self.baseline.brow_furrow:.3f}")


# ════════════════════════════════════════════════════════
# TEMPORAL SMOOTHER
# ════════════════════════════════════════════════════════
class Smoother:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ema = np.ones(NUM_CLASSES) / NUM_CLASSES
        self.init = False
        self.votes = deque(maxlen=cfg.vote_window)
        self.current = 'neutral'
        self.count = 0

    def update(self, scores, mode):
        self.count += 1

        if mode == PipelineMode.RAW_CNN:
            idx = int(np.argmax(scores))
            self.current = CLASS_LABELS[idx]
            return {
                'emotion': self.current,
                'confidence': float(scores[idx]),
                'scores': {CLASS_LABELS[i]: round(float(scores[i]) * 100, 1)
                           for i in range(NUM_CLASSES)},
                'margin': float(np.sort(scores)[::-1][0] - np.sort(scores)[::-1][1])
                          if NUM_CLASSES > 1 else 0,
            }

        a = self.cfg.ema_alpha
        if not self.init:
            self.ema = 0.5 * (np.ones(NUM_CLASSES) / NUM_CLASSES) + 0.5 * scores
            self.init = True
        else:
            self.ema = a * scores + (1 - a) * self.ema

        t = self.ema.sum()
        if t > 1e-6:
            self.ema /= t

        idx = int(np.argmax(self.ema))
        cand = CLASS_LABELS[idx]
        conf = float(self.ema[idx])

        self.votes.append(cand)
        if len(self.votes) >= 3:
            vc = Counter(self.votes)
            top, cnt = vc.most_common(1)[0]
            if cnt / len(self.votes) >= 0.5:
                cand = top

        sa = np.sort(self.ema)[::-1]
        margin = float(sa[0] - sa[1]) if len(sa) > 1 else 0
        cal_conf = 0.6 * conf + 0.4 * margin

        if cal_conf >= self.cfg.min_confidence or self.count <= 5:
            if cand != self.current:
                log.info(f"EMOTION: {self.current} → {cand} ({cal_conf:.0%})")
                self.current = cand

        return {
            'emotion': self.current,
            'confidence': cal_conf,
            'scores': {CLASS_LABELS[i]: round(float(self.ema[i]) * 100, 1)
                       for i in range(NUM_CLASSES)},
            'margin': margin,
        }

    def reset(self):
        self.ema = np.ones(NUM_CLASSES) / NUM_CLASSES
        self.init = False
        self.votes.clear()
        self.current = 'neutral'
        self.count = 0


# ════════════════════════════════════════════════════════
# MAIN DETECTOR
# ════════════════════════════════════════════════════════
class EmotionDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self._banner()

        # ── Load model ───────────────────────────────
        self.model = load_model(self.cfg.model_path, compile=False)
        inp = self.model.input_shape
        out = self.model.output_shape
        log.info(f"Model: input={inp} output={out}")
        assert inp == (None, 48, 48, 1), f"Unexpected input: {inp}"
        assert out == (None, 5), f"Unexpected output: {out}"

        # Fast inference
        try:
            test = np.zeros((1, 48, 48, 1), dtype=np.float32)
            _ = self.model(test, training=False)
            self._infer = lambda x: self.model(x, training=False).numpy()
            log.info("Using fast model.__call__()")
        except Exception:
            self._infer = lambda x: self.model.predict(x, verbose=0)
            log.info("Using model.predict()")

        # Self-test
        test_out = self._infer(
            np.random.rand(1, 48, 48, 1).astype(np.float32))[0]
        log.info(f"Self-test: sum={test_out.sum():.3f}")
        self.needs_softmax = abs(test_out.sum() - 1.0) > 0.1

        # ── Haar cascade ─────────────────────────────
        self.cascade = cv2.CascadeClassifier(self.cfg.cascade_path)
        assert not self.cascade.empty(), f"Cannot load {self.cfg.cascade_path}"

        # ── CLAHE ────────────────────────────────────
        self.clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_grid, self.cfg.clahe_grid))

        # ── MediaPipe ────────────────────────────────
        self.face_mesh = None
        if MEDIAPIPE_OK:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

        # ── Engines ──────────────────────────────────
        self.smoother = Smoother(self.cfg)
        self.geo = GeoAnalyzer(self.cfg) if MEDIAPIPE_OK else None
        self.mode = PipelineMode(self.cfg.pipeline_mode if hasattr(self.cfg, 'pipeline_mode') else 'full')

        # State
        self.no_face = 0
        self.last_result = None
        self.frozen = False
        self.frozen_frame = None
        self.total_frames = 0

        log.info(f"TTA:   {'ON' if self.cfg.tta_enabled else 'OFF'}")
        log.info(f"CLAHE: {'ON' if self.cfg.clahe_enabled else 'OFF'} "
                 f"(clip={self.cfg.clahe_clip})")
        log.info(f"Class boost: {'ON' if self.cfg.class_boost_enabled else 'OFF'}")
        log.info(f"Geo boost:   {'ON' if self.cfg.geo_boost_enabled else 'OFF'}")
        log.info(f"Multi-crop:  {'ON' if self.cfg.multi_crop_enabled else 'OFF'}")
        print("=" * 55)

    def _banner(self):
        print("=" * 55)
        print("  EMOTION DETECTOR v7 — ANGRY/SAD FIX")
        print("  Fixes: TTA + CLAHE + Geometric Boosting")
        print("=" * 55)

    # ══════════════════════════════════════════════════
    # PREPROCESSING — The key to angry/sad accuracy
    # ══════════════════════════════════════════════════

    def _preprocess_roi(self, gray_roi, apply_clahe=False):
        """Resize + optional CLAHE + normalize → tensor."""
        if gray_roi.size == 0:
            return np.zeros((1, 48, 48, 1), dtype='float32')
            
        if apply_clahe and self.cfg.clahe_enabled:
            gray_roi = self.clahe.apply(gray_roi)
        
        # Using INTER_AREA for downsizing (better for 48x48)
        resized = cv2.resize(gray_roi, (48, 48),
                              interpolation=cv2.INTER_AREA)
        normalized = resized.astype('float32') / 255.0
        return normalized.reshape(1, 48, 48, 1)

    def _predict_single(self, tensor):
        """Raw CNN prediction from single tensor."""
        raw = self._infer(tensor)[0]
        if self.needs_softmax:
            e = np.exp(raw - np.max(raw))
            raw = e / e.sum()
        return raw

    # ══════════════════════════════════════════════════
    # FIX 1: TEST-TIME AUGMENTATION
    # ══════════════════════════════════════════════════

    def _predict_with_tta(self, gray_roi):
        """
        Average predictions from:
        1. Original face
        2. Horizontally flipped face
        This helps because facial expressions aren't perfectly symmetric,
        and the model sees slightly different features in each orientation.
        """
        # Original
        t1 = self._preprocess_roi(gray_roi, apply_clahe=True)
        p1 = self._predict_single(t1)

        if not self.cfg.tta_enabled:
            return p1, p1, None

        # Flipped
        flipped = cv2.flip(gray_roi, 1)
        t2 = self._preprocess_roi(flipped, apply_clahe=True)
        p2 = self._predict_single(t2)

        # Weighted average
        wo = self.cfg.tta_weight_original
        wf = self.cfg.tta_weight_flipped
        combined = wo * p1 + wf * p2
        total = combined.sum()
        if total > 1e-6:
            combined /= total

        return combined, p1, p2

    # ══════════════════════════════════════════════════
    # FIX 2: MULTI-CROP ENSEMBLE
    # ══════════════════════════════════════════════════

    def _predict_multi_crop(self, gray, x, y, w, h):
        """
        Try 3 crop sizes and average:
        1. Original Haar bbox
        2. Slightly tighter (crop out some background)
        3. Slightly larger (include more context)
        """
        gh, gw = gray.shape[:2]
        crops = []

        # Crop 1: Original
        roi1 = gray[y:y+h, x:x+w]
        if roi1.size > 0:
            crops.append(roi1)

        if not self.cfg.multi_crop_enabled:
            if crops:
                return self._predict_with_tta(crops[0])
            return np.ones(NUM_CLASSES) / NUM_CLASSES, None, None

        # Crop 2: Tighter (10% inward)
        pad = int(0.10 * min(w, h))
        x2, y2 = x + pad, y + pad
        w2, h2 = w - 2 * pad, h - 2 * pad
        if w2 > 20 and h2 > 20:
            roi2 = gray[y2:y2+h2, x2:x2+w2]
            if roi2.size > 0:
                crops.append(roi2)

        # Crop 3: Larger (15% outward)
        pad = int(0.15 * min(w, h))
        x3 = max(0, x - pad)
        y3 = max(0, y - pad)
        w3 = min(gw - x3, w + 2 * pad)
        h3 = min(gh - y3, h + 2 * pad)
        if w3 > 20 and h3 > 20:
            roi3 = gray[y3:y3+h3, x3:x3+w3]
            if roi3.size > 0:
                crops.append(roi3)

        if not crops:
            return np.ones(NUM_CLASSES) / NUM_CLASSES, None, None

        # Predict each crop (with TTA) and average
        all_preds = []
        first_raw = None
        first_flip = None

        for i, roi in enumerate(crops):
            combined, raw, flip = self._predict_with_tta(roi)
            all_preds.append(combined)
            if i == 0:
                first_raw = raw
                first_flip = flip

        # Average all crop predictions
        ensemble = np.mean(all_preds, axis=0)
        total = ensemble.sum()
        if total > 1e-6:
            ensemble /= total

        return ensemble, first_raw, first_flip

    # ══════════════════════════════════════════════════
    # FIX 3: PER-CLASS CALIBRATION
    # ══════════════════════════════════════════════════

    def _apply_class_boost(self, scores):
        """Apply per-class multipliers to compensate for model bias."""
        if not self.cfg.class_boost_enabled:
            return scores.copy()

        boosted = scores.copy()
        for i, label in enumerate(CLASS_LABELS):
            boosted[i] *= self.cfg.class_boost.get(label, 1.0)

        total = boosted.sum()
        return boosted / total if total > 1e-6 else scores.copy()

    # ══════════════════════════════════════════════════
    # FIX 4: GEOMETRIC BOOSTING FOR ANGRY/SAD
    # ══════════════════════════════════════════════════

    def _apply_geo_boost(self, scores, geo):
        """
        When geometry STRONGLY indicates angry or sad,
        boost those CNN scores. This only activates when
        geometric signals are clear (above threshold).
        """
        if (not self.cfg.geo_boost_enabled or
                geo is None or not geo.is_valid or
                not self.geo or not self.geo.calibrated):
            return scores.copy()

        boosted = scores.copy()
        strength = self.cfg.geo_boost_strength

        # Angry boost: furrowed brows + squinting + pressed lips
        if geo.angry_score > self.cfg.geo_angry_min_confidence:
            # Scale boost by how strong the geometric signal is
            boost = strength * (geo.angry_score - self.cfg.geo_angry_min_confidence) / (1.0 - self.cfg.geo_angry_min_confidence)
            boosted[IDX_ANGRY] += boost
            # Reduce neutral proportionally
            boosted[IDX_NEUTRAL] = max(0, boosted[IDX_NEUTRAL] - boost * 0.5)

        # Sad boost: inner brow raise + frown + narrow mouth
        if geo.sad_score > self.cfg.geo_sad_min_confidence:
            boost = strength * (geo.sad_score - self.cfg.geo_sad_min_confidence) / (1.0 - self.cfg.geo_sad_min_confidence)
            boosted[IDX_SAD] += boost
            boosted[IDX_NEUTRAL] = max(0, boosted[IDX_NEUTRAL] - boost * 0.5)

        # Re-normalize
        total = boosted.sum()
        return boosted / total if total > 1e-6 else scores.copy()

    # ══════════════════════════════════════════════════
    # MAIN PROCESS
    # ══════════════════════════════════════════════════

    def process(self, frame):
        result = {
            'emotion': 'neutral', 'confidence': 0.0, 'scores': {},
            'bbox': None, 'geo': None, 'lm_px': None,
            'faces_count': 0, 'cnn_raw': None, 'cnn_flip': None,
            'boosted': None, 'margin': 0.0,
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ── Face detection (Haar) ────────────────────
        faces = self.cascade.detectMultiScale(
            gray, self.cfg.haar_scale, self.cfg.haar_neighbors,
            minSize=(self.cfg.haar_min_size, self.cfg.haar_min_size))

        result['faces_count'] = len(faces)

        if len(faces) == 0:
            self.no_face += 1
            if self.last_result and self.no_face <= self.cfg.face_persist:
                result['emotion'] = self.last_result['emotion']
                result['confidence'] = self.last_result['confidence'] * 0.7
                result['scores'] = self.last_result.get('scores', {})
            return result

        self.no_face = 0

        # Largest face
        areas = [fw * fh for _, _, fw, fh in faces]
        bi = int(np.argmax(areas))
        fx, fy, fw, fh = faces[bi]
        result['bbox'] = (fx, fy, fw, fh)

        # ── CNN prediction (with TTA + multi-crop) ───
        ensemble, cnn_raw, cnn_flip = self._predict_multi_crop(
            gray, fx, fy, fw, fh)

        result['cnn_raw'] = cnn_raw
        result['cnn_flip'] = cnn_flip

        # ── Geometric features ───────────────────────
        geo = None
        if MEDIAPIPE_OK and self.face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_out = self.face_mesh.process(rgb)
            if mp_out.multi_face_landmarks:
                fl = mp_out.multi_face_landmarks[0]
                lm_px = np.array([(l.x * w, l.y * h) for l in fl.landmark])
                result['lm_px'] = lm_px
                geo = self.geo.extract(lm_px, fl)
                result['geo'] = geo

                # Auto-calibrate
                if geo.is_valid:
                    self.geo.try_auto_cal(geo, float(ensemble[IDX_NEUTRAL]))

        # ── Apply fixes ──────────────────────────────
        # Step 1: Per-class calibration
        calibrated = self._apply_class_boost(ensemble)

        # Step 2: Geometric boosting for angry/sad
        boosted = self._apply_geo_boost(calibrated, geo)

        result['boosted'] = boosted.copy()

        # ── Temporal smoothing ───────────────────────
        temporal = self.smoother.update(boosted, self.mode)
        result.update(temporal)

        self.last_result = result
        return result

    # ══════════════════════════════════════════════════
    # VISUALIZATION
    # ══════════════════════════════════════════════════

    def _draw_box(self, fr, bbox, emotion, conf):
        x, y, w, h = bbox
        col = EMOTION_COLORS.get(emotion, (180, 180, 180))
        cl = int(min(w, h) * 0.2)
        for cx, cy, dx, dy, ex, ey in [
            (x, y, cl, 0, 0, cl), (x+w, y, -cl, 0, 0, cl),
            (x, y+h, cl, 0, 0, -cl), (x+w, y+h, -cl, 0, 0, -cl)]:
            cv2.line(fr, (cx, cy), (cx+dx, cy+dy), col, 3)
            cv2.line(fr, (cx, cy), (cx+ex, cy+ey), col, 3)
        label = EMOTION_DISPLAY.get(emotion, emotion.upper())
        cv2.putText(fr, label, (x, y-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(fr, f"{conf*100:.0f}%", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    def _draw_bars(self, fr, scores, x, y):
        for i, l in enumerate(CLASS_LABELS):
            by = y + i * 17
            v = scores.get(l, 0) / 100.0
            col = EMOTION_COLORS.get(l, (180, 180, 180))
            cv2.putText(fr, l[:3].upper(), (x, by+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
            bx = x + 35
            cv2.rectangle(fr, (bx, by), (bx+110, by+12), (40, 40, 40), -1)
            fill = int(110 * v)
            if fill > 0:
                cv2.rectangle(fr, (bx, by), (bx+fill, by+12), col, -1)
            cv2.putText(fr, f"{v*100:.0f}%", (bx+113, by+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 150), 1)

    def _draw_raw_debug(self, fr, r):
        """Show CNN raw + boosted scores side by side."""
        fh, fw = fr.shape[:2]
        x = fw - 330
        y = 10

        cnn_raw = r.get('cnn_raw')
        cnn_flip = r.get('cnn_flip')
        boosted = r.get('boosted')

        # Background
        ov = fr.copy()
        cv2.rectangle(ov, (x-5, y-5), (x+320, y+130), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, fr, 0.35, 0, fr)

        # Headers
        cv2.putText(fr, "RAW CNN", (x, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        if cnn_flip is not None:
            cv2.putText(fr, "FLIPPED", (x+100, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 0), 1)
        cv2.putText(fr, "BOOSTED", (x+200, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 100), 1)

        for i, l in enumerate(CLASS_LABELS):
            ly = y + 25 + i * 18

            # Highlight angry/sad
            is_target = l in ('angry', 'sad')
            name_col = (100, 200, 255) if is_target else (140, 140, 140)
            cv2.putText(fr, f"{l[:3].upper()}", (x-35, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, name_col, 1)

            # Raw
            if cnn_raw is not None:
                v = float(cnn_raw[i])
                col = (0, 255, 200) if i == int(np.argmax(cnn_raw)) else (140, 140, 140)
                cv2.putText(fr, f"{v*100:5.1f}%", (x, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

            # Flipped
            if cnn_flip is not None:
                v = float(cnn_flip[i])
                col = (255, 200, 0) if i == int(np.argmax(cnn_flip)) else (100, 100, 100)
                cv2.putText(fr, f"{v*100:5.1f}%", (x+100, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

            # Boosted
            if boosted is not None:
                v = float(boosted[i])
                col = (0, 255, 100) if i == int(np.argmax(boosted)) else (120, 120, 120)
                cv2.putText(fr, f"{v*100:5.1f}%", (x+200, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

    def _draw_geo_angry_sad(self, fr, geo):
        """Show angry/sad geometric indicators."""
        if not geo or not geo.is_valid:
            return

        fh, fw = fr.shape[:2]
        x, y = fw - 220, 155

        ov = fr.copy()
        cv2.rectangle(ov, (x-5, y-12), (x+210, y+100), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, fr, 0.4, 0, fr)

        lines = [
            ("─── ANGRY SIGNALS ───", (0, 100, 255)),
            (f"  Brow furrow:  {geo.angry_brow_furrow:.2f} {'█' * int(geo.angry_brow_furrow * 10)}", None),
            (f"  Eye squint:   {geo.angry_eye_squint:.2f} {'█' * int(geo.angry_eye_squint * 10)}", None),
            (f"  Lip press:    {geo.angry_lip_press:.2f} {'█' * int(geo.angry_lip_press * 10)}", None),
            (f"  ANGRY TOTAL:  {geo.angry_score:.2f}", (0, 100, 255)),
            ("─── SAD SIGNALS ─────", (255, 100, 0)),
            (f"  Inner brow ↑: {geo.sad_inner_brow_raise:.2f} {'█' * int(geo.sad_inner_brow_raise * 10)}", None),
            (f"  Lip corner ↓: {geo.sad_lip_corner_down:.2f} {'█' * int(geo.sad_lip_corner_down * 10)}", None),
            (f"  Mouth narrow: {geo.sad_mouth_narrow:.2f} {'█' * int(geo.sad_mouth_narrow * 10)}", None),
            (f"  SAD TOTAL:    {geo.sad_score:.2f}", (255, 100, 0)),
        ]

        if self.geo and self.geo.calibrated:
            lines.insert(0, ("CALIBRATED ✓", (0, 255, 0)))
        else:
            lines.insert(0, ("NOT CALIBRATED (press c)", (0, 0, 255)))

        for i, (txt, col) in enumerate(lines):
            c = col or (160, 160, 160)
            cv2.putText(fr, txt, (x, y + i * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1)

    def _draw_info(self, fr, fps, r):
        fh, fw = fr.shape[:2]
        ov = fr.copy()
        cv2.rectangle(ov, (0, 0), (260, 140), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, fr, 0.45, 0, fr)

        cal = "CAL:✓" if (self.geo and self.geo.calibrated) else "CAL:—"
        lines = [
            (f"FPS: {fps:.1f}", (0, 255, 0), 0.50, 2),
            (f"Faces: {r['faces_count']}", (0, 200, 255), 0.40, 1),
            (f"Mode: {self.mode.value}  {cal}", (100, 200, 255), 0.40, 1),
            (f"Conf: {r['confidence']*100:.0f}%  Margin: {r.get('margin',0):.2f}",
             (160, 160, 160), 0.38, 1),
            (f"TTA:{'ON' if self.cfg.tta_enabled else 'off'} "
             f"CLAHE:{'ON' if self.cfg.clahe_enabled else 'off'} "
             f"Boost:{'ON' if self.cfg.class_boost_enabled else 'off'}",
             (120, 180, 120), 0.34, 1),
            (f"GeoBst:{'ON' if self.cfg.geo_boost_enabled else 'off'} "
             f"MCrop:{'ON' if self.cfg.multi_crop_enabled else 'off'}",
             (120, 180, 120), 0.34, 1),
        ]
        for i, (txt, col, sc, th) in enumerate(lines):
            cv2.putText(fr, txt, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)

    def _draw_landmarks(self, fr, lm_px):
        if lm_px is None:
            return
        for i in LM.DISPLAY_LANDMARKS:
            if i < len(lm_px):
                cv2.circle(fr, (int(lm_px[i][0]), int(lm_px[i][1])),
                           2, (0, 255, 0), -1)

    # ══════════════════════════════════════════════════
    # DIAGNOSTIC: Freeze frame and analyze
    # ══════════════════════════════════════════════════

    def _analyze_frozen(self, frame):
        """Deep analysis of a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, self.cfg.haar_scale, self.cfg.haar_neighbors,
            minSize=(self.cfg.haar_min_size, self.cfg.haar_min_size))

        if len(faces) == 0:
            log.info("No face in frozen frame")
            return

        areas = [w*h for _, _, w, h in faces]
        fx, fy, fw, fh = faces[int(np.argmax(areas))]
        roi = gray[fy:fy+fh, fx:fx+fw]

        log.info("=" * 50)
        log.info("FROZEN FRAME ANALYSIS")
        log.info("=" * 50)

        # Raw (no CLAHE, no TTA)
        t1 = roi.copy()
        t1 = cv2.resize(t1, (48, 48), interpolation=cv2.INTER_LINEAR)
        t1 = (t1.astype('float32') / 255.0).reshape(1, 48, 48, 1)
        raw = self._infer(t1)[0]
        log.info(f"RAW (no CLAHE): {' '.join(f'{CLASS_LABELS[i][:3]}={raw[i]:.3f}' for i in range(5))}")
        log.info(f"  → {CLASS_LABELS[np.argmax(raw)]} ({raw.max():.1%})")

        # With CLAHE
        t2 = self.clahe.apply(roi.copy())
        t2 = cv2.resize(t2, (48, 48), interpolation=cv2.INTER_LINEAR)
        t2 = (t2.astype('float32') / 255.0).reshape(1, 48, 48, 1)
        clahe_pred = self._infer(t2)[0]
        log.info(f"CLAHE:          {' '.join(f'{CLASS_LABELS[i][:3]}={clahe_pred[i]:.3f}' for i in range(5))}")
        log.info(f"  → {CLASS_LABELS[np.argmax(clahe_pred)]} ({clahe_pred.max():.1%})")

        # Flipped
        t3 = cv2.flip(roi.copy(), 1)
        t3 = self.clahe.apply(t3)
        t3 = cv2.resize(t3, (48, 48), interpolation=cv2.INTER_LINEAR)
        t3 = (t3.astype('float32') / 255.0).reshape(1, 48, 48, 1)
        flip_pred = self._infer(t3)[0]
        log.info(f"FLIPPED+CLAHE:  {' '.join(f'{CLASS_LABELS[i][:3]}={flip_pred[i]:.3f}' for i in range(5))}")

        # TTA combined
        combined = 0.55 * clahe_pred + 0.45 * flip_pred
        combined /= combined.sum()
        log.info(f"TTA COMBINED:   {' '.join(f'{CLASS_LABELS[i][:3]}={combined[i]:.3f}' for i in range(5))}")
        log.info(f"  → {CLASS_LABELS[np.argmax(combined)]} ({combined.max():.1%})")

        # Boosted
        boosted = combined.copy()
        for i, l in enumerate(CLASS_LABELS):
            boosted[i] *= self.cfg.class_boost.get(l, 1.0)
        boosted /= boosted.sum()
        log.info(f"CLASS BOOSTED:  {' '.join(f'{CLASS_LABELS[i][:3]}={boosted[i]:.3f}' for i in range(5))}")
        log.info(f"  → {CLASS_LABELS[np.argmax(boosted)]} ({boosted.max():.1%})")

        # Difference analysis
        log.info("-" * 50)
        diff = boosted - raw
        for i, l in enumerate(CLASS_LABELS):
            arrow = "↑" if diff[i] > 0.01 else "↓" if diff[i] < -0.01 else "="
            log.info(f"  {l:10s}: raw={raw[i]:.3f} → final={boosted[i]:.3f}  {arrow} {diff[i]:+.3f}")
        log.info("=" * 50)

    # ══════════════════════════════════════════════════
    # RUN LOOP
    # ══════════════════════════════════════════════════

    def run(self):
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        os.makedirs(self.cfg.screenshot_dir, exist_ok=True)

        prev_t = time.time()
        fc = 0
        fps = 0.0
        show_debug = self.cfg.show_debug
        show_raw = self.cfg.show_raw
        verbose = self.cfg.verbose

        log.info("Camera started — press 'h' for help")

        while True:
            if not self.frozen:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)
                self.frozen_frame = frame.copy()
            else:
                frame = self.frozen_frame.copy()

            clean = frame.copy()
            fh, fw = frame.shape[:2]
            self.total_frames += 1

            fc += 1
            now = time.time()
            if now - prev_t >= 1.0:
                fps = fc / (now - prev_t)
                fc = 0
                prev_t = now

            # ── Process ──────────────────────────────
            r = self.process(frame)

            emotion = r['emotion']
            conf    = r['confidence']
            scores  = r.get('scores', {})
            bbox    = r['bbox']
            geo     = r.get('geo')
            lm_px   = r.get('lm_px')

            # ── Verbose ──────────────────────────────
            if verbose and (self.total_frames <= 20 or
                            self.total_frames % self.cfg.verbose_interval == 0):
                raw = r.get('cnn_raw')
                boosted = r.get('boosted')
                parts = [f"F{self.total_frames}"]
                if raw is not None:
                    parts.append("RAW:[" + " ".join(
                        f"{CLASS_LABELS[i][:3]}={v:.2f}"
                        for i, v in enumerate(raw)) + "]")
                if boosted is not None:
                    parts.append("BST:[" + " ".join(
                        f"{CLASS_LABELS[i][:3]}={v:.2f}"
                        for i, v in enumerate(boosted)) + "]")
                parts.append(f"→ {emotion.upper()} ({conf:.0%})")
                if geo and geo.is_valid:
                    parts.append(f"ang={geo.angry_score:.2f} sad={geo.sad_score:.2f}")
                log.info(" | ".join(parts))

            # ── Draw ─────────────────────────────────
            if bbox:
                self._draw_box(frame, bbox, emotion, conf)
                bx = bbox[0] + bbox[2] + 15
                if bx + 170 < fw:
                    self._draw_bars(frame, scores, bx, bbox[1])
                elif bbox[0] > 180:
                    self._draw_bars(frame, scores, bbox[0] - 175, bbox[1])

            if show_debug and lm_px is not None:
                self._draw_landmarks(frame, lm_px)

            if show_raw:
                self._draw_raw_debug(frame, r)

            if show_debug and geo and geo.is_valid:
                self._draw_geo_angry_sad(frame, geo)

            self._draw_info(frame, fps, r)

            if self.frozen:
                cv2.putText(frame, "FROZEN (press 'p' to unfreeze)",
                            (fw//2-150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            if r['faces_count'] == 0 and self.no_face > self.cfg.face_persist:
                cv2.putText(frame, "No face detected",
                            (fw//2-100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(
                frame,
                "q:Quit s:Shot c:Cal d:Debug r:Reset "
                "t:TTA e:CLAHE g:GeoBst p:Freeze 1/2/3:Mode +/-:Smooth",
                (5, fh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (90, 90, 90), 1)

            cv2.imshow("Emotion Detector v7", frame)

            # ── Keys ─────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                path = os.path.join(
                    self.cfg.screenshot_dir,
                    f"emo_{emotion}_{int(time.time())}.png")
                cv2.imwrite(path, clean)
                log.info(f"Screenshot: {path}")

            elif key == ord('c'):
                if geo and geo.is_valid:
                    self.geo.calibrate(geo)
                else:
                    log.warning("No valid face — look at camera with neutral expression")

            elif key == ord('d'):
                show_debug = not show_debug

            elif key == ord('r'):
                self.smoother.reset()
                if self.geo:
                    self.geo.calibrated = False
                    self.geo.cal_buffer.clear()
                log.info("Reset")

            elif key == ord('t'):
                self.cfg.tta_enabled = not self.cfg.tta_enabled
                log.info(f"TTA: {'ON' if self.cfg.tta_enabled else 'OFF'}")

            elif key == ord('e'):
                self.cfg.clahe_enabled = not self.cfg.clahe_enabled
                log.info(f"CLAHE: {'ON' if self.cfg.clahe_enabled else 'OFF'}")

            elif key == ord('g'):
                self.cfg.geo_boost_enabled = not self.cfg.geo_boost_enabled
                log.info(f"Geo boost: {'ON' if self.cfg.geo_boost_enabled else 'OFF'}")

            elif key == ord('b'):
                self.cfg.class_boost_enabled = not self.cfg.class_boost_enabled
                log.info(f"Class boost: {'ON' if self.cfg.class_boost_enabled else 'OFF'}")

            elif key == ord('m'):
                self.cfg.multi_crop_enabled = not self.cfg.multi_crop_enabled
                log.info(f"Multi-crop: {'ON' if self.cfg.multi_crop_enabled else 'OFF'}")

            elif key == ord('v'):
                verbose = not verbose
                log.info(f"Verbose: {'ON' if verbose else 'OFF'}")

            elif key == ord('w'):
                show_raw = not show_raw

            elif key == ord('p'):
                self.frozen = not self.frozen
                if self.frozen:
                    log.info("Frame FROZEN — analyzing...")
                    self._analyze_frozen(self.frozen_frame)
                else:
                    log.info("Unfrozen")

            elif key == ord('1'):
                self.mode = PipelineMode.RAW_CNN
                self.smoother.reset()
                log.info("Mode: RAW CNN")

            elif key == ord('2'):
                self.mode = PipelineMode.SMOOTHED
                self.smoother.reset()
                log.info("Mode: SMOOTHED")

            elif key == ord('3'):
                self.mode = PipelineMode.FULL
                self.smoother.reset()
                log.info("Mode: FULL")

            elif key in (ord('+'), ord('=')):
                self.cfg.ema_alpha = min(0.95, self.cfg.ema_alpha + 0.05)
                log.info(f"Smoothing: {self.cfg.ema_alpha:.2f}")

            elif key in (ord('-'), ord('_')):
                self.cfg.ema_alpha = max(0.10, self.cfg.ema_alpha - 0.05)
                log.info(f"Smoothing: {self.cfg.ema_alpha:.2f}")

            elif key == ord('h'):
                print("""
╔═══════════════════════════════════════════╗
║  CONTROLS                                ║
╠═══════════════════════════════════════════╣
║  q — Quit            s — Screenshot      ║
║  c — Calibrate (IMPORTANT for angry/sad) ║
║  d — Toggle debug overlay                ║
║  w — Toggle raw scores display           ║
║  r — Reset all buffers                   ║
║  p — FREEZE frame + deep analysis        ║
║  v — Verbose logging                     ║
║                                          ║
║  ANGRY/SAD FIXES:                        ║
║  t — Toggle TTA (flip averaging)         ║
║  e — Toggle CLAHE (contrast boost)       ║
║  b — Toggle per-class boosting           ║
║  g — Toggle geometric boosting           ║
║  m — Toggle multi-crop ensemble          ║
║                                          ║
║  PIPELINE:                               ║
║  1 — RAW CNN (no processing)             ║
║  2 — SMOOTHED (light EMA)               ║
║  3 — FULL (TTA+CLAHE+boost+geo)         ║
║  +/- — Adjust smoothing                 ║
║                                          ║
║  TIP: Press 'c' while showing a NEUTRAL  ║
║  face to calibrate. This dramatically    ║
║  improves angry/sad detection.           ║
╚═══════════════════════════════════════════╝
                """)

        cap.release()
        cv2.destroyAllWindows()
        if self.face_mesh:
            self.face_mesh.close()
        log.info(f"Done. Frames: {self.total_frames}")


if __name__ == "__main__":
    detector = EmotionDetector(CFG)
    detector.run()