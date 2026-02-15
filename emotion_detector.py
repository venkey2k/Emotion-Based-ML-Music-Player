"""
═══════════════════════════════════════════════════════════════
  MOODIFY — Emotion Detector v7 (Module)

  4 Accuracy Fixes for angry/sad:
  ✔ Test-Time Augmentation (original + flip averaged)
  ✔ CLAHE contrast boost (subtle muscles visible)
  ✔ Per-class calibration (FER2013 bias correction)
  ✔ Geometric boosting (brow furrow → angry, frown → sad)

  Usage:
      from emotion_detector import EmotionDetector, DetectorConfig

      detector = EmotionDetector(
          model_path='../keras/Emotion_Detection.h5',
          cascade_path='../keras/haarcascade_frontalface_default.xml'
      )

      # Per-frame (live):
      result = detector.detect(frame)
      print(result['emotion'], result['confidence'])

      # Batch (one-shot):
      emotion, conf, scores = detector.detect_batch(frames)

      # Detailed debug:
      result = detector.detect(frame)
      print(result['raw_scores'])     # before boosting
      print(result['boosted_scores']) # after boosting
      print(result['geo'])            # geometric features

  Model: 48×48 grayscale, 5-class
  Classes: angry, happy, neutral, sad, surprise
═══════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import os
import math
import logging
import threading
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

log = logging.getLogger('EmotionDetector')

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ──────────────────────────────────────────────
# KERAS IMPORT
# ──────────────────────────────────────────────
_load_model = None


def _import_keras():
    global _load_model
    if _load_model is not None:
        return
    for attempt in [
        lambda: __import__('tf_keras.models', fromlist=['load_model']).load_model,
        lambda: __import__('tensorflow.keras.models', fromlist=['load_model']).load_model,
        lambda: __import__('keras.models', fromlist=['load_model']).load_model,
    ]:
        try:
            _load_model = attempt()
            return
        except ImportError:
            continue
    raise ImportError("Cannot import Keras. Install: pip install tf-keras")


# ──────────────────────────────────────────────
# MEDIAPIPE (optional)
# ──────────────────────────────────────────────
try:
    import mediapipe as mp_lib
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    log.info("MediaPipe not installed — geometric boosting disabled")


# ════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
IDX_ANGRY    = 0
IDX_HAPPY    = 1
IDX_NEUTRAL  = 2
IDX_SAD      = 3
IDX_SURPRISE = 4
NUM_CLASSES  = 5

EMOTION_COLORS = {
    'angry':    (0, 0, 255),
    'happy':    (0, 220, 0),
    'neutral':  (200, 200, 0),
    'sad':      (255, 100, 0),
    'surprise': (0, 200, 220),
}

EMOTION_DISPLAY = {
    'angry':    'ANGRY  >:(',
    'happy':    'HAPPY  :D',
    'neutral':  'NEUTRAL :|',
    'sad':      'SAD    :(',
    'surprise': 'SURPRISE :O',
}


# ════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════
@dataclass
class DetectorConfig:
    # TTA
    tta_enabled:         bool  = True
    tta_weight_original: float = 0.55
    tta_weight_flipped:  float = 0.45

    # CLAHE
    clahe_enabled:  bool  = True
    clahe_clip:     float = 1.5
    clahe_grid:     int   = 4

    # Per-class calibration
    class_boost_enabled: bool = True
    class_boost: dict = field(default_factory=lambda: {
        'angry': 1.50, 'happy': 1.00, 'neutral': 0.80,
        'sad': 1.55, 'surprise': 1.10,
    })

    # Geometric boosting
    geo_boost_enabled:         bool  = True
    geo_boost_strength:        float = 0.20
    geo_angry_min_confidence:  float = 0.50
    geo_sad_min_confidence:    float = 0.50

    # Multi-crop
    multi_crop_enabled: bool = True

    # Haar
    haar_scale:     float = 1.15
    haar_neighbors: int   = 5
    haar_min_size:  int   = 60

    # Smoothing
    ema_alpha:       float = 0.50
    vote_window:     int   = 5
    min_confidence:  float = 0.25

    # Face persistence
    face_persist_frames: int = 3

    # Auto-calibration
    auto_calibrate:       bool  = True
    auto_cal_frames:      int   = 20
    auto_cal_neutral_min: float = 0.45


# ════════════════════════════════════════════════
# LANDMARK INDICES
# ════════════════════════════════════════════════
class _LM:
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_INNER = 133; LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362; RIGHT_EYE_OUTER = 263
    LEFT_EYEBROW_TOP = 105; RIGHT_EYEBROW_TOP = 334
    LEFT_EYEBROW_INNER = 107; RIGHT_EYEBROW_INNER = 336
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_LEFT = 61; MOUTH_RIGHT = 291
    MOUTH_TOP = 13; MOUTH_BOTTOM = 14
    NOSE_TIP = 1; CHIN = 152
    JAW_LEFT = 234; JAW_RIGHT = 454
    QUALITY_LANDMARKS = [1, 33, 133, 362, 263, 61, 291, 13, 14, 105, 334, 152]
    DISPLAY_LANDMARKS = list(set(
        LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW +
        [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM, NOSE_TIP]
    ))


# ════════════════════════════════════════════════
# GEOMETRIC FEATURES
# ════════════════════════════════════════════════
@dataclass
class GeoFeatures:
    avg_ear: float = 0.0; mar: float = 0.0
    mouth_width_ratio: float = 0.0; smile_curvature: float = 0.0
    avg_brow_height: float = 0.0; brow_furrow: float = 0.0
    face_width: float = 0.0; quality: float = 0.0; is_valid: bool = False
    angry_brow_furrow: float = 0.0; angry_eye_squint: float = 0.0
    angry_lip_press: float = 0.0; angry_score: float = 0.0
    sad_inner_brow_raise: float = 0.0; sad_lip_corner_down: float = 0.0
    sad_mouth_narrow: float = 0.0; sad_score: float = 0.0


@dataclass
class _Baseline:
    avg_ear: float = 0.25; mar: float = 0.08
    mouth_width_ratio: float = 0.40; avg_brow_height: float = 0.20
    brow_furrow: float = 0.09


# ════════════════════════════════════════════════
# GEOMETRIC ANALYZER
# ════════════════════════════════════════════════
class _GeoAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.baseline = _Baseline()
        self._cal_buffer = []
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

    def extract(self, lm_px, face_lms=None):
        f = GeoFeatures()

        def p(i):
            return (float(lm_px[i][0]), float(lm_px[i][1]))

        if face_lms:
            good = sum(
                1 for idx in _LM.QUALITY_LANDMARKS
                if 0 <= face_lms.landmark[idx].x <= 1
                and 0 <= face_lms.landmark[idx].y <= 1
            )
            f.quality = good / len(_LM.QUALITY_LANDMARKS)
            if f.quality < 0.5:
                return f

        f.is_valid = True
        b = self.baseline

        left_ear = self._ear([p(i) for i in _LM.LEFT_EYE])
        right_ear = self._ear([p(i) for i in _LM.RIGHT_EYE])
        f.avg_ear = (left_ear + right_ear) / 2.0

        mt, mb = p(_LM.MOUTH_TOP), p(_LM.MOUTH_BOTTOM)
        ml, mr = p(_LM.MOUTH_LEFT), p(_LM.MOUTH_RIGHT)
        mw, mh = self._d(ml, mr), self._d(mt, mb)
        f.mar = mh / max(mw, 1e-6)

        f.face_width = self._d(p(_LM.JAW_LEFT), p(_LM.JAW_RIGHT))
        if f.face_width > 1e-6:
            f.mouth_width_ratio = mw / f.face_width

        mouth_center = self._mid(mt, mb)
        f.smile_curvature = (mouth_center[1] - (ml[1]+mr[1])/2) / max(mw, 1e-6)

        lbt, rbt = p(_LM.LEFT_EYEBROW_TOP), p(_LM.RIGHT_EYEBROW_TOP)
        lec = self._mid(p(_LM.LEFT_EYE_INNER), p(_LM.LEFT_EYE_OUTER))
        rec = self._mid(p(_LM.RIGHT_EYE_INNER), p(_LM.RIGHT_EYE_OUTER))
        if f.face_width > 1e-6:
            f.avg_brow_height = (self._d(lbt, lec) + self._d(rbt, rec)) / (2 * f.face_width)
            f.brow_furrow = self._d(
                p(_LM.LEFT_EYEBROW_INNER), p(_LM.RIGHT_EYEBROW_INNER)
            ) / f.face_width

        # Angry signals
        if b.brow_furrow > 0.01:
            f.angry_brow_furrow = float(np.clip(
                (b.brow_furrow - f.brow_furrow) / (b.brow_furrow * 0.3), 0, 1))
        if b.avg_ear > 0.01:
            f.angry_eye_squint = float(np.clip(
                (b.avg_ear - f.avg_ear) / (b.avg_ear * 0.25), 0, 1))
        if b.mar > 0.01:
            f.angry_lip_press = float(np.clip(
                (b.mar - f.mar) / (b.mar * 0.4), 0, 1))
        f.angry_score = 0.50*f.angry_brow_furrow + 0.25*f.angry_eye_squint + 0.25*f.angry_lip_press

        # Sad signals
        if b.avg_brow_height > 0.01:
            f.sad_inner_brow_raise = float(np.clip(
                (f.avg_brow_height - b.avg_brow_height) / (b.avg_brow_height * 0.25), 0, 1))
        f.sad_lip_corner_down = float(np.clip(-f.smile_curvature / 0.10, 0, 1))
        if b.mouth_width_ratio > 0.01:
            f.sad_mouth_narrow = float(np.clip(
                (b.mouth_width_ratio - f.mouth_width_ratio) / (b.mouth_width_ratio * 0.15), 0, 1))
        f.sad_score = 0.40*f.sad_inner_brow_raise + 0.40*f.sad_lip_corner_down + 0.20*f.sad_mouth_narrow

        return f

    def calibrate(self, f):
        self.baseline = _Baseline(
            avg_ear=f.avg_ear, mar=f.mar,
            mouth_width_ratio=f.mouth_width_ratio,
            avg_brow_height=f.avg_brow_height,
            brow_furrow=f.brow_furrow,
        )
        self.calibrated = True
        log.info(f"Calibrated: EAR={f.avg_ear:.3f} MAR={f.mar:.3f} BF={f.brow_furrow:.3f}")

    def try_auto_calibrate(self, f, cnn_neutral):
        if self.calibrated or not self.cfg.auto_calibrate:
            return
        if not f.is_valid or cnn_neutral < self.cfg.auto_cal_neutral_min:
            return
        self._cal_buffer.append(f)
        if len(self._cal_buffer) >= self.cfg.auto_cal_frames:
            self.baseline = _Baseline(
                avg_ear=float(np.mean([g.avg_ear for g in self._cal_buffer])),
                mar=float(np.mean([g.mar for g in self._cal_buffer])),
                mouth_width_ratio=float(np.mean([g.mouth_width_ratio for g in self._cal_buffer])),
                avg_brow_height=float(np.mean([g.avg_brow_height for g in self._cal_buffer])),
                brow_furrow=float(np.mean([g.brow_furrow for g in self._cal_buffer])),
            )
            self.calibrated = True
            self._cal_buffer.clear()
            log.info(f"Auto-calibrated: EAR={self.baseline.avg_ear:.3f}")

    def reset(self):
        self.baseline = _Baseline()
        self.calibrated = False
        self._cal_buffer.clear()


# ════════════════════════════════════════════════
# TEMPORAL SMOOTHER
# ════════════════════════════════════════════════
class _Smoother:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ema = np.ones(NUM_CLASSES) / NUM_CLASSES
        self._init = False
        self._votes = deque(maxlen=cfg.vote_window)
        self.current = 'neutral'
        self._count = 0

    def update(self, scores):
        self._count += 1
        a = self.cfg.ema_alpha
        if not self._init:
            self.ema = 0.5 * (np.ones(NUM_CLASSES) / NUM_CLASSES) + 0.5 * scores
            self._init = True
        else:
            self.ema = a * scores + (1 - a) * self.ema
        t = self.ema.sum()
        if t > 1e-6:
            self.ema /= t

        idx = int(np.argmax(self.ema))
        cand = CLASS_LABELS[idx]
        conf = float(self.ema[idx])

        self._votes.append(cand)
        if len(self._votes) >= 3:
            vc = Counter(self._votes)
            top, cnt = vc.most_common(1)[0]
            if cnt / len(self._votes) >= 0.5:
                cand = top

        sa = np.sort(self.ema)[::-1]
        margin = float(sa[0] - sa[1]) if len(sa) > 1 else 0
        cal_conf = 0.6 * conf + 0.4 * margin

        if cal_conf >= self.cfg.min_confidence or self._count <= 5:
            if cand != self.current:
                self.current = cand

        return {
            'emotion': self.current,
            'confidence': cal_conf,
            'all_scores': {CLASS_LABELS[i]: round(float(self.ema[i]) * 100, 1)
                           for i in range(NUM_CLASSES)},
            'margin': margin,
        }

    def update_raw(self, scores):
        """No smoothing — just argmax."""
        idx = int(np.argmax(scores))
        sa = np.sort(scores)[::-1]
        margin = float(sa[0] - sa[1]) if len(sa) > 1 else 0
        return {
            'emotion': CLASS_LABELS[idx],
            'confidence': float(scores[idx]),
            'all_scores': {CLASS_LABELS[i]: round(float(scores[i]) * 100, 1)
                           for i in range(NUM_CLASSES)},
            'margin': margin,
        }

    def reset(self):
        self.ema = np.ones(NUM_CLASSES) / NUM_CLASSES
        self._init = False
        self._votes.clear()
        self.current = 'neutral'
        self._count = 0


# ════════════════════════════════════════════════════════
# MAIN DETECTOR
# ════════════════════════════════════════════════════════
class EmotionDetector:
    """
    High-accuracy emotion detector.

    Usage:
        detector = EmotionDetector(model_path, cascade_path)
        result = detector.detect(frame)
        # result keys: emotion, confidence, all_scores, bbox,
        #              faces_count, margin, geo, lm_px,
        #              raw_scores, boosted_scores
    """

    def __init__(self, model_path, cascade_path, config=None):
        self.cfg = config or DetectorConfig()

        _import_keras()
        log.info("Loading emotion model...")
        self.model = _load_model(model_path, compile=False)

        inp, out = self.model.input_shape, self.model.output_shape
        log.info(f"Model: input={inp} output={out}")
        assert inp == (None, 48, 48, 1), f"Bad input: {inp}"
        assert out == (None, 5), f"Bad output: {out}"

        # Fast inference
        try:
            test = np.zeros((1, 48, 48, 1), dtype=np.float32)
            _ = self.model(test, training=False)
            self._infer = lambda x: self.model(x, training=False).numpy()
        except Exception:
            self._infer = lambda x: self.model.predict(x, verbose=0)

        test_out = self._infer(np.random.rand(1, 48, 48, 1).astype(np.float32))[0]
        self._needs_softmax = abs(test_out.sum() - 1.0) > 0.1

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load cascade: {cascade_path}")

        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_grid, self.cfg.clahe_grid))

        self._face_mesh = None
        if MEDIAPIPE_OK and self.cfg.geo_boost_enabled:
            self._face_mesh = mp_lib.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

        self._smoother = _Smoother(self.cfg)
        self._geo = _GeoAnalyzer(self.cfg) if MEDIAPIPE_OK else None
        self._lock = threading.Lock()

        self._no_face = 0
        self._last_result = None
        self._use_smoothing = True

        log.info(
            f"TTA={'ON' if self.cfg.tta_enabled else 'OFF'} | "
            f"CLAHE={'ON' if self.cfg.clahe_enabled else 'OFF'} | "
            f"Boost={'ON' if self.cfg.class_boost_enabled else 'OFF'} | "
            f"Geo={'ON' if self.cfg.geo_boost_enabled and MEDIAPIPE_OK else 'OFF'} | "
            f"MCrop={'ON' if self.cfg.multi_crop_enabled else 'OFF'}"
        )

    # ── Public API ──────────────────────────────

    def detect(self, frame):
        """
        Process one BGR frame. Returns detailed result dict.

        Returns dict with:
            emotion, confidence, all_scores, bbox, faces_count,
            margin, geo, lm_px, raw_scores, boosted_scores
        """
        result = {
            'emotion': 'neutral', 'confidence': 0.0,
            'all_scores': {}, 'bbox': None, 'faces_count': 0,
            'margin': 0.0, 'geo': None, 'lm_px': None,
            'raw_scores': None, 'boosted_scores': None,
        }

        if frame is None or frame.size == 0:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = self.cascade.detectMultiScale(
            gray, self.cfg.haar_scale, self.cfg.haar_neighbors,
            minSize=(self.cfg.haar_min_size, self.cfg.haar_min_size))
        result['faces_count'] = len(faces)

        if len(faces) == 0:
            self._no_face += 1
            if self._last_result and self._no_face <= self.cfg.face_persist_frames:
                result['emotion'] = self._last_result['emotion']
                result['confidence'] = self._last_result['confidence'] * 0.7
                result['all_scores'] = self._last_result.get('all_scores', {})
            return result

        self._no_face = 0
        areas = [fw * fh for _, _, fw, fh in faces]
        bi = int(np.argmax(areas))
        fx, fy, fw, fh = faces[bi]
        result['bbox'] = (int(fx), int(fy), int(fw), int(fh))

        # CNN prediction
        ensemble = self._predict_multi_crop(gray, fx, fy, fw, fh)
        result['raw_scores'] = ensemble.copy()

        # Geometric features
        geo = None
        if self._face_mesh and self._geo:
            with self._lock:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_out = self._face_mesh.process(rgb)
                    if mp_out.multi_face_landmarks:
                        fl = mp_out.multi_face_landmarks[0]
                        lm_px = np.array([(lm.x * w, lm.y * h) for lm in fl.landmark])
                        result['lm_px'] = lm_px
                        geo = self._geo.extract(lm_px, fl)
                        result['geo'] = geo
                        if geo.is_valid:
                            self._geo.try_auto_calibrate(geo, float(ensemble[IDX_NEUTRAL]))
                except Exception as e:
                    log.error(f"MediaPipe process error: {e}")
                    # Re-init face mesh if it gets into an invalid state
                    try:
                        self._face_mesh.close()
                    except:
                        pass
                    self._face_mesh = mp_lib.solutions.face_mesh.FaceMesh(
                        static_image_mode=False, max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
                    log.info("MediaPipe FaceMesh re-initialized after crash")

        # Apply fixes
        calibrated = self._apply_class_boost(ensemble)
        boosted = self._apply_geo_boost(calibrated, geo)
        result['boosted_scores'] = boosted.copy()

        # Temporal smoothing
        if self._use_smoothing:
            temporal = self._smoother.update(boosted)
        else:
            temporal = self._smoother.update_raw(boosted)
        result.update(temporal)

        self._last_result = result
        return result

    def detect_batch(self, frames):
        """Process multiple frames, return (emotion, conf, scores)."""
        self.reset()
        last = None
        for frame in frames:
            if frame is None:
                continue
            r = self.detect(frame)
            if r['confidence'] > 0:
                last = r
        if last is None:
            return 'neutral', 0.0, {}
        return last['emotion'], last['confidence'], last['all_scores']

    def detect_raw(self, frame):
        """Single frame, no smoothing."""
        old = self._use_smoothing
        self._use_smoothing = False
        result = self.detect(frame)
        self._use_smoothing = old
        return result

    def reset(self):
        self._smoother.reset()
        self._no_face = 0
        self._last_result = None
        if self._geo:
            self._geo.reset()

    def calibrate_neutral(self, frame):
        """Calibrate neutral baseline. Returns True if successful."""
        if not self._face_mesh or not self._geo:
            return False
        h, w = frame.shape[:2]
        with self._lock:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_out = self._face_mesh.process(rgb)
        if not mp_out.multi_face_landmarks:
            return False
        fl = mp_out.multi_face_landmarks[0]
        lm_px = np.array([(lm.x * w, lm.y * h) for lm in fl.landmark])
        geo = self._geo.extract(lm_px, fl)
        if not geo.is_valid:
            return False
        self._geo.calibrate(geo)
        return True

    @property
    def is_calibrated(self):
        return self._geo is not None and self._geo.calibrated

    @property
    def geo_analyzer(self):
        return self._geo

    @property
    def smoother(self):
        return self._smoother

    # ── Internal ────────────────────────────────

    def _preprocess_roi(self, gray_roi, apply_clahe=True):
        if gray_roi.size == 0:
            return np.zeros((1, 48, 48, 1), dtype='float32')
        if apply_clahe and self.cfg.clahe_enabled:
            gray_roi = self._clahe.apply(gray_roi)
        resized = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)
        return (resized.astype('float32') / 255.0).reshape(1, 48, 48, 1)

    def _predict_single(self, tensor):
        raw = self._infer(tensor)[0]
        if self._needs_softmax:
            e = np.exp(raw - np.max(raw))
            raw = e / e.sum()
        return raw

    def _predict_with_tta(self, gray_roi):
        t1 = self._preprocess_roi(gray_roi, apply_clahe=True)
        p1 = self._predict_single(t1)
        if not self.cfg.tta_enabled:
            return p1
        flipped = cv2.flip(gray_roi, 1)
        t2 = self._preprocess_roi(flipped, apply_clahe=True)
        p2 = self._predict_single(t2)
        combined = self.cfg.tta_weight_original * p1 + self.cfg.tta_weight_flipped * p2
        t = combined.sum()
        return combined / t if t > 1e-6 else p1

    def _predict_multi_crop(self, gray, x, y, w, h):
        gh, gw = gray.shape[:2]
        crops = []
        roi1 = gray[y:y+h, x:x+w]
        if roi1.size > 0:
            crops.append(roi1)
        if not self.cfg.multi_crop_enabled:
            return self._predict_with_tta(crops[0]) if crops else np.ones(NUM_CLASSES) / NUM_CLASSES

        pad = int(0.10 * min(w, h))
        x2, y2, w2, h2 = x+pad, y+pad, w-2*pad, h-2*pad
        if w2 > 20 and h2 > 20:
            roi2 = gray[y2:y2+h2, x2:x2+w2]
            if roi2.size > 0:
                crops.append(roi2)

        pad = int(0.15 * min(w, h))
        x3, y3 = max(0, x-pad), max(0, y-pad)
        w3, h3 = min(gw-x3, w+2*pad), min(gh-y3, h+2*pad)
        if w3 > 20 and h3 > 20:
            roi3 = gray[y3:y3+h3, x3:x3+w3]
            if roi3.size > 0:
                crops.append(roi3)

        if not crops:
            return np.ones(NUM_CLASSES) / NUM_CLASSES
        preds = [self._predict_with_tta(roi) for roi in crops]
        ens = np.mean(preds, axis=0)
        t = ens.sum()
        return ens / t if t > 1e-6 else preds[0]

    def _apply_class_boost(self, scores):
        if not self.cfg.class_boost_enabled:
            return scores.copy()
        b = scores.copy()
        for i, l in enumerate(CLASS_LABELS):
            b[i] *= self.cfg.class_boost.get(l, 1.0)
        t = b.sum()
        return b / t if t > 1e-6 else scores.copy()

    def _apply_geo_boost(self, scores, geo):
        if (not self.cfg.geo_boost_enabled or geo is None or
                not geo.is_valid or not self._geo or not self._geo.calibrated):
            return scores.copy()
        b = scores.copy()
        s = self.cfg.geo_boost_strength
        if geo.angry_score > self.cfg.geo_angry_min_confidence:
            boost = s * (geo.angry_score - self.cfg.geo_angry_min_confidence) / \
                    (1.0 - self.cfg.geo_angry_min_confidence)
            b[IDX_ANGRY] += boost
            b[IDX_NEUTRAL] = max(0, b[IDX_NEUTRAL] - boost * 0.5)
        if geo.sad_score > self.cfg.geo_sad_min_confidence:
            boost = s * (geo.sad_score - self.cfg.geo_sad_min_confidence) / \
                    (1.0 - self.cfg.geo_sad_min_confidence)
            b[IDX_SAD] += boost
            b[IDX_NEUTRAL] = max(0, b[IDX_NEUTRAL] - boost * 0.5)
        t = b.sum()
        return b / t if t > 1e-6 else scores.copy()

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass