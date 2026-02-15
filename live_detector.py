#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════╗
║   MOODIFY — Live Emotion Detection (Standalone)              ║
║                                                                ║
║   Uses EmotionDetector v7 module                              ║
║                                                                ║
║   Controls:                                                    ║
║   q — Quit          s — Screenshot     c — Calibrate neutral  ║
║   d — Debug toggle  g — GeoBst toggle  r — Reset buffers      ║
║   t — TTA toggle    e — CLAHE toggle   b — ClassBst toggle    ║
║   m — MultiCrop     v — Verbose        w — Raw scores panel   ║
║   1 — Raw mode      2 — Smoothed       3 — Full pipeline      ║
║   +/- — Smoothing   p — Freeze frame   h — Help               ║
║   f — Fullscreen    a — Auto-calibrate                         ║
╚════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import sys
import time
import logging
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('LiveDetector')

from emotion_detector import (
    EmotionDetector, DetectorConfig,
    CLASS_LABELS, NUM_CLASSES,
    EMOTION_COLORS, EMOTION_DISPLAY,
    _LM, MEDIAPIPE_OK,
)


class PipelineMode(Enum):
    RAW      = "raw"
    SMOOTHED = "smoothed"
    FULL     = "full"


# ════════════════════════════════════════════════
# LIVE DETECTOR UI
# ════════════════════════════════════════════════
class LiveDetectorUI:
    """
    Standalone OpenCV window with full visualization
    and keyboard controls for testing/debugging.
    """

    def __init__(self,
                 model_path='../keras/Emotion_Detection.h5',
                 cascade_path='../keras/haarcascade_frontalface_default.xml',
                 camera_index=0):

        self.camera_index = camera_index
        self.screenshot_dir = './screenshots'
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # ── Create detector ──────────────────────
        self.cfg = DetectorConfig(
            tta_enabled=True,
            clahe_enabled=True,
            class_boost_enabled=True,
            geo_boost_enabled=True,
            multi_crop_enabled=True,
            ema_alpha=0.50,
        )

        print("=" * 60)
        print("  MOODIFY — Live Emotion Detection v7")
        print("  Fixes: TTA + CLAHE + ClassBoost + GeoBoost + MultiCrop")
        print("=" * 60)

        self.detector = EmotionDetector(
            model_path=model_path,
            cascade_path=cascade_path,
            config=self.cfg,
        )

        # ── UI state ─────────────────────────────
        self.mode = PipelineMode.FULL
        self.show_debug = True
        self.show_raw_panel = True
        self.show_landmarks = True
        self.show_geo_panel = True
        self.verbose = True
        self.verbose_interval = 15
        self.frozen = False
        self.frozen_frame = None
        self.fullscreen = False
        self.total_frames = 0

        # ── Stats ────────────────────────────────
        self._emotion_counts = {l: 0 for l in CLASS_LABELS}
        self._fps_history = []
        self._session_start = time.time()

    # ══════════════════════════════════════════════
    # DRAWING METHODS
    # ══════════════════════════════════════════════

    def _draw_box(self, fr, bbox, emotion, conf):
        """Corner-style bounding box with label."""
        x, y, w, h = bbox
        col = EMOTION_COLORS.get(emotion, (180, 180, 180))
        cl = int(min(w, h) * 0.2)

        for cx, cy, dx, dy, ex, ey in [
            (x, y, cl, 0, 0, cl), (x+w, y, -cl, 0, 0, cl),
            (x, y+h, cl, 0, 0, -cl), (x+w, y+h, -cl, 0, 0, -cl)
        ]:
            cv2.line(fr, (cx, cy), (cx+dx, cy+dy), col, 3)
            cv2.line(fr, (cx, cy), (cx+ex, cy+ey), col, 3)

        label = EMOTION_DISPLAY.get(emotion, emotion.upper())
        cv2.putText(fr, label, (x, y-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(fr, f"{conf*100:.0f}%", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    def _draw_score_bars(self, fr, scores, x, y):
        """Horizontal score bars."""
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

    def _draw_raw_panel(self, fr, r):
        """Side panel: RAW vs BOOSTED scores."""
        fh, fw = fr.shape[:2]
        x = fw - 280
        y = 10

        raw = r.get('raw_scores')
        boosted = r.get('boosted_scores')
        if raw is None:
            return

        # Background
        ov = fr.copy()
        cv2.rectangle(ov, (x-5, y-5), (x+270, y+125), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, fr, 0.35, 0, fr)

        cv2.putText(fr, "RAW CNN", (x, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        cv2.putText(fr, "BOOSTED", (x+140, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 100), 1)

        for i, l in enumerate(CLASS_LABELS):
            ly = y + 25 + i * 18
            is_target = l in ('angry', 'sad')
            nc = (100, 200, 255) if is_target else (140, 140, 140)
            cv2.putText(fr, f"{l[:3].upper()}", (x-35, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, nc, 1)

            v = float(raw[i])
            col = (0, 255, 200) if i == int(np.argmax(raw)) else (140, 140, 140)
            cv2.putText(fr, f"{v*100:5.1f}%", (x, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

            if boosted is not None:
                v2 = float(boosted[i])
                col2 = (0, 255, 100) if i == int(np.argmax(boosted)) else (120, 120, 120)
                cv2.putText(fr, f"{v2*100:5.1f}%", (x+140, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col2, 1)

    def _draw_geo_panel(self, fr, geo):
        """Angry/Sad geometric signal panel."""
        if not geo or not geo.is_valid:
            return

        fh, fw = fr.shape[:2]
        x, y = fw - 220, 150

        ov = fr.copy()
        cv2.rectangle(ov, (x-5, y-12), (x+210, y+130), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, fr, 0.4, 0, fr)

        cal_ok = self.detector.is_calibrated
        lines = [
            ("CALIBRATED ✓" if cal_ok else "NOT CALIBRATED (c)",
             (0, 255, 0) if cal_ok else (0, 0, 255)),
            ("── ANGRY SIGNALS ──", (0, 100, 255)),
            (f"  Brow furrow:  {geo.angry_brow_furrow:.2f} {'█'*int(geo.angry_brow_furrow*10)}", None),
            (f"  Eye squint:   {geo.angry_eye_squint:.2f} {'█'*int(geo.angry_eye_squint*10)}", None),
            (f"  Lip press:    {geo.angry_lip_press:.2f} {'█'*int(geo.angry_lip_press*10)}", None),
            (f"  ANGRY TOTAL:  {geo.angry_score:.2f}", (0, 100, 255)),
            ("── SAD SIGNALS ────", (255, 100, 0)),
            (f"  Inner brow ↑: {geo.sad_inner_brow_raise:.2f} {'█'*int(geo.sad_inner_brow_raise*10)}", None),
            (f"  Lip corner ↓: {geo.sad_lip_corner_down:.2f} {'█'*int(geo.sad_lip_corner_down*10)}", None),
            (f"  Mouth narrow: {geo.sad_mouth_narrow:.2f} {'█'*int(geo.sad_mouth_narrow*10)}", None),
            (f"  SAD TOTAL:    {geo.sad_score:.2f}", (255, 100, 0)),
        ]

        for i, (txt, col) in enumerate(lines):
            cv2.putText(fr, txt, (x, y + i * 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col or (160, 160, 160), 1)

    def _draw_info(self, fr, fps, r):
        """Top-left info panel."""
        fh, fw = fr.shape[:2]
        ov = fr.copy()
        cv2.rectangle(ov, (0, 0), (280, 155), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, fr, 0.45, 0, fr)

        cal = "CAL:✓" if self.detector.is_calibrated else "CAL:—"
        elapsed = time.time() - self._session_start

        lines = [
            (f"FPS: {fps:.1f}", (0, 255, 0), 0.50, 2),
            (f"Faces: {r['faces_count']}   Mode: {self.mode.value}", (0, 200, 255), 0.40, 1),
            (f"Conf: {r['confidence']*100:.0f}%  Margin: {r.get('margin',0):.2f}  {cal}",
             (160, 160, 160), 0.36, 1),
            (f"TTA:{'ON' if self.cfg.tta_enabled else 'off'} "
             f"CLAHE:{'ON' if self.cfg.clahe_enabled else 'off'} "
             f"Boost:{'ON' if self.cfg.class_boost_enabled else 'off'}",
             (120, 180, 120), 0.34, 1),
            (f"Geo:{'ON' if self.cfg.geo_boost_enabled else 'off'} "
             f"MCrop:{'ON' if self.cfg.multi_crop_enabled else 'off'} "
             f"Smooth:{self.cfg.ema_alpha:.2f}",
             (120, 180, 120), 0.34, 1),
            (f"Frame: {self.total_frames}  Time: {elapsed:.0f}s",
             (100, 100, 100), 0.32, 1),
            (f"Session: " + " ".join(
                f"{l[:3]}={self._emotion_counts[l]}"
                for l in CLASS_LABELS),
             (100, 150, 100), 0.30, 1),
        ]

        for i, (txt, col, sc, th) in enumerate(lines):
            cv2.putText(fr, txt, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)

    def _draw_landmarks(self, fr, lm_px):
        """Draw facial landmark dots."""
        if lm_px is None:
            return
        for i in _LM.DISPLAY_LANDMARKS:
            if i < len(lm_px):
                cv2.circle(fr, (int(lm_px[i][0]), int(lm_px[i][1])),
                           2, (0, 255, 0), -1)

    def _draw_emotion_history_bar(self, fr):
        """Bottom bar showing emotion distribution."""
        fh, fw = fr.shape[:2]
        total = sum(self._emotion_counts.values())
        if total == 0:
            return

        bar_y = fh - 30
        bar_h = 15
        x = 10

        ov = fr.copy()
        cv2.rectangle(ov, (5, bar_y - 5), (fw - 5, bar_y + bar_h + 5),
                       (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.5, fr, 0.5, 0, fr)

        bar_w = fw - 20
        for l in CLASS_LABELS:
            pct = self._emotion_counts[l] / total
            seg_w = int(bar_w * pct)
            if seg_w > 0:
                col = EMOTION_COLORS.get(l, (128, 128, 128))
                cv2.rectangle(fr, (x, bar_y), (x + seg_w, bar_y + bar_h),
                               col, -1)
                if seg_w > 30:
                    cv2.putText(fr, f"{l[:3]} {pct*100:.0f}%",
                                (x + 3, bar_y + 11),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                (255, 255, 255), 1)
                x += seg_w

    # ══════════════════════════════════════════════
    # FREEZE FRAME ANALYSIS
    # ══════════════════════════════════════════════

    def _analyze_frozen(self, frame):
        """Deep analysis of a frozen frame."""
        log.info("=" * 55)
        log.info("FROZEN FRAME ANALYSIS")
        log.info("=" * 55)

        # Test with each fix toggled
        configs = [
            ("RAW (nothing)",
             dict(tta_enabled=False, clahe_enabled=False,
                  class_boost_enabled=False, geo_boost_enabled=False,
                  multi_crop_enabled=False)),
            ("+ CLAHE only",
             dict(tta_enabled=False, clahe_enabled=True,
                  class_boost_enabled=False, geo_boost_enabled=False,
                  multi_crop_enabled=False)),
            ("+ TTA only",
             dict(tta_enabled=True, clahe_enabled=False,
                  class_boost_enabled=False, geo_boost_enabled=False,
                  multi_crop_enabled=False)),
            ("+ CLAHE + TTA",
             dict(tta_enabled=True, clahe_enabled=True,
                  class_boost_enabled=False, geo_boost_enabled=False,
                  multi_crop_enabled=False)),
            ("+ All fixes",
             dict(tta_enabled=True, clahe_enabled=True,
                  class_boost_enabled=True, geo_boost_enabled=True,
                  multi_crop_enabled=True)),
        ]

        for label, overrides in configs:
            # Temporarily override config
            saved = {}
            for k, v in overrides.items():
                saved[k] = getattr(self.cfg, k)
                setattr(self.cfg, k, v)

            result = self.detector.detect_raw(frame)

            # Restore
            for k, v in saved.items():
                setattr(self.cfg, k, v)

            raw = result.get('raw_scores')
            emotion = result['emotion']
            conf = result['confidence']

            raw_str = ""
            if raw is not None:
                raw_str = " ".join(
                    f"{CLASS_LABELS[i][:3]}={raw[i]:.3f}"
                    for i in range(NUM_CLASSES)
                )

            log.info(f"  {label:25s} → {emotion:10s} ({conf:.0%})  [{raw_str}]")

        log.info("=" * 55)

    # ══════════════════════════════════════════════
    # MAIN RUN LOOP
    # ══════════════════════════════════════════════

    def run(self):
        """Main loop — opens camera, processes, displays."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        prev_t = time.time()
        fc = 0
        fps = 0.0

        window_name = "Moodify — Live Emotion Detection v7"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        log.info("Camera started — press 'h' for help")
        print()

        try:
            while True:
                # ── Read frame ───────────────────
                if not self.frozen:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    self.frozen_frame = frame.copy()
                else:
                    frame = self.frozen_frame.copy()

                clean = frame.copy()
                fh, fw = frame.shape[:2]
                self.total_frames += 1

                # FPS
                fc += 1
                now = time.time()
                if now - prev_t >= 1.0:
                    fps = fc / (now - prev_t)
                    self._fps_history.append(fps)
                    fc = 0
                    prev_t = now

                # ── Set pipeline mode ────────────
                if self.mode == PipelineMode.RAW:
                    self.detector._use_smoothing = False
                else:
                    self.detector._use_smoothing = True

                # ── Detect ───────────────────────
                r = self.detector.detect(frame)
                emotion = r['emotion']
                conf = r['confidence']
                scores = r.get('all_scores', {})
                bbox = r['bbox']
                geo = r.get('geo')
                lm_px = r.get('lm_px')

                # Track stats
                self._emotion_counts[emotion] = \
                    self._emotion_counts.get(emotion, 0) + 1

                # ── Verbose logging ──────────────
                if self.verbose and (
                    self.total_frames <= 20 or
                    self.total_frames % self.verbose_interval == 0
                ):
                    raw = r.get('raw_scores')
                    boosted = r.get('boosted_scores')
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
                        parts.append(
                            f"ang={geo.angry_score:.2f} "
                            f"sad={geo.sad_score:.2f}"
                        )
                    log.info(" | ".join(parts))

                # ── Draw ─────────────────────────
                if bbox:
                    self._draw_box(frame, bbox, emotion, conf)
                    bx = bbox[0] + bbox[2] + 15
                    if bx + 170 < fw:
                        self._draw_score_bars(frame, scores, bx, bbox[1])
                    elif bbox[0] > 180:
                        self._draw_score_bars(
                            frame, scores, bbox[0] - 175, bbox[1])

                if self.show_landmarks and lm_px is not None:
                    self._draw_landmarks(frame, lm_px)

                if self.show_raw_panel:
                    self._draw_raw_panel(frame, r)

                if self.show_geo_panel and geo and geo.is_valid:
                    self._draw_geo_panel(frame, geo)

                self._draw_info(frame, fps, r)
                self._draw_emotion_history_bar(frame)

                if self.frozen:
                    cv2.putText(
                        frame, "FROZEN (press 'p' to unfreeze)",
                        (fw//2 - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if r['faces_count'] == 0:
                    cv2.putText(
                        frame, "No face detected", (fw//2 - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Controls hint
                cv2.putText(
                    frame,
                    "q:Quit s:Shot c:Cal d:Debug r:Reset "
                    "t:TTA e:CLAHE g:Geo b:Boost m:MCrop "
                    "p:Freeze 1/2/3:Mode h:Help",
                    (5, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (90, 90, 90), 1)

                cv2.imshow(window_name, frame)

                # ── Keyboard ─────────────────────
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, clean, geo, fps)

                if key == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            self._print_session_summary()

    # ══════════════════════════════════════════════
    # KEY HANDLER
    # ══════════════════════════════════════════════

    def _handle_key(self, key, clean_frame, geo, fps):
        if key == 255:
            return

        if key == ord('s'):
            emotion = self.detector._smoother.current
            path = os.path.join(
                self.screenshot_dir,
                f"emo_{emotion}_{int(time.time())}.png")
            cv2.imwrite(path, clean_frame)
            log.info(f"Screenshot: {path}")

        elif key == ord('c'):
            ok = self.detector.calibrate_neutral(clean_frame)
            if ok:
                log.info("✔ Manually calibrated!")
            else:
                log.warning("No face — look at camera with neutral face")

        elif key == ord('a'):
            if self.detector.geo_analyzer:
                self.detector.geo_analyzer.reset()
                log.info("Auto-calibration reset — will re-calibrate")

        elif key == ord('d'):
            self.show_debug = not self.show_debug
            self.show_landmarks = self.show_debug
            self.show_geo_panel = self.show_debug

        elif key == ord('w'):
            self.show_raw_panel = not self.show_raw_panel

        elif key == ord('r'):
            self.detector.reset()
            self._emotion_counts = {l: 0 for l in CLASS_LABELS}
            log.info("Reset all buffers + stats")

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
            self.verbose = not self.verbose
            log.info(f"Verbose: {'ON' if self.verbose else 'OFF'}")

        elif key == ord('p'):
            self.frozen = not self.frozen
            if self.frozen:
                log.info("Frame FROZEN — analyzing...")
                self._analyze_frozen(self.frozen_frame)
            else:
                log.info("Unfrozen")

        elif key == ord('f'):
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                cv2.setWindowProperty(
                    "Moodify — Live Emotion Detection v7",
                    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(
                    "Moodify — Live Emotion Detection v7",
                    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        elif key == ord('1'):
            self.mode = PipelineMode.RAW
            self.detector.smoother.reset()
            log.info("Mode: RAW (no smoothing)")

        elif key == ord('2'):
            self.mode = PipelineMode.SMOOTHED
            self.detector.smoother.reset()
            log.info("Mode: SMOOTHED")

        elif key == ord('3'):
            self.mode = PipelineMode.FULL
            self.detector.smoother.reset()
            log.info("Mode: FULL")

        elif key in (ord('+'), ord('=')):
            self.cfg.ema_alpha = min(0.95, self.cfg.ema_alpha + 0.05)
            log.info(f"Smoothing alpha: {self.cfg.ema_alpha:.2f}")

        elif key in (ord('-'), ord('_')):
            self.cfg.ema_alpha = max(0.10, self.cfg.ema_alpha - 0.05)
            log.info(f"Smoothing alpha: {self.cfg.ema_alpha:.2f}")

        elif key == ord('h'):
            self._print_help()

    def _print_help(self):
        print("""
╔═══════════════════════════════════════════════╗
║  CONTROLS                                     ║
╠═══════════════════════════════════════════════╣
║  q — Quit              s — Screenshot         ║
║  c — Calibrate neutral (IMPORTANT!)           ║
║  a — Reset auto-calibration                   ║
║  d — Toggle debug overlays                    ║
║  w — Toggle raw scores panel                  ║
║  r — Reset all buffers + stats                ║
║  p — FREEZE frame + deep analysis             ║
║  v — Verbose logging                          ║
║  f — Fullscreen toggle                        ║
║                                               ║
║  ACCURACY FIXES:                              ║
║  t — Toggle TTA (flip averaging)              ║
║  e — Toggle CLAHE (contrast boost)            ║
║  b — Toggle per-class boosting                ║
║  g — Toggle geometric boosting                ║
║  m — Toggle multi-crop ensemble               ║
║                                               ║
║  PIPELINE:                                    ║
║  1 — RAW (no processing, no smoothing)        ║
║  2 — SMOOTHED (EMA + vote)                    ║
║  3 — FULL (all fixes + smoothing)             ║
║  +/- — Adjust EMA smoothing factor            ║
║                                               ║
║  TIP: Press 'c' while showing a NEUTRAL       ║
║  face for best angry/sad accuracy.            ║
╚═══════════════════════════════════════════════╝
        """)

    def _print_session_summary(self):
        """Print session stats on exit."""
        elapsed = time.time() - self._session_start
        total = sum(self._emotion_counts.values())
        avg_fps = np.mean(self._fps_history) if self._fps_history else 0

        print()
        print("=" * 55)
        print("  SESSION SUMMARY")
        print("=" * 55)
        print(f"  Duration:     {elapsed:.1f}s")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Average FPS:  {avg_fps:.1f}")
        print()
        print("  Emotion Distribution:")
        for l in CLASS_LABELS:
            count = self._emotion_counts[l]
            pct = (count / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"    {l:10s}: {count:5d} ({pct:5.1f}%) {bar}")
        print("=" * 55)


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Moodify Live Emotion Detection v7"
    )
    parser.add_argument(
        '--model', default='keras/Emotion_Detection.h5',
        help='Path to emotion model')
    parser.add_argument(
        '--cascade', default='keras/haarcascade_frontalface_default.xml',
        help='Path to Haar cascade')
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera index')
    parser.add_argument(
        '--no-tta', action='store_true',
        help='Disable TTA')
    parser.add_argument(
        '--no-clahe', action='store_true',
        help='Disable CLAHE')
    parser.add_argument(
        '--no-geo', action='store_true',
        help='Disable geometric boosting')

    args = parser.parse_args()

    ui = LiveDetectorUI(
        model_path=args.model,
        cascade_path=args.cascade,
        camera_index=args.camera,
    )

    if args.no_tta:
        ui.cfg.tta_enabled = False
    if args.no_clahe:
        ui.cfg.clahe_enabled = False
    if args.no_geo:
        ui.cfg.geo_boost_enabled = False

    ui.run()