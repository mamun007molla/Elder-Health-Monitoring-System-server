import os
import sys
import time
import json
import tempfile
import subprocess
import importlib.util
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

import config


def load_module(mod_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_xgb(model_path):
    import xgboost as xgb

    if model_path.lower().endswith((".json", ".ubj")):
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        return clf
    try:
        import joblib

        return joblib.load(model_path)
    except:
        import pickle

        with open(model_path, "rb") as f:
            return pickle.load(f)


def extract_audio_to_array(video_path, sr=16000):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        tmp_path,
    ]
    subprocess.run(cmd, check=True)
    import librosa

    wav, _ = librosa.load(tmp_path, sr=sr, mono=True)
    os.remove(tmp_path)
    return wav


def soft_voting_ensemble(
    audio_probs, vision_probs, audio_weight=0.5, vision_weight=0.5
):
    total = audio_weight + vision_weight
    aw = audio_weight / total
    vw = vision_weight / total
    ens = aw * audio_probs + vw * vision_probs
    idx = int(np.argmax(ens))
    return ens, idx, float(ens[idx])


def to_fall_text(idx, label_map):
    lab = label_map[idx].lower()
    if "fall" in lab and "non" not in lab:
        return "FALL"
    return "NOT FALL"


def run_parallel_per_window(
    video_path: str,
    xgb_model_path: str,
    ast_prep_config: str,
    ast_model_path: str,
    ast_label_map: str,
    out_path: str,
    win_s: float = 3.0,
    sr: int = 16000,
    show: bool = True,
):
    # Load helper modules
    per_frame = load_module("per_frame_best", config.PER_FRAME_PATH)
    fe_eng = load_module("final_feature_eng_best", config.FE_ENG_PATH)

    # Load XGBoost model
    clf = load_xgb(xgb_model_path)

    # Load AST model & config
    from transformers import ASTFeatureExtractor
    import torch

    fe = ASTFeatureExtractor.from_pretrained(ast_prep_config)
    ast_model = torch.jit.load(ast_model_path, map_location="cpu").eval()
    with open(ast_label_map, "r") as f:
        lm = json.load(f)
    label_map = [lm[k] for k in sorted(lm, key=lambda x: int(x))]

    # Precompute audio probabilities per window
    audio = extract_audio_to_array(video_path, sr=sr)
    window_size = int(sr * win_s)
    total_samples = len(audio)
    num_windows = int(np.ceil(total_samples / window_size))
    audio_probs_list = []
    for i in range(num_windows):
        seg = audio[i * window_size : (i + 1) * window_size]
        if len(seg) < window_size:
            seg = np.pad(seg, (0, window_size - len(seg)), "constant")
        inp = fe(
            seg,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            return_attention_mask=False,
        )["input_values"]
        with torch.no_grad():
            logits = ast_model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        audio_probs_list.append(probs)

    # Video I/O setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (W, H))

    win_frames = int(round(win_s * fps))
    # We'll delay showing the prediction until the center of the window
    overlay_frames = int(round(0.5 * fps))
    center_frame = win_frames // 2

    # Initialize sticky-fall flag
    fall_detected = False

    frame_idx = 0
    seg_idx = 0
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            frames = []
            for _ in range(win_frames):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            if not frames:
                break

            # Vision inference task
            def vis_task(frames_local):
                tmpv = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                vw = cv2.VideoWriter(
                    tmpv,
                    fourcc,
                    fps,
                    (frames_local[0].shape[1], frames_local[0].shape[0]),
                )
                for f in frames_local:
                    v = (
                        f
                        if f.dtype == np.uint8
                        else np.clip(f, 0, 255).astype(np.uint8)
                    )
                    if v.ndim == 2:
                        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
                    vw.write(v)
                vw.release()
                df_pf = per_frame.extract_per_frame(tmpv, fps=fps, n=len(frames_local))
                os.remove(tmpv)
                if df_pf is None or df_pf.empty:
                    return np.array([0.5, 0.5])
                feats = fe_eng.extract_advanced_features(df_pf)
                row = pd.DataFrame([feats])
                if hasattr(clf, "feature_names_in_"):
                    names = list(clf.feature_names_in_)
                    for c in names:
                        if c not in row:
                            row[c] = 0.0
                    row = row.reindex(columns=names)
                try:
                    return clf.predict_proba(row)[0]
                except:
                    return np.array([0.5, 0.5])

            vision_probs = executor.submit(vis_task, frames.copy()).result()

            # Audio probabilities for this segment
            audio_probs = (
                audio_probs_list[seg_idx]
                if seg_idx < len(audio_probs_list)
                else np.array([0.5, 0.5])
            )

            # Soft voting ensemble
            ensemble_probs, ensemble_idx, ensemble_conf = soft_voting_ensemble(
                audio_probs, vision_probs
            )

            # Sticky fall logic
            if not fall_detected and ensemble_idx == 1:
                fall_detected = True
            if fall_detected:
                ensemble_idx = 1
                ensemble_conf = 1.0

            # Individual model predictions
            ai = int(np.argmax(audio_probs))
            vi = int(np.argmax(vision_probs))

            txt_a = f"AUDIO (Audio Spectogram Transformer): {to_fall_text(ai, label_map)} {audio_probs[ai]*100:.1f}%"
            txt_v = f"VISION (Temporal Awared XGboost): {to_fall_text(vi, label_map)} {vision_probs[vi]*100:.1f}%"
            txt_e = f"ENSEMBLE (Soft Voting): {to_fall_text(ensemble_idx, label_map)} {ensemble_conf*100:.1f}%"

            # Annotate frames: only draw the predictions starting at the window center
            for i, fr in enumerate(frames):
                disp = fr.copy()
                # Dynamically size the background box to fit the texts
                x, y = 12, 12
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                line_gap = 8
                # Measure each text line
                (w1, h1), _ = cv2.getTextSize(txt_a, font, font_scale, thickness)
                (w2, h2), _ = cv2.getTextSize(txt_v, font, font_scale, thickness)
                (w3, h3), _ = cv2.getTextSize(txt_e, font, font_scale, thickness)
                max_w = max(w1, w2, w3)
                text_h = max(h1, h2, h3)
                pad_x = 12
                pad_y = 10
                pw = max(420, max_w + pad_x * 2)
                ph = pad_y * 2 + text_h * 3 + line_gap * 2
                overlay = disp.copy()
                cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)

                # Only show the labels from the center of the window onward
                # if i >= center_frame:
                # Draw each text line with measured spacing
                y0 = y + pad_y + text_h
                cv2.putText(
                    disp,
                    txt_a,
                    (x + pad_x, y0),
                    font,
                    font_scale,
                    (0, 255, 0) if "NOT" in txt_a else (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    disp,
                    txt_v,
                    (x + pad_x, y0 + text_h + line_gap),
                    font,
                    font_scale,
                    (0, 255, 0) if "NOT" in txt_v else (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    disp,
                    txt_e,
                    (x + pad_x, y0 + 2 * (text_h + line_gap)),
                    font,
                    font_scale,
                    (0, 255, 0) if "NOT" in txt_e else (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                writer.write(disp)
                # if show:
                #     target_t = start_time + frame_idx / max(1e-6, fps)
                #     now = time.perf_counter()
                #     if target_t > now:
                #         time.sleep(target_t - now)
                #     cv2.imshow("Soft Voting Ensemble", disp)
                #     if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                #         cap.release()
                #         writer.release()
            center_time = seg_idx * win_s + (win_s / 2.0)
            print(
                f"[{seg_idx}] center={center_time:.2f}s | {txt_a} | {txt_v} | {txt_e}"
            )

            seg_idx += 1

    finally:
        cap.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        executor.shutdown(wait=False)

    # Mux audio back
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        tmp_out_path,
        "-i",
        video_path,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp_out_path)
    print(f"Final annotated video with audio: {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 6:
        run_parallel_per_window(*args[:6])
    else:
        # Use defaults from config
        video = config.VIDEO_PATH
        xgbm = config.XGB_MODEL_PATH
        ast_prep = config.AST_PREP_CONFIG
        ast_mod = config.AST_MODEL_PATH
        ast_map = config.AST_LABEL_MAP
        # If the input video is known, name the annotated file using its basename
        if video:
            base = os.path.splitext(os.path.basename(video))[0]
            outp = os.path.join(os.getcwd(), f"{base}_annotated.mp4")
        else:
            outp = os.path.join(os.getcwd(), "annotated_parallel.mp4")
        run_parallel_per_window(video, xgbm, ast_prep, ast_mod, ast_map, outp)
