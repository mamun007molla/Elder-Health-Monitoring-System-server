# backend/app/ai/fall_detection/detector.py
"""
Fall Detection wrapper — integrates Multimodal_Final.py into the FastAPI backend.
Automatically falls back to vision-only if audio extraction fails.
"""
import os, sys, uuid, asyncio, subprocess, re
from pathlib import Path

BASE_DIR = Path(__file__).parent
XGB_MODEL_PATH = str(BASE_DIR / "models" / "video" / "xgb_final_model.json")
AST_MODEL_PATH = str(BASE_DIR / "models" / "audio" / "ast_model.torchscript.pt")
AST_PREP_CONFIG = str(BASE_DIR / "models" / "audio" / "preprocessor_config.json")
AST_LABEL_MAP = str(BASE_DIR / "models" / "audio" / "label_map.json")
PER_FRAME_PATH = str(BASE_DIR / "feature_extraction" / "per-frame-best.py")
FE_ENG_PATH = str(BASE_DIR / "feature_extraction" / "final-feature-eng-best.py")


def _patch_config():
    import types

    config_mod = types.ModuleType("config")
    config_mod.VIDEO_PATH = ""
    config_mod.XGB_MODEL_PATH = XGB_MODEL_PATH
    config_mod.AST_MODEL_PATH = AST_MODEL_PATH
    config_mod.AST_PREP_CONFIG = AST_PREP_CONFIG
    config_mod.AST_LABEL_MAP = AST_LABEL_MAP
    config_mod.PER_FRAME_PATH = PER_FRAME_PATH
    config_mod.FE_ENG_PATH = FE_ENG_PATH
    sys.modules["config"] = config_mod


def _test_ffmpeg_audio(video_path: str) -> bool:
    """Actually try to extract a tiny audio clip — if it works, audio exists."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-t",
                "1",  # only 1 second test
                tmp,
            ],
            capture_output=True,
            timeout=15,
        )
        success = (
            result.returncode == 0
            and os.path.exists(tmp)
            and os.path.getsize(tmp) > 1000
        )
        return success
    except Exception:
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass


async def run_fall_detection(video_bytes: bytes, upload_dir: str = "./uploads") -> dict:
    os.makedirs(os.path.join(upload_dir, "fall_detection"), exist_ok=True)

    input_id = uuid.uuid4().hex
    input_path = os.path.join(upload_dir, "fall_detection", f"input_{input_id}.mp4")
    output_path = os.path.join(upload_dir, "fall_detection", f"output_{input_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(video_bytes)

    # Test if audio extraction actually works
    has_audio = _test_ffmpeg_audio(input_path)
    print(f"[Fall Detection] has_audio={has_audio}")

    try:
        _patch_config()
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "multimodal_final", str(BASE_DIR / "Multimodal_Final.py")
        )
        multimodal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(multimodal)

        loop = asyncio.get_event_loop()

        def run_pipeline():
            import io
            from contextlib import redirect_stdout

            if has_audio:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    multimodal.run_parallel_per_window(
                        input_path,
                        XGB_MODEL_PATH,
                        AST_PREP_CONFIG,
                        AST_MODEL_PATH,
                        AST_LABEL_MAP,
                        output_path,
                        win_s=3.0,
                        show=False,
                    )
                return buf.getvalue()
            else:
                print("[Fall Detection] No audio — using vision-only pipeline")
                return _run_vision_only(input_path, output_path)

        log_output = await loop.run_in_executor(None, run_pipeline)

        # Parse results
        fall_detected = False
        confidence = 0.0
        segments = []
        for line in log_output.strip().split("\n"):
            if "FALL" in line:
                is_fall = "NOT FALL" not in line
                if is_fall:
                    fall_detected = True
                m = re.search(r"(\d+\.\d+)%", line)
                if m:
                    confidence = max(confidence, float(m.group(1)) / 100)
                segments.append({"fall": is_fall, "log": line.strip()})

        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass

        return {
            "fall_detected": fall_detected,
            "confidence": round(confidence, 2),
            "output_video_url": (
                f"/uploads/fall_detection/output_{input_id}.mp4"
                if os.path.exists(output_path)
                else None
            ),
            "segments": segments,
            "has_audio": has_audio,
            "mode": "multimodal (audio+vision)" if has_audio else "vision-only",
            "message": "⚠️ FALL DETECTED!" if fall_detected else "✅ No fall detected",
        }

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[Fall Detection ERROR]\n{tb}")
        for p in [input_path, output_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
        raise RuntimeError(f"Fall detection failed: {str(e)}\n{tb}")


def _run_vision_only(input_path: str, output_path: str) -> str:
    import cv2, numpy as np, pandas as pd, importlib.util, tempfile

    per_frame = importlib.util.module_from_spec(
        s := importlib.util.spec_from_file_location("per_frame_best", PER_FRAME_PATH)
    )
    s.loader.exec_module(per_frame)

    fe_eng = importlib.util.module_from_spec(
        s2 := importlib.util.spec_from_file_location(
            "final_feature_eng_best", FE_ENG_PATH
        )
    )
    s2.loader.exec_module(fe_eng)

    import xgboost as xgb

    clf = xgb.XGBClassifier()
    clf.load_model(XGB_MODEL_PATH)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    win_frames = int(round(3.0 * fps))
    label_map = ["NOT FALL", "FALL"]
    log_lines, seg_idx = [], 0

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

            tmpv = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            vw = cv2.VideoWriter(tmpv, fourcc, fps, (W, H))
            for f in frames:
                vw.write(f)
            vw.release()

            try:
                df_pf = per_frame.extract_per_frame(tmpv, fps=fps, n=len(frames))
                os.remove(tmpv)
                if df_pf is None or df_pf.empty:
                    probs = np.array([0.5, 0.5])
                else:
                    feats = fe_eng.extract_advanced_features(df_pf)
                    row = pd.DataFrame([feats])
                    if hasattr(clf, "feature_names_in_"):
                        names = list(clf.feature_names_in_)
                        for c in names:
                            if c not in row:
                                row[c] = 0.0
                        row = row.reindex(columns=names)
                    probs = clf.predict_proba(row)[0]
            except Exception:
                probs = np.array([0.5, 0.5])

            vi = int(np.argmax(probs))
            conf = float(probs[vi]) * 100
            color = (0, 255, 0) if vi == 0 else (0, 0, 255)
            txt_v = f"VISION (XGBoost): {label_map[vi]} {conf:.1f}%"
            txt_e = f"ENSEMBLE: {label_map[vi]} {conf:.1f}%"

            for fr in frames:
                disp = fr.copy()
                cv2.rectangle(disp, (10, 10), (520, 75), (0, 0, 0), -1)
                cv2.putText(
                    disp, txt_v, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                cv2.putText(
                    disp, txt_e, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                writer.write(disp)

            log_lines.append(f"[{seg_idx}] {txt_v} | {txt_e}")
            seg_idx += 1
    finally:
        cap.release()
        writer.release()

    return "\n".join(log_lines)
