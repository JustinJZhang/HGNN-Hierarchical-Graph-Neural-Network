import os
import cv2
import openface
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import face_alignment
from feat import Detector

INPUT_ROOT = Path("Clinical Photos/Healthy Controls")
RESULT_ROOT = Path("Clinical Photos/landmark_HC")
MODEL_PATH = Path("/openface/models/dlib/shape_predictor_68_face_landmarks.dat")

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

LANDMARK_HEADER = (
        ['face', 'confidence'] +
        [f'x_{i}' for i in range(68)] +
        [f'y_{i}' for i in range(68)]
)


def validate_landmarks(lm, img_shape):
    if lm is None:
        return None
    try:
        lm = np.asarray(lm, dtype=np.float64).reshape(-1, 2)
        if lm.shape[0] != 68:
            return None
        if np.any(np.isnan(lm)):
            return None
        h, w = img_shape[:2]
        if np.any(lm[:, 0] < -w * 0.1) or np.any(lm[:, 0] > w * 1.1):
            return None
        if np.any(lm[:, 1] < -h * 0.1) or np.any(lm[:, 1] > h * 1.1):
            return None
        return lm
    except Exception:
        return None


def draw_overlay(img, landmarks, save_path, color=(0, 255, 0), radius=4, thickness=3):
    draw_img = img.copy()
    if landmarks is not None:
        try:
            landmarks = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
            for x, y in landmarks:
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(draw_img, (int(x), int(y)), radius, color, thickness)
        except Exception:
            pass
    cv2.imwrite(str(save_path), draw_img)


def main():
    """Detect and export 68 two-dimensional landmarks for every image under the input root."""
    align = openface.AlignDlib(str(MODEL_PATH))
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cpu",
        flip_input=False,
    )
    feat_detector = Detector()
    img_paths = sorted([p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in IMG_EXT])

    for img_path in img_paths:
        img_original = cv2.imread(str(img_path))
        if img_original is None:
            continue

        orig_height, orig_width = img_original.shape[:2]
        img = cv2.resize(
            img_original,
            (int(orig_width / 2), int(orig_height / 2)),
            interpolation=cv2.INTER_AREA,
        )
        rel_path = img_path.relative_to(INPUT_ROOT)
        out_dir = RESULT_ROOT / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = img_path.stem
        suffix = img_path.suffix
        out_img_path = out_dir / f"{base_name}_landmarks{suffix}"
        out_csv_path = out_dir / f"{base_name}_landmarks.csv"

        bbs = align.getAllFaceBoundingBoxes(img)
        if not bbs:
            pd.DataFrame(columns=LANDMARK_HEADER).to_csv(str(out_csv_path), index=False)
            cv2.imwrite(str(out_img_path), img)
            continue

        shape = align.predictor(img, bbs[0])
        lm_of = np.array([[point.x, point.y] for point in shape.parts()])
        draw_overlay(img, lm_of, out_img_path, color=(0, 255, 0))
        row = {column: np.nan for column in LANDMARK_HEADER}
        row["face"], row["confidence"] = 0, 0.95
        for landmark_index in range(68):
            row[f"x_{landmark_index}"], row[f"y_{landmark_index}"] = lm_of[landmark_index]
        pd.DataFrame([row], columns=LANDMARK_HEADER).to_csv(
            str(out_csv_path), index=False, float_format="%.4f"
        )

        lm_fan, lm_pyfeat = None, None
        try:
            lms = fa.get_landmarks(img)
            if lms is not None and len(lms) > 0:
                lm_fan = validate_landmarks(lms[0], img.shape)
        except Exception as error:
            print(f"FAN: {error}")

        if lm_fan is not None:
            draw_overlay(
                img,
                lm_fan,
                out_dir / f"{base_name}_fan_overlay{suffix}",
                color=(255, 0, 0),
            )

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary_file:
                cv2.imwrite(temporary_file.name, img)
                temporary_path = temporary_file.name
            out_feat = feat_detector.detect_image(temporary_path)
            os.unlink(temporary_path)

            if hasattr(out_feat, "landmarks") and out_feat.landmarks is not None:
                lms_df = out_feat.landmarks
                if isinstance(lms_df, pd.DataFrame) and not lms_df.empty:
                    x_cols = sorted(
                        [column for column in lms_df.columns
                         if column.startswith("x_") and column[2:].isdigit()],
                        key=lambda column: int(column.split("_")[1]),
                    )
                    y_cols = sorted(
                        [column for column in lms_df.columns
                         if column.startswith("y_") and column[2:].isdigit()],
                        key=lambda column: int(column.split("_")[1]),
                    )
                    if len(x_cols) == 68 and len(y_cols) == 68:
                        xs = lms_df[x_cols].iloc[0].astype(float).values
                        ys = lms_df[y_cols].iloc[0].astype(float).values
                        lm_pyfeat = validate_landmarks(
                            np.column_stack((xs, ys)), img.shape
                        )
        except Exception as error:
            print(f"Py-Feat: {error}")

        if lm_pyfeat is not None:
            draw_overlay(
                img,
                lm_pyfeat,
                out_dir / f"{base_name}_pyfeat_overlay{suffix}",
                color=(0, 0, 255),
            )


if __name__ == "__main__":
    main()
