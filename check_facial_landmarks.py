from pathlib import Path
import tempfile

import cv2
import face_alignment
import numpy as np
from feat import Detector

from get_facial_landmarks import get_openface_landmarks

IMAGE_ROOT = Path("Clinical Photos")
IMAGE_EXTENSIONS = {".jpg"}
FACIAL_REGIONS = {
    "Jaw": slice(0, 17),
    "Right eyebrow": slice(17, 22),
    "Left eyebrow": slice(22, 27),
    "Nose": slice(27, 36),
    "Right eye": slice(36, 42),
    "Left eye": slice(42, 48),
    "Mouth": slice(48, 68),
}


def get_pyfeat_landmarks(detector, image_path):
    """Read PyFeat's first detected face as pixel coordinates [68,2].

    Args:
        detector: Initialized PyFeat Detector.
        image_path: Path to one resized image.
    Returns:
        Floating-point coordinates [68,2].
    """
    result = detector.detect_image(str(image_path))
    x_names = [f"x_{index}" for index in range(68)]
    y_names = [f"y_{index}" for index in range(68)]
    first_face = result.landmarks.iloc[0]
    return np.column_stack((first_face[x_names], first_face[y_names])).astype(float)


def get_three_landmark_sets(image, fan, pyfeat, temporary_image_path):
    """Detect three landmark sets [3,68,2].

    Args:
        image: Original image [height,width,3].
        fan: Initialized FAN face-alignment detector.
        pyfeat: Initialized PyFeat Detector.
        temporary_image_path: Path for a resized image supplied to PyFeat.
    Returns:
        Pixel coordinates [3,68,2], ordered OpenFace, PyFeat, FAN.
    """
    original_height, original_width = image.shape[:2]
    resized = cv2.resize(
        image,
        (round(original_width * 0.5), round(original_height * 0.5)),
        interpolation=cv2.INTER_AREA,
    )
    openface_landmarks, _ = get_openface_landmarks(resized)

    fan_faces = fan.get_landmarks_from_image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    fan_landmarks = np.asarray(fan_faces[0], dtype=float)

    pyfeat_landmarks = get_pyfeat_landmarks(pyfeat, temporary_image_path)

    coordinate_scale = np.array([
        original_width / resized.shape[1],
        original_height / resized.shape[0],
    ])
    return np.stack((openface_landmarks, pyfeat_landmarks, fan_landmarks)) * coordinate_scale


def calculate_distances(landmarks):
    """Measure detector-pair errors [68] and mean inter-ocular span.

    Args:
        landmarks: OpenFace, PyFeat, FAN pixel coordinates [3,68,2].
    Returns:
        Mean pairwise Euclidean error per landmark [68] and scalar mean
        inter-ocular distance across the three detectors.
    """
    pair_distances = np.stack([
        np.linalg.norm(landmarks[first] - landmarks[second], axis=1)
        for first, second in ((0, 1), (0, 2), (1, 2))
    ])
    right_eye_centers = landmarks[:, FACIAL_REGIONS["Right eye"]].mean(axis=1)
    left_eye_centers = landmarks[:, FACIAL_REGIONS["Left eye"]].mean(axis=1)
    interocular_distance = np.linalg.norm(right_eye_centers - left_eye_centers, axis=1).mean()
    return pair_distances.mean(axis=0), interocular_distance


def compare_landmarks():
    image_paths = sorted(
        path for path in IMAGE_ROOT.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.lower().startswith("landmark_")
                    for part in path.relative_to(IMAGE_ROOT).parts)
        and not path.stem.lower().endswith(("_landmarks", "_overlay"))
    )

    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device="cpu", flip_input=False
    )
    pyfeat = Detector()
    landmark_distances = []
    interocular_distances = []
    image_sizes = []

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_image_path = Path(temporary_dir) / "landmark_comparison.png"
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            landmarks = get_three_landmark_sets(
                image, fan, pyfeat, temporary_image_path
            )
            distances, interocular_distance = calculate_distances(landmarks)
            landmark_distances.append(distances)
            interocular_distances.append(interocular_distance)
            image_sizes.append((image.shape[1], image.shape[0]))

    mean_distances = np.mean(landmark_distances, axis=0)
    mean_distance = mean_distances.mean()
    mean_interocular_distance = np.mean(interocular_distances)
    mean_width, mean_height = np.mean(image_sizes, axis=0)
    print(f"Mean image resolution: {mean_width:.2f} x {mean_height:.2f} pixels")
    print(f"Mean inter-ocular distance: {mean_interocular_distance:.2f} pixels")
    print(f"Mean landmark distance: {mean_distance:.2f} pixels")
    print(f"Relative inter-ocular distance: {100 * mean_distance / mean_interocular_distance:.2f}%")
    print("Mean landmark distance by facial region (pixels):")
    for region, indices in FACIAL_REGIONS.items():
        print(f"  {region}: {mean_distances[indices].mean():.2f}")


if __name__ == "__main__":
    compare_landmarks()
