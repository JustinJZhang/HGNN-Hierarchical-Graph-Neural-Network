EXPRESSIONS = (
    "resting position",
    "closing the eyes gently",
    "closing the eyes firmly",
    "raising the eyebrows",
    "frowning",
    "pursing the lips",
    "showing the teeth",
    "puffing of the cheeks",
)

EXPRESSION_DISPLAY_NAMES = (
    "Resting position",
    "Closing the eyes gently",
    "Closing the eyes firmly",
    "Raising the eyebrows",
    "Frowning",
    "Pursing the lips",
    "Showing the teeth",
    "Puffing of the cheeks",
)

ACTION_UNIT_KEYS = (
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
)

ACTION_UNIT_DISPLAY_NAMES = (
    "AU-IBR",
    "AU-OBR",
    "AU-BL",
    "AU-ULidR",
    "AU-CheR",
    "AU-LidT",
    "AU-NW",
    "AU-ULipR",
    "AU-LCP",
    "AU-D",
    "AU-LCD",
    "AU-ChiR",
    "AU-LS",
    "AU-LipT",
    "AU-LP",
    "AU-JD",
    "AU-B",
)

ASYMMETRY_DEVIATION_KEYS = (
    "CommisureHeightDeviation",
    "UpperLipHeightDeviation",
    "LowerLipHeightDeviation",
    "CommissureExcursion",
    "SmileAngle",
    "MarginalReflexDistance1",
    "MarginalReflexDistance2",
    "BrowHeight",
    "DentalShow",
    "PalpebralFissureHeight",
)

ASYMMETRY_SIDE_KEYS = (
    "CommissureExcursion",
    "SmileAngle",
    "MarginalReflexDistance1",
    "MarginalReflexDistance2",
    "BrowHeight",
    "DentalShow",
    "PalpebralFissureHeight",
)

ASYMMETRY_DEVIATION_DISPLAY_NAMES = (
    "Commissure Height Deviation",
    "Upper Lip Height Deviation",
    "Lower Lip Height Deviation",
    "Commissure Excursion Deviation",
    "Smile Angle Deviation",
    "Marginal Reflex Deviation 1",
    "Marginal Reflex Deviation 2",
    "Brow Height Deviation",
    "Dental Show Deviation",
    "Palpebral Fissure Height Deviation",
)

ASYMMETRY_SIDE_DISPLAY_NAMES = (
    "Commissure Excursion",
    "Smile Angle",
    "Marginal Reflex Distance 1",
    "Marginal Reflex Distance 2",
    "Brow Height",
    "Dental Show",
    "Palpebral Fissure Height",
)

ASYMMETRY_LEFT_DISPLAY_NAMES = tuple(
    f"Left {name}" for name in ASYMMETRY_SIDE_DISPLAY_NAMES
)
ASYMMETRY_RIGHT_DISPLAY_NAMES = tuple(
    f"Right {name}" for name in ASYMMETRY_SIDE_DISPLAY_NAMES
)

SEVERITY_CLASSES = ("absence", "moderate", "severe")
ASYMMETRY_CLASSES = ("non-asymmetry", "left", "right")

NUM_EXPRESSIONS = len(EXPRESSIONS)
NUM_ACTION_UNITS = len(ACTION_UNIT_KEYS)
NUM_ASYMMETRY_DEVIATIONS = len(ASYMMETRY_DEVIATION_KEYS)
NUM_ASYMMETRY_SIDE_FEATURES = len(ASYMMETRY_SIDE_KEYS)
NUM_ASYMMETRY_FEATURES = NUM_ASYMMETRY_DEVIATIONS + 2 * NUM_ASYMMETRY_SIDE_FEATURES
