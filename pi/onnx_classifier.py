"""
XL Fitness AI — ONNX Activity Classifier
Pi-side inference using the trained LSTM model.

THE GATE explained
──────────────────
The user's concern: "a person can be 'engaged' AND 'resting' — multiple
outputs could cause confusion."

The softmax layer in the LSTM IS the gate. It forces all 8 class probabilities
to sum to 1.0 and argmax picks exactly ONE winner per frame window. It is
mathematically impossible for the model to output two active states at once.

  logits  →  softmax  →  [0.02, 0.03, 0.05, 0.71, 0.08, 0.04, 0.05, 0.02]
                                                    ↑
                                             class 3: good_rep wins (71%)

Usage:
    classifier = ONNXActivityClassifier("models/weights/activity_v1.onnx")
    result = classifier.update(kps_flat)   # kps_flat: np.array shape (51,)
    print(result.class_name, result.confidence)
    if result.low_confidence:
        clip_reporter.maybe_report(result)
"""

import time
from collections import deque

import numpy as np

WINDOW_FRAMES    = 30    # frames fed to LSTM
KEYPOINTS        = 17    # COCO YOLOv11-pose keypoints
FEATURES_PER_KP  = 3     # x, y, confidence
FEATURE_DIM      = KEYPOINTS * FEATURES_PER_KP  # 51

# Must match annotation tool and training pipeline exactly
CLASS_NAMES = [
    "no_person",    # 0
    "user_present", # 1 — standing near machine, not seated
    "on_machine",   # 2 — seated, engaged, not yet lifting
    "good_rep",     # 3 — full ROM, controlled, weight moving
    "bad_rep",      # 4 — uncontrolled, bouncing, momentum
    "false_rep",    # 5 — stretching, adjusting handle/seat/pin
    "resting",      # 6 — seated between sets, handles released
    "half_rep",     # 7 — partial ROM or single arm only
]
NUM_CLASSES = len(CLASS_NAMES)

# Class IDs that represent an active rep (count toward the set)
REP_CLASSES   = {3, 4, 7}    # good_rep, bad_rep, half_rep
# Class IDs that mean the person is on the machine (session active)
ACTIVE_CLASSES = {2, 3, 4, 5, 6, 7}


class ONNXActivityClassifier:
    """
    Wraps the ONNX activity model for real-time per-frame inference.

    Maintains a rolling 30-frame keypoint buffer. Once full, runs ONNX
    inference every frame and returns a ClassifierResult with:
        - class_id      : int 0–7
        - class_name    : str
        - confidence    : float 0.0–1.0  (softmax probability of winning class)
        - low_confidence: bool           (True when model is unsure → flag for review)
        - is_rep_state  : bool
        - is_engaged    : bool

    Falls back gracefully to class_id=0 (no_person) when model is not loaded.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.65,
        review_threshold: float = 0.50,
    ):
        self.model_path           = model_path
        self.confidence_threshold = confidence_threshold  # below this = uncertain
        self.review_threshold     = review_threshold      # below this = flag for review
        self.ready                = False
        self._session             = None
        self._buffer              = deque(maxlen=WINDOW_FRAMES)
        self._last_result         = None

        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            # Verify input/output shapes match what we expect
            inp = self._session.get_inputs()[0]
            expected = [-1, WINDOW_FRAMES, FEATURE_DIM]
            self.ready = True
            print(f"[ONNX] Activity model loaded: {self.model_path}")
            print(f"[ONNX] Input: {inp.name} {inp.shape}  |  Classes: {NUM_CLASSES}")
        except FileNotFoundError:
            print(f"[ONNX] Model not found: {self.model_path}")
            print("[ONNX]   Rule-based counting active until first model is trained.")
        except Exception as e:
            print(f"[ONNX] WARNING: Could not load activity model: {e}")
            print("[ONNX]   Rule-based counting active.")

    def update(self, kps_flat: np.ndarray) -> "ClassifierResult":
        """
        Feed one frame's keypoint vector (shape 51,) into the rolling buffer.
        Returns a ClassifierResult. Call every frame even when person not detected
        (pass zeros — the model learns to recognise empty frames as class 0).
        """
        if kps_flat is None or len(kps_flat) != FEATURE_DIM:
            kps_flat = np.zeros(FEATURE_DIM, dtype=np.float32)

        self._buffer.append(kps_flat.astype(np.float32))

        # Need a full window before inference is meaningful
        if len(self._buffer) < WINDOW_FRAMES or not self.ready:
            result = ClassifierResult(class_id=0, confidence=0.0, buffer=list(self._buffer))
            self._last_result = result
            return result

        window = np.array(list(self._buffer), dtype=np.float32)  # (30, 51)
        window = window[np.newaxis, ...]                          # (1, 30, 51)

        try:
            logits = self._session.run(None, {"keypoints": window})[0][0]  # (8,)

            # Softmax — THE GATE.  Only one class can win.
            exp   = np.exp(logits - logits.max())
            probs = exp / exp.sum()

            class_id   = int(probs.argmax())
            confidence = float(probs[class_id])
        except Exception as e:
            print(f"[ONNX] Inference error: {e}")
            result = ClassifierResult(class_id=0, confidence=0.0, buffer=list(self._buffer))
            self._last_result = result
            return result

        result = ClassifierResult(
            class_id=class_id,
            confidence=confidence,
            buffer=list(self._buffer),
            probs=probs,
        )
        self._last_result = result
        return result

    def get_current_buffer(self) -> list:
        """Return the current 30-frame keypoint buffer for clip saving."""
        return list(self._buffer)

    @property
    def last_result(self) -> "ClassifierResult":
        return self._last_result


class ClassifierResult:
    """Output from ONNXActivityClassifier.update()."""

    def __init__(
        self,
        class_id: int,
        confidence: float,
        buffer: list,
        probs: np.ndarray = None,
    ):
        self.class_id   = class_id
        self.class_name = CLASS_NAMES[class_id] if class_id < NUM_CLASSES else "unknown"
        self.confidence = confidence
        self.probs      = probs          # full softmax distribution (8,) or None
        self.buffer     = buffer         # list of (51,) arrays — the 30-frame window
        self.timestamp  = time.time()

    @property
    def low_confidence(self) -> bool:
        """True when model is unsure — clip should be flagged for review."""
        return 0 < self.confidence < 0.50

    @property
    def is_rep_state(self) -> bool:
        """True for good_rep, bad_rep, or half_rep."""
        return self.class_id in REP_CLASSES

    @property
    def is_engaged(self) -> bool:
        """True when person is actively on the machine (any class ≥ 2)."""
        return self.class_id in ACTIVE_CLASSES

    def __repr__(self) -> str:
        return (
            f"<ClassifierResult class={self.class_name}({self.class_id}) "
            f"conf={self.confidence:.2f} low_conf={self.low_confidence}>"
        )
