import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from fastapi import HTTPException
from tensorflow import keras

from config import FILLER_DECISION_THRESHOLD, FILLER_METADATA_PATH, FILLER_MODEL_PATH

# FILLER MODEL CUSTOM OBJECTS
class WeightedSparseCategoricalCrossEntropy(keras.losses.Loss):
    def __init__(self, class_weight=None, name="weighted_sparse_categorical_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weight = class_weight

    def call(self, y_true, logits):
        y_true = tf.cast(y_true, tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits
        )
        return tf.reduce_mean(loss)


class MacroF1Score(keras.metrics.Metric):
    def __init__(self, num_classes=13, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, logits, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32)

        y_true_oh = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)
        y_pred_oh = tf.one_hot(y_pred, depth=self.num_classes, dtype=tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true_oh * y_pred_oh, axis=0))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true_oh) * y_pred_oh, axis=0))
        self.fn.assign_add(tf.reduce_sum(y_true_oh * (1.0 - y_pred_oh), axis=0))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))



# FILLER MODEL LOADING
_filler_model = None
_filler_metadata = None
_mel_weight_matrix = None

def get_filler_metadata():
    global _filler_metadata

    if _filler_metadata is None:
        if not FILLER_METADATA_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Model metadata was not found: {FILLER_METADATA_PATH}")

        with open(FILLER_METADATA_PATH, "r", encoding="utf-8") as f:
            _filler_metadata = json.load(f)

    return _filler_metadata


def get_filler_model():
    global _filler_model

    if _filler_model is None:
        if not FILLER_MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Filler model was not found: {FILLER_MODEL_PATH}")

        _filler_model = keras.models.load_model(
            FILLER_MODEL_PATH,
            custom_objects={
                "WeightedSparseCategoricalCrossEntropy": WeightedSparseCategoricalCrossEntropy,
                "MacroF1Score": MacroF1Score,
            },
            compile=False
        )

    return _filler_model


def get_mel_weight_matrix():
    global _mel_weight_matrix

    metadata = get_filler_metadata()

    sample_rate = int(metadata.get("sample_rate", 16000))
    num_mels = int(metadata.get("num_mels", 64))
    fft_length = int(metadata.get("fft_length", 512))
    num_spectrogram_bins = fft_length // 2 + 1

    if _mel_weight_matrix is None:
        _mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mels,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )

    return _mel_weight_matrix


# =========================
# BINARY FILLER DETECTION HELPERS
# =========================
# This backend version uses the binary model from:
# podcastfillers_filler_detection_training.ipynb
#
# Expected model output:
# - non_filler
# - filler
#
# The endpoint response is kept compatible with the previous pipeline:
# - summary_counts
# - filler_rate
# - delivery_score
# - timeline
#
# But full 13-class outputs such as Uh, Um, Breath, Noise, etc. are no longer used.

FILLER_DECISION_THRESHOLD = 0.50


def normalize_binary_label(label_name: str) -> str:
    label = str(label_name).lower().strip().replace("-", "_").replace(" ", "_")

    if label in ["filler", "fillers", "1", "true", "yes"]:
        return "filler"

    if label in ["non_filler", "nonfiller", "non_fillers", "non_filler_speech", "not_filler", "0", "false", "no"]:
        return "non_filler"

    return label


def get_binary_label_names(metadata: Dict[str, Any]) -> List[str]:
    """
    Read binary class labels from metadata.

    The training notebook may save labels with different key names.
    This function supports several common metadata formats.
    """

    candidate_keys = [
        "binary_labels",
        "class_names",
        "label_names",
        "labels",
        "idx_to_label",
        "id_to_label",
    ]

    labels = None

    for key in candidate_keys:
        if key not in metadata:
            continue

        value = metadata.get(key)

        if isinstance(value, list):
            labels = [str(x) for x in value]
            break

        if isinstance(value, dict):
            # Handles {"0": "non_filler", "1": "filler"} or {0: "...", 1: "..."}.
            labels = [
                str(value[k])
                for k in sorted(value.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
            ]
            break

    if not labels:
        labels = ["non_filler", "filler"]

    normalized = [normalize_binary_label(label) for label in labels]

    # Guarantee a valid 2-class binary order.
    if "non_filler" in normalized and "filler" in normalized:
        return normalized

    return ["non_filler", "filler"]


def get_binary_class_indices(binary_labels: List[str]) -> Dict[str, int]:
    normalized = [normalize_binary_label(label) for label in binary_labels]

    filler_idx = normalized.index("filler") if "filler" in normalized else 1
    non_filler_idx = normalized.index("non_filler") if "non_filler" in normalized else 0

    return {
        "filler": filler_idx,
        "non_filler": non_filler_idx,
    }


def wav_to_tensor(audio_path: Path) -> tf.Tensor:
    metadata = get_filler_metadata()
    sample_rate = int(metadata.get("sample_rate", 16000))

    audio_bytes = tf.io.read_file(str(audio_path))
    audio, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    if int(sr.numpy()) != sample_rate:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rate must be {sample_rate}. Converted audio does not match the expected sample rate."
        )

    audio = tf.cast(audio, tf.float32)
    audio = audio / (tf.reduce_max(tf.abs(audio)) + 1e-6)
    return audio


def waveform_to_logmel(audio: tf.Tensor) -> tf.Tensor:
    metadata = get_filler_metadata()

    frame_length = int(metadata.get("frame_length", 400))
    frame_step = int(metadata.get("frame_step", 160))
    fft_length = int(metadata.get("fft_length", 512))

    mel_weight_matrix = get_mel_weight_matrix()

    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        pad_end=True
    )

    spectrogram = tf.abs(stft)
    power_spectrogram = tf.square(spectrogram)
    mel_spectrogram = tf.matmul(power_spectrogram, mel_weight_matrix)

    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    log_mel = (log_mel - tf.reduce_mean(log_mel)) / (tf.math.reduce_std(log_mel) + 1e-6)
    log_mel = tf.expand_dims(log_mel, axis=-1)

    return log_mel


def binary_probs_from_logits(logits: tf.Tensor, binary_labels: List[str]) -> np.ndarray:
    """
    Return probabilities with shape [num_windows, 2] in this order:
    [non_filler_probability, filler_probability].

    Supports:
    - 2-logit softmax model
    - 1-logit sigmoid model
    """

    logits_np = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)

    if logits_np.ndim == 1:
        logits_np = logits_np.reshape(-1, 1)

    if logits_np.shape[-1] == 1:
        filler_prob = tf.sigmoid(logits).numpy().reshape(-1)
        non_filler_prob = 1.0 - filler_prob
        return np.stack([non_filler_prob, filler_prob], axis=1)

    probs = tf.nn.softmax(logits, axis=-1).numpy()
    class_indices = get_binary_class_indices(binary_labels)

    non_filler_prob = probs[:, class_indices["non_filler"]]
    filler_prob = probs[:, class_indices["filler"]]

    return np.stack([non_filler_prob, filler_prob], axis=1)


def analyze_filler_from_wav(audio_path: Path, window_seconds: float = 1.0, hop_seconds: float = 0.5) -> Dict[str, Any]:
    model = get_filler_model()
    metadata = get_filler_metadata()

    binary_labels = get_binary_label_names(metadata)
    sample_rate = int(metadata.get("sample_rate", 16000))

    audio = wav_to_tensor(audio_path)

    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)

    total_samples = int(audio.shape[0])
    duration = total_samples / sample_rate

    features = []
    times = []

    start = 0
    while start < total_samples:
        end = start + window_samples
        clip = audio[start:end]

        if tf.shape(clip)[0] < window_samples:
            clip = tf.pad(clip, [[0, window_samples - tf.shape(clip)[0]]])

        log_mel = waveform_to_logmel(clip)
        features.append(log_mel)
        times.append((start / sample_rate, min(end / sample_rate, duration)))

        start += hop_samples

    if not features:
        raise HTTPException(status_code=400, detail="Audio is empty.")

    features = tf.stack(features, axis=0)

    logits = model(features, training=False)
    binary_probs = binary_probs_from_logits(logits, binary_labels)

    timeline = []

    for idx in range(binary_probs.shape[0]):
        start_time, end_time = times[idx]

        non_filler_probability = float(binary_probs[idx, 0])
        filler_probability = float(binary_probs[idx, 1])

        if filler_probability >= FILLER_DECISION_THRESHOLD:
            prediction_label = "filler"
            confidence = filler_probability
        else:
            prediction_label = "non_filler"
            confidence = non_filler_probability

        timeline.append({
            "start_time": float(start_time),
            "end_time": float(end_time),
            "prediction_label": prediction_label,
            "confidence": float(confidence),
            "interview_summary": prediction_label,
            "binary_prediction": prediction_label,
            "filler_probability": filler_probability,
            "non_filler_probability": non_filler_probability,
        })

    summary_classes = ["filler", "non_filler"]

    summary_counts = {
        cls: sum(1 for item in timeline if item["interview_summary"] == cls)
        for cls in summary_classes
    }

    total_windows = len(timeline)
    filler_windows = summary_counts["filler"]
    non_filler_windows = summary_counts["non_filler"]

    filler_rate = filler_windows / max(total_windows, 1)
    non_filler_rate = non_filler_windows / max(total_windows, 1)

    # Binary model only penalizes detected filler windows.
    # Repetition/audio-noise are no longer available because this is not the 13-class model.
    delivery_score = 100
    delivery_score -= filler_rate * 100
    delivery_score = max(0, min(100, delivery_score))

    return {
        "model_type": "binary_filler_detector",
        "duration_seconds": float(duration),
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "total_windows": total_windows,
        "summary_counts": summary_counts,
        "binary_label_counts": summary_counts,
        "filler_rate": float(filler_rate),
        "non_filler_rate": float(non_filler_rate),
        "repetition_rate": 0.0,
        "audio_noise_rate": 0.0,
        "delivery_score": round(float(delivery_score), 2),
        "deployment_thresholds": {
            "filler_decision_threshold": FILLER_DECISION_THRESHOLD
        },
        "timeline": timeline
    }
