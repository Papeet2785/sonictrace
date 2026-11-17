
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class AudioPeak:
    time_index: int
    frequency_index: int
    time_seconds: float
    frequency_hz: float
    magnitude: float


@dataclass
class AudioFingerprint:
    hash_key: tuple[int, int, int]
    anchor_time: float
    anchor_time_index: int


def detect_peaks(
    mel_spectrogram: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    neighborhood_size: tuple[int, int] = (7, 11),
    threshold_db: float = -35.0,
    min_time_distance_frames: int = 5,
    min_frequency_distance_bins: int = 3,
    max_peaks_per_second: int = 10,
) -> list[AudioPeak]:
    """
    Detect prominent spectral peaks using local maxima,
    greedy two-dimensional suppression, and a maximum
    number of peaks per second.
    """

    # 1. Validate inputs
    if mel_spectrogram.ndim != 2:
        raise ValueError("Expected a 2D Mel spectrogram.")

    if mel_spectrogram.size == 0:
        raise ValueError("Mel spectrogram is empty.")

    if not np.all(np.isfinite(mel_spectrogram)):
        raise ValueError("Mel spectrogram contains invalid values.")

    if sample_rate <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("Audio parameters must be positive.")

    if min_time_distance_frames < 0:
        raise ValueError("Time distance cannot be negative.")

    if min_frequency_distance_bins < 0:
        raise ValueError("Frequency distance cannot be negative.")

    if max_peaks_per_second <= 0:
        raise ValueError("Maximum peaks per second must be positive.")

    # 2. Identify local maxima
    local_max = maximum_filter(
        mel_spectrogram,
        size=neighborhood_size,
    )

    peak_mask = (
        (mel_spectrogram == local_max)
        & (mel_spectrogram >= threshold_db)
    )

    candidates = np.argwhere(peak_mask)

    if len(candidates) == 0:
        return []

    # 3. Sort candidates by magnitude, strongest first
    magnitudes = mel_spectrogram[
        candidates[:, 0],
        candidates[:, 1],
    ]

    order = np.argsort(magnitudes)[::-1]
    candidates = candidates[order]

    # 4. Convert Mel bins to their center frequencies
    mel_frequencies = librosa.mel_frequencies(
        n_mels=mel_spectrogram.shape[0] + 2,
        fmin=50.0,
        fmax=sample_rate / 2,
    )[1:-1]

    # 5. Initialize peak suppression
    suppressed = np.zeros(
        mel_spectrogram.shape,
        dtype=bool,
    )

    peaks = []
    peaks_per_second = {}

    freq_radius = min_frequency_distance_bins
    time_radius = min_time_distance_frames

    # 6. Select the strongest valid peaks
    for freq_idx, time_idx in candidates:
        freq_idx = int(freq_idx)
        time_idx = int(time_idx)

        if suppressed[freq_idx, time_idx]:
            continue

        time_seconds = time_idx * hop_length / sample_rate
        second_bucket = int(time_seconds)

        if peaks_per_second.get(second_bucket, 0) >= max_peaks_per_second:
            continue

        frequency_hz = float(mel_frequencies[freq_idx])

        magnitude = float(
            mel_spectrogram[freq_idx, time_idx]
        )

        peaks.append(
            AudioPeak(
                time_index=time_idx,
                frequency_index=freq_idx,
                time_seconds=float(time_seconds),
                frequency_hz=frequency_hz,
                magnitude=magnitude,
            )
        )

        peaks_per_second[second_bucket] = (
            peaks_per_second.get(second_bucket, 0) + 1
        )

        # Suppress nearby candidates
        f_start = max(0, freq_idx - freq_radius)
        f_end = min(
            mel_spectrogram.shape[0],
            freq_idx + freq_radius + 1,
        )

        t_start = max(0, time_idx - time_radius)
        t_end = min(
            mel_spectrogram.shape[1],
            time_idx + time_radius + 1,
        )

        suppressed[f_start:f_end, t_start:t_end] = True

    # 7. Return peaks in chronological order
    peaks.sort(key=lambda peak: peak.time_seconds)

    return peaks


def generate_fingerprints(
    peaks: list[AudioPeak],
    min_delta_seconds: float = 0.10,
    max_delta_seconds: float = 1.50,
    max_targets: int = 5,
    time_quantization_frames: int = 2,
    hop_length: int = 512,
    sample_rate: int = 22050,
) -> list[AudioFingerprint]:
    """
    Generate fingerprints by pairing each anchor peak
    with nearby peaks in a future time window.

    Hash structure:
        (anchor_frequency, target_frequency, time_delta)
    """

    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("Sample rate and hop length must be positive.")

    if min_delta_seconds < 0 or max_delta_seconds <= min_delta_seconds:
        raise ValueError("Invalid target time window.")

    if max_targets <= 0 or time_quantization_frames <= 0:
        raise ValueError("Target count and quantization must be positive.")

    peaks = sorted(
        peaks,
        key=lambda peak: peak.time_index,
    )

    fingerprints = []

    for i, anchor in enumerate(peaks):
        targets = []

        # Find peaks occurring after the anchor
        for target in peaks[i + 1:]:
            delta_seconds = (
                target.time_seconds - anchor.time_seconds
            )

            if delta_seconds > max_delta_seconds:
                break

            if delta_seconds < min_delta_seconds:
                continue

            targets.append(target)

        # Keep the strongest target peaks
        targets.sort(
            key=lambda peak: peak.magnitude,
            reverse=True,
        )

        for target in targets[:max_targets]:
            delta_frames = (
                target.time_index - anchor.time_index
            )

            # Quantize the time difference
            quantized_delta = int(
                round(delta_frames / time_quantization_frames)
                * time_quantization_frames
            )

            hash_key = (
                anchor.frequency_index,
                target.frequency_index,
                quantized_delta,
            )

            fingerprints.append(
                AudioFingerprint(
                    hash_key=hash_key,
                    anchor_time=anchor.time_seconds,
                    anchor_time_index=anchor.time_index,
                )
            )

    return fingerprints
