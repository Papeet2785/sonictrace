from dataclasses import dataclass

import numpy as np

from src.sonictrace.segments import MatchingSegment
from src.sonictrace.features import AudioFeatures
from scipy.ndimage import median_filter


@dataclass
class SegmentVerification:
    segment: MatchingSegment
    mean_similarity: float
    median_similarity: float
    high_similarity_percent: float
    compared_frames: int


def cosine_similarity(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity between corresponding columns."""

    dot_products = np.sum(matrix_a * matrix_b, axis=0)

    norms_a = np.linalg.norm(matrix_a, axis=0)
    norms_b = np.linalg.norm(matrix_b, axis=0)

    denominator = norms_a * norms_b

    return np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products, dtype=float),
        where=denominator > 1e-10,
    )


def verify_segments(
    segments: list[MatchingSegment],
    features_a: AudioFeatures,
    features_b: AudioFeatures,
    frame_tolerance: float = 0.04,
) -> list[SegmentVerification]:
    """
    Verify fingerprint-aligned segments using chroma similarity.

    This measures harmonic similarity, not source separation
    or definitive identification of the underlying song.
    """

    if frame_tolerance < 0:
        raise ValueError("Frame tolerance cannot be negative.")

    times_a = features_a.frame_times
    times_b = features_b.frame_times

    chroma_a = features_a.chroma
    chroma_b = features_b.chroma

    if chroma_a.shape[0] != chroma_b.shape[0]:
        raise ValueError("Chroma dimensions do not match.")

    results = []

    for segment in segments:
        indices_a = np.where(
            (times_a >= segment.start_a)
            & (times_a <= segment.end_a)
        )[0]

        if len(indices_a) == 0:
            continue

        target_times_b = (
            times_a[indices_a] + segment.offset
        )

        indices_b = np.searchsorted(
            times_b,
            target_times_b,
        )

        valid = indices_b < len(times_b)

        indices_a = indices_a[valid]
        indices_b = indices_b[valid]
        target_times_b = target_times_b[valid]

        if len(indices_b) == 0:
            continue

        # Choose the nearest frame in Audio B.
        previous_indices = np.maximum(indices_b - 1, 0)

        previous_errors = np.abs(
            times_b[previous_indices] - target_times_b
        )

        current_errors = np.abs(
            times_b[indices_b] - target_times_b
        )

        use_previous = previous_errors < current_errors

        indices_b = np.where(
            use_previous,
            previous_indices,
            indices_b,
        )

        frame_errors = np.abs(
            times_b[indices_b] - target_times_b
        )

        valid = frame_errors <= frame_tolerance

        indices_a = indices_a[valid]
        indices_b = indices_b[valid]

        if len(indices_a) == 0:
            continue

        similarities = cosine_similarity(
            chroma_a[:, indices_a],
            chroma_b[:, indices_b],
        )

        results.append(
            SegmentVerification(
                segment=segment,
                mean_similarity=float(np.mean(similarities)),
                median_similarity=float(np.median(similarities)),
                high_similarity_percent=float(
                    np.mean(similarities >= 0.80) * 100
                ),
                compared_frames=len(similarities),
            )
        )

    return results

@dataclass
class ChromaInterval:
    start_a: float
    end_a: float
    start_b: float
    end_b: float
    duration: float
    mean_similarity: float


def detect_chroma_intervals(
    features_a: AudioFeatures,
    features_b: AudioFeatures,
    offset: float,
    similarity_threshold: float = 0.80,
    min_duration_seconds: float = 2.0,
    max_gap_seconds: float = 0.5,
    smoothing_frames: int = 9,
) -> list[ChromaInterval]:
    """
    Scan the entire recordings for harmonically similar
    passages at a known timestamp offset.

    Offset = time in B - time in A.
    """

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1.")

    if min_duration_seconds <= 0 or max_gap_seconds < 0:
        raise ValueError("Invalid duration or gap.")

    if smoothing_frames <= 0:
        raise ValueError("Smoothing window must be positive.")

    times_a = features_a.frame_times
    times_b = features_b.frame_times

    # Map every Audio A frame to its corresponding Audio B time.
    target_times_b = times_a + offset

    valid = (
        (target_times_b >= times_b[0])
        & (target_times_b <= times_b[-1])
    )

    indices_a = np.where(valid)[0]
    target_times_b = target_times_b[valid]

    if len(indices_a) == 0:
        return []

    # Find the nearest Audio B frame.
    indices_b = np.searchsorted(times_b, target_times_b)

    indices_b = np.clip(indices_b, 0, len(times_b) - 1)

    previous_indices = np.maximum(indices_b - 1, 0)

    use_previous = (
        np.abs(times_b[previous_indices] - target_times_b)
        < np.abs(times_b[indices_b] - target_times_b)
    )

    indices_b = np.where(
        use_previous,
        previous_indices,
        indices_b,
    )

    # Calculate frame-by-frame chroma similarity.
    similarities = cosine_similarity(
        features_a.chroma[:, indices_a],
        features_b.chroma[:, indices_b],
    )

    # Smooth short-lived fluctuations.
    smoothed = median_filter(
        similarities,
        size=smoothing_frames,
        mode="nearest",
    )

    matching = smoothed >= similarity_threshold

    # Fill small gaps, such as brief dialogue interference.
    if len(matching) > 1 and max_gap_seconds > 0:
        frame_interval = float(np.median(np.diff(times_a)))
        max_gap_frames = int(max_gap_seconds / frame_interval)

        if max_gap_frames > 0:
            false_indices = np.where(~matching)[0]

            if len(false_indices):
                gap_starts = false_indices[
                    np.r_[True, np.diff(false_indices) > 1]
                ]

                gap_ends = false_indices[
                    np.r_[np.diff(false_indices) > 1, True]
                ]

                for start, end in zip(gap_starts, gap_ends):
                    gap_length = end - start + 1

                    if (
                        gap_length <= max_gap_frames
                        and start > 0
                        and end < len(matching) - 1
                        and matching[start - 1]
                        and matching[end + 1]
                    ):
                        matching[start:end + 1] = True

    # Extract continuous matching intervals.
    transitions = np.diff(
        np.r_[False, matching, False].astype(int)
    )

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    intervals = []

    for start, end in zip(starts, ends):
        if end <= start:
            continue

        start_a = float(times_a[indices_a[start]])
        end_a = float(times_a[indices_a[end - 1]])

        duration = end_a - start_a

        if duration < min_duration_seconds:
            continue

        interval_similarities = smoothed[start:end]

        start_b = start_a + offset
        end_b = end_a + offset

        intervals.append(
            ChromaInterval(
                start_a=start_a,
                end_a=end_a,
                start_b=start_b,
                end_b=end_b,
                duration=duration,
                mean_similarity=float(
                    np.mean(interval_similarities)
                ),
            )
        )

    return sorted(
        intervals,
        key=lambda interval: interval.duration,
        reverse=True,
    )

@dataclass
class MelVerification:
    interval: ChromaInterval
    mean_similarity: float
    median_similarity: float
    high_similarity_percent: float
    compared_frames: int


def verify_mel_intervals(
    intervals: list[ChromaInterval],
    features_a: AudioFeatures,
    features_b: AudioFeatures,
    frame_tolerance: float = 0.04,
) -> list[MelVerification]:
    """
    Compare aligned Mel spectrogram frames within
    candidate matching intervals.

    Mel values are converted from dB to relative power
    before cosine similarity is calculated.
    """

    if frame_tolerance < 0:
        raise ValueError("Frame tolerance cannot be negative.")

    mel_a = 10 ** (features_a.mel_spectrogram / 10.0)
    mel_b = 10 ** (features_b.mel_spectrogram / 10.0)

    times_a = features_a.frame_times
    times_b = features_b.frame_times

    results = []

    for interval in intervals:
        indices_a = np.where(
            (times_a >= interval.start_a)
            & (times_a <= interval.end_a)
        )[0]

        if len(indices_a) == 0:
            continue

        target_times_b = times_a[indices_a] + (
            interval.start_b - interval.start_a
        )

        indices_b = np.searchsorted(
            times_b,
            target_times_b,
        )

        valid = indices_b < len(times_b)

        indices_a = indices_a[valid]
        indices_b = indices_b[valid]
        target_times_b = target_times_b[valid]

        if len(indices_b) == 0:
            continue

        previous_indices = np.maximum(indices_b - 1, 0)

        previous_errors = np.abs(
            times_b[previous_indices] - target_times_b
        )

        current_errors = np.abs(
            times_b[indices_b] - target_times_b
        )

        indices_b = np.where(
            previous_errors < current_errors,
            previous_indices,
            indices_b,
        )

        errors = np.abs(
            times_b[indices_b] - target_times_b
        )

        valid = errors <= frame_tolerance

        indices_a = indices_a[valid]
        indices_b = indices_b[valid]

        if len(indices_a) == 0:
            continue

        similarities = cosine_similarity(
            mel_a[:, indices_a],
            mel_b[:, indices_b],
        )

        results.append(
            MelVerification(
                interval=interval,
                mean_similarity=float(np.mean(similarities)),
                median_similarity=float(np.median(similarities)),
                high_similarity_percent=float(
                    np.mean(similarities >= 0.80) * 100
                ),
                compared_frames=len(similarities),
            )
        )

    return results
