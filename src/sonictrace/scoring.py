from dataclasses import dataclass

from src.sonictrace.segments import MatchingSegment


@dataclass
class CoverageResult:
    matched_duration_a: float
    matched_duration_b: float
    coverage_a_percent: float
    coverage_b_percent: float
    segment_count: int


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping time intervals."""

    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def calculate_coverage(
    segments: list[MatchingSegment],
    duration_a: float,
    duration_b: float,
) -> CoverageResult:
    """
    Calculate the proportion of each recording covered
    by detected matching segments.

    These are fingerprint coverage measurements,
    not definitive music similarity percentages.
    """

    if duration_a <= 0 or duration_b <= 0:
        raise ValueError("Audio durations must be positive.")

    intervals_a = []
    intervals_b = []

    for segment in segments:
        start_a = max(0.0, segment.start_a)
        end_a = min(duration_a, segment.end_a)

        start_b = max(0.0, segment.start_b)
        end_b = min(duration_b, segment.end_b)

        if end_a > start_a:
            intervals_a.append((start_a, end_a))

        if end_b > start_b:
            intervals_b.append((start_b, end_b))

    merged_a = merge_intervals(intervals_a)
    merged_b = merge_intervals(intervals_b)

    matched_a = sum(
        end - start for start, end in merged_a
    )

    matched_b = sum(
        end - start for start, end in merged_b
    )

    return CoverageResult(
        matched_duration_a=matched_a,
        matched_duration_b=matched_b,
        coverage_a_percent=matched_a / duration_a * 100,
        coverage_b_percent=matched_b / duration_b * 100,
        segment_count=len(segments),
    )

from src.sonictrace.verification import (
    ChromaInterval,
    MelVerification,
)


@dataclass
class IntervalScore:
    start_a: float
    end_a: float
    start_b: float
    end_b: float
    duration: float
    chroma_score: float
    mel_score: float
    combined_score: float
    evidence: str


@dataclass
class SimilarityReport:
    interval_scores: list[IntervalScore]
    weighted_similarity: float
    candidate_duration_a: float
    candidate_duration_b: float
    candidate_coverage_a: float
    candidate_coverage_b: float
    strong_duration_a: float
    strong_duration_b: float
    strong_coverage_a: float
    strong_coverage_b: float
    verified_interval_count: int
    strong_interval_count: int


def calculate_similarity_report(
    chroma_intervals: list[ChromaInterval],
    mel_results: list[MelVerification],
    duration_a: float,
    duration_b: float,
    chroma_weight: float = 0.40,
    mel_weight: float = 0.60,
    strong_chroma_threshold: float = 0.80,
    strong_mel_threshold: float = 0.75,
    minimum_strong_duration: float = 5.0,
) -> SimilarityReport:
    """
    Combine chroma and Mel verification into an
    interval-level similarity report.

    The combined score is a heuristic, not a calibrated
    probability or plagiarism percentage.
    """

    if duration_a <= 0 or duration_b <= 0:
        raise ValueError("Audio durations must be positive.")

    if chroma_weight < 0 or mel_weight < 0:
        raise ValueError("Weights cannot be negative.")

    if chroma_weight + mel_weight == 0:
        raise ValueError("At least one weight must be positive.")

    if not 0 <= strong_chroma_threshold <= 1:
        raise ValueError("Invalid chroma threshold.")

    if not 0 <= strong_mel_threshold <= 1:
        raise ValueError("Invalid Mel threshold.")

    if minimum_strong_duration < 0:
        raise ValueError("Minimum duration cannot be negative.")

    total_weight = chroma_weight + mel_weight

    chroma_weight /= total_weight
    mel_weight /= total_weight

    # Associate Mel results with their original chroma intervals.
    mel_by_interval = {
        id(result.interval): result
        for result in mel_results
    }

    interval_scores = []

    for interval in chroma_intervals:
        mel_result = mel_by_interval.get(id(interval))

        # An interval without Mel verification is not scored.
        if mel_result is None:
            continue

        chroma_score = interval.mean_similarity
        mel_score = mel_result.mean_similarity

        combined_score = (
            chroma_weight * chroma_score
            + mel_weight * mel_score
        )

        if (
            chroma_score >= strong_chroma_threshold
            and mel_score >= strong_mel_threshold
            and interval.duration >= minimum_strong_duration
        ):
            evidence = "Strong"
        elif combined_score >= 0.70:
            evidence = "Moderate"
        else:
            evidence = "Weak"

        interval_scores.append(
            IntervalScore(
                start_a=interval.start_a,
                end_a=interval.end_a,
                start_b=interval.start_b,
                end_b=interval.end_b,
                duration=interval.duration,
                chroma_score=chroma_score,
                mel_score=mel_score,
                combined_score=combined_score,
                evidence=evidence,
            )
        )

    # Duration-weighted score across verified intervals.
    total_verified_duration = sum(
        item.duration for item in interval_scores
    )

    if total_verified_duration:
        weighted_similarity = sum(
            item.combined_score * item.duration
            for item in interval_scores
        ) / total_verified_duration
    else:
        weighted_similarity = 0.0

    # Calculate candidate coverage without double-counting overlaps.
    candidate_a = merge_intervals([
        (item.start_a, item.end_a)
        for item in chroma_intervals
    ])

    candidate_b = merge_intervals([
        (item.start_b, item.end_b)
        for item in chroma_intervals
    ])

    candidate_duration_a = sum(
        end - start for start, end in candidate_a
    )

    candidate_duration_b = sum(
        end - start for start, end in candidate_b
    )

    # Strong evidence requires both measurements to pass.
    strong_intervals = [
        item for item in interval_scores
        if item.evidence == "Strong"
    ]

    strong_a = merge_intervals([
        (item.start_a, item.end_a)
        for item in strong_intervals
    ])

    strong_b = merge_intervals([
        (item.start_b, item.end_b)
        for item in strong_intervals
    ])

    strong_duration_a = sum(
        end - start for start, end in strong_a
    )

    strong_duration_b = sum(
        end - start for start, end in strong_b
    )

    return SimilarityReport(
        interval_scores=interval_scores,
        weighted_similarity=weighted_similarity,
        candidate_duration_a=candidate_duration_a,
        candidate_duration_b=candidate_duration_b,
        candidate_coverage_a=(
            candidate_duration_a / duration_a * 100
        ),
        candidate_coverage_b=(
            candidate_duration_b / duration_b * 100
        ),
        strong_duration_a=strong_duration_a,
        strong_duration_b=strong_duration_b,
        strong_coverage_a=strong_duration_a / duration_a * 100,
        strong_coverage_b=strong_duration_b / duration_b * 100,
        verified_interval_count=len(interval_scores),
        strong_interval_count=len(strong_intervals),
    )
