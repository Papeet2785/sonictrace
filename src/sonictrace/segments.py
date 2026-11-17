from dataclasses import dataclass
from statistics import median

from src.sonictrace.matcher import FingerprintMatch


@dataclass
class MatchingSegment:
    start_a: float
    end_a: float
    start_b: float
    end_b: float
    duration: float
    match_count: int
    offset: float


def detect_matching_segments(
    matches: list[FingerprintMatch],
    max_gap_seconds: float = 1.5,
    max_offset_deviation: float = 0.25,
    min_duration_seconds: float = 5.0,
    min_matches: int = 10,
) -> list[MatchingSegment]:
    """
    Group timestamp matches into continuous matching segments.

    A new segment starts when:
    - The gap between consecutive matches is too large.
    - The timestamp offset changes significantly.

    Duplicate anchor timestamps are removed before scoring.
    """

    if max_gap_seconds <= 0:
        raise ValueError("Maximum gap must be positive.")

    if max_offset_deviation <= 0:
        raise ValueError("Offset deviation must be positive.")

    if min_duration_seconds < 0 or min_matches <= 0:
        raise ValueError("Invalid minimum segment requirements.")

    if not matches:
        return []

    # Sort matches by their position in Audio A.
    ordered = sorted(matches, key=lambda match: match.time_a)

    # Remove duplicate anchor timestamps.
    unique_matches = []
    seen_times = set()

    for match in ordered:
        timestamp_key = round(match.time_a, 2)

        if timestamp_key in seen_times:
            continue

        seen_times.add(timestamp_key)
        unique_matches.append(match)

    if not unique_matches:
        return []

    # Build groups of temporally consistent matches.
    groups = []
    current_group = [unique_matches[0]]

    for match in unique_matches[1:]:
        previous = current_group[-1]

        time_gap = match.time_a - previous.time_a

        current_offset = median(
            item.offset for item in current_group
        )

        offset_difference = abs(
            match.offset - current_offset
        )

        if (
            time_gap > max_gap_seconds
            or offset_difference > max_offset_deviation
        ):
            groups.append(current_group)
            current_group = [match]
        else:
            current_group.append(match)

    groups.append(current_group)

    # Convert sufficiently large groups into segments.
    segments = []

    for group in groups:
        if len(group) < min_matches:
            continue

        start_a = group[0].time_a
        end_a = group[-1].time_a
        duration = end_a - start_a

        if duration < min_duration_seconds:
            continue

        segment_offset = median(
            match.offset for match in group
        )

        start_b = start_a + segment_offset
        end_b = end_a + segment_offset

        segments.append(
            MatchingSegment(
                start_a=start_a,
                end_a=end_a,
                start_b=start_b,
                end_b=end_b,
                duration=duration,
                match_count=len(group),
                offset=segment_offset,
            )
        )

    return sorted(
        segments,
        key=lambda segment: segment.duration,
        reverse=True,
    )
