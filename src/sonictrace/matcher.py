
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median

from src.sonictrace.fingerprint import AudioFingerprint


@dataclass
class FingerprintMatch:
    hash_key: tuple[int, int, int]
    time_a: float
    time_b: float
    offset: float


@dataclass
class OffsetCandidate:
    offset: float
    matches: list[FingerprintMatch]
    match_count: int
    vote_count: int


@dataclass
class MatchResult:
    matches: list[FingerprintMatch]
    best_offset: float | None
    best_match_count: int
    offset_histogram: dict[int, int]
    candidates: list[OffsetCandidate]


def match_fingerprint_candidates(
    fingerprints_a: list[AudioFingerprint],
    fingerprints_b: list[AudioFingerprint],
    offset_tolerance: float = 0.10,
    max_seed_pairs_per_hash: int = 25,
    max_candidates: int = 30,
    candidate_separation: float = 0.25,
) -> MatchResult:

    if offset_tolerance <= 0:
        raise ValueError("Offset tolerance must be positive.")

    if max_seed_pairs_per_hash <= 0 or max_candidates <= 0:
        raise ValueError("Candidate limits must be positive.")

    if candidate_separation < 0:
        raise ValueError("Candidate separation cannot be negative.")

    index_a = defaultdict(list)
    index_b = defaultdict(list)

    for fingerprint in fingerprints_a:
        index_a[fingerprint.hash_key].append(fingerprint.anchor_time)

    for fingerprint in fingerprints_b:
        index_b[fingerprint.hash_key].append(fingerprint.anchor_time)

    for times in index_b.values():
        times.sort()

    common_hashes = index_a.keys() & index_b.keys()
    offset_votes = Counter()

    # Generate offset hypotheses.
    for hash_key in common_hashes:
        times_a = index_a[hash_key]
        times_b = index_b[hash_key]

        pair_count = len(times_a) * len(times_b)

        if pair_count > max_seed_pairs_per_hash:
            continue

        for time_a in times_a:
            for time_b in times_b:
                offset = time_b - time_a
                offset_bin = round(offset / offset_tolerance)
                offset_votes[offset_bin] += 1

    if not offset_votes:
        return MatchResult(
            matches=[],
            best_offset=None,
            best_match_count=0,
            offset_histogram={},
            candidates=[],
        )

    candidates = []

    # Examine more than just the single strongest offset.
    for offset_bin, vote_count in offset_votes.most_common(200):

        candidate_offset = offset_bin * offset_tolerance

        verified_matches = []
        used_b_times = defaultdict(set)

        for hash_key in common_hashes:
            times_b = index_b[hash_key]

            for time_a in index_a[hash_key]:
                expected_time_b = time_a + candidate_offset

                position = bisect_left(times_b, expected_time_b)

                possible_positions = [position - 1, position]

                best_position = None
                smallest_difference = float("inf")

                for pos in possible_positions:
                    if pos < 0 or pos >= len(times_b):
                        continue

                    if pos in used_b_times[hash_key]:
                        continue

                    difference = abs(
                        times_b[pos] - expected_time_b
                    )

                    if difference < smallest_difference:
                        smallest_difference = difference
                        best_position = pos

                if (
                    best_position is not None
                    and smallest_difference <= offset_tolerance
                ):
                    time_b = times_b[best_position]

                    used_b_times[hash_key].add(best_position)

                    verified_matches.append(
                        FingerprintMatch(
                            hash_key=hash_key,
                            time_a=time_a,
                            time_b=time_b,
                            offset=time_b - time_a,
                        )
                    )

        if not verified_matches:
            continue

        verified_offset = median(
            match.offset for match in verified_matches
        )

        candidates.append(
            OffsetCandidate(
                offset=float(verified_offset),
                matches=verified_matches,
                match_count=len(verified_matches),
                vote_count=vote_count,
            )
        )

    # Sort by fingerprint support.
    candidates.sort(
        key=lambda candidate: candidate.match_count,
        reverse=True,
    )

    # Remove near-duplicate offset hypotheses.
    distinct_candidates = []

    for candidate in candidates:
        duplicate = any(
            abs(candidate.offset - existing.offset)
            < candidate_separation
            for existing in distinct_candidates
        )

        if not duplicate:
            distinct_candidates.append(candidate)

        if len(distinct_candidates) >= max_candidates:
            break

    best = distinct_candidates[0] if distinct_candidates else None

    return MatchResult(
        matches=best.matches if best else [],
        best_offset=best.offset if best else None,
        best_match_count=best.match_count if best else 0,
        offset_histogram=dict(offset_votes),
        candidates=distinct_candidates,
    )


def match_fingerprints(
    fingerprints_a: list[AudioFingerprint],
    fingerprints_b: list[AudioFingerprint],
    offset_tolerance: float = 0.10,
    max_seed_pairs_per_hash: int = 25,
) -> MatchResult:

    return match_fingerprint_candidates(
        fingerprints_a=fingerprints_a,
        fingerprints_b=fingerprints_b,
        offset_tolerance=offset_tolerance,
        max_seed_pairs_per_hash=max_seed_pairs_per_hash,
    )
