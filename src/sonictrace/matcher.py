from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from math import sqrt

from src.sonictrace.fingerprint import AudioFingerprint


@dataclass
class FingerprintMatch:
    hash_key: tuple[int, int, int]
    time_a: float
    time_b: float
    offset: float


@dataclass
class SequenceChain:
    """A chronologically consistent run of matching fingerprint anchors."""

    matches: list[FingerprintMatch]
    start_a: float
    end_a: float
    start_b: float
    end_b: float
    duration: float
    median_offset: float
    anchor_count: int
    distinct_hash_count: int
    offset_mad: float


@dataclass
class OffsetCandidate:
    offset: float
    matches: list[FingerprintMatch]
    match_count: int
    vote_count: int
    weighted_support: float = 0.0
    distinct_hash_count: int = 0
    sequence_chains: list[SequenceChain] = field(default_factory=list)
    longest_chain_duration: float = 0.0
    total_chain_duration: float = 0.0
    sequence_anchor_count: int = 0


@dataclass
class MatchResult:
    matches: list[FingerprintMatch]
    best_offset: float | None
    best_match_count: int
    offset_histogram: dict[int, int]
    candidates: list[OffsetCandidate]


def build_sequence_chains(
    matches: list[FingerprintMatch],
    max_gap_seconds: float = 1.5,
    max_offset_deviation: float = 0.25,
    min_duration_seconds: float = 5.0,
    min_anchor_count: int = 10,
) -> list[SequenceChain]:
    """
    Group matches into chronological chains.

    Matches at effectively the same Audio A timestamp are treated as one
    anchor event. A chain continues only when both recordings advance in
    time, the gap is bounded, and the event's offset remains close to the
    chain's running median offset.
    """
    if max_gap_seconds <= 0:
        raise ValueError("Maximum gap must be positive.")
    if max_offset_deviation < 0:
        raise ValueError("Maximum offset deviation cannot be negative.")
    if min_duration_seconds < 0 or min_anchor_count < 1:
        raise ValueError("Minimum chain requirements are invalid.")

    if not matches:
        return []

    grouped = defaultdict(list)
    for match in matches:
        grouped[round(match.time_a, 2)].append(match)

    events = []
    for time_a_key, group in grouped.items():
        representative_time_a = median(m.time_a for m in group)
        representative_time_b = median(m.time_b for m in group)
        offsets = [m.offset for m in group]
        events.append({
            "time_a": representative_time_a,
            "time_b": representative_time_b,
            "offset": median(offsets),
            "matches": group,
        })

    events.sort(key=lambda event: (event["time_a"], event["time_b"]))

    chains = []
    current = []

    def finish_chain():
        if not current:
            return

        chain_matches = [
            match
            for event in current
            for match in event["matches"]
        ]
        start_a = current[0]["time_a"]
        end_a = current[-1]["time_a"]
        start_b = current[0]["time_b"]
        end_b = current[-1]["time_b"]
        duration = end_a - start_a
        offsets = [event["offset"] for event in current]
        median_offset = median(offsets)
        offset_mad = median(
            abs(offset - median_offset)
            for offset in offsets
        )
        anchor_count = len(current)
        distinct_hash_count = len({
            match.hash_key for match in chain_matches
        })

        if (
            duration >= min_duration_seconds
            and anchor_count >= min_anchor_count
        ):
            chains.append(
                SequenceChain(
                    matches=chain_matches,
                    start_a=start_a,
                    end_a=end_a,
                    start_b=start_b,
                    end_b=end_b,
                    duration=duration,
                    median_offset=median_offset,
                    anchor_count=anchor_count,
                    distinct_hash_count=distinct_hash_count,
                    offset_mad=offset_mad,
                )
            )

    for event in events:
        if not current:
            current.append(event)
            continue

        previous = current[-1]
        delta_a = event["time_a"] - previous["time_a"]
        delta_b = event["time_b"] - previous["time_b"]
        running_offset = median(
            item["offset"] for item in current
        )

        continues = (
            delta_a > 0.02
            and delta_b > 0.02
            and delta_a <= max_gap_seconds
            and abs(event["offset"] - running_offset)
            <= max_offset_deviation
        )

        if continues:
            current.append(event)
        else:
            finish_chain()
            current = [event]

    finish_chain()

    chains.sort(
        key=lambda chain: (
            chain.duration,
            chain.anchor_count,
            chain.distinct_hash_count,
        ),
        reverse=True,
    )
    return chains


def _attach_sequence_metrics(candidate: OffsetCandidate) -> None:
    chains = build_sequence_chains(candidate.matches)
    candidate.sequence_chains = chains
    candidate.longest_chain_duration = (
        chains[0].duration if chains else 0.0
    )
    candidate.total_chain_duration = sum(
        chain.duration for chain in chains
    )
    candidate.sequence_anchor_count = sum(
        chain.anchor_count for chain in chains
    )


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

    for times in index_a.values():
        times.sort()
    for times in index_b.values():
        times.sort()

    common_hashes = index_a.keys() & index_b.keys()

    raw_offset_votes = Counter()
    weighted_offset_votes = defaultdict(float)
    hash_weights = {}

    # Calculate rarity weights for hashes shared by both recordings.
    for hash_key in common_hashes:
        count_a = len(index_a[hash_key])
        count_b = len(index_b[hash_key])
        pair_count = count_a * count_b

        # Avoid hashes that generate too many ambiguous seed pairs.
        if pair_count > max_seed_pairs_per_hash:
            continue

        rarity_weight = 1.0 / sqrt(count_a * count_b)
        hash_weights[hash_key] = rarity_weight

        for time_a in index_a[hash_key]:
            for time_b in index_b[hash_key]:
                offset = time_b - time_a
                offset_bin = round(offset / offset_tolerance)
                raw_offset_votes[offset_bin] += 1
                weighted_offset_votes[offset_bin] += rarity_weight

    if not weighted_offset_votes:
        return MatchResult(
            matches=[],
            best_offset=None,
            best_match_count=0,
            offset_histogram={},
            candidates=[],
        )

    candidates = []

    # Generate and verify offset hypotheses.
    for offset_bin, _weighted_votes in sorted(
        weighted_offset_votes.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:200]:

        candidate_offset = offset_bin * offset_tolerance
        verified_matches = []
        used_b_times = defaultdict(set)

        for hash_key in common_hashes:
            if hash_key not in hash_weights:
                continue

            times_a = index_a[hash_key]
            times_b = index_b[hash_key]

            for time_a in times_a:
                expected_time_b = time_a + candidate_offset
                position = bisect_left(times_b, expected_time_b)

                possible_positions = (position - 1, position)
                best_position = None
                smallest_difference = float("inf")

                for pos in possible_positions:
                    if pos < 0 or pos >= len(times_b):
                        continue
                    if pos in used_b_times[hash_key]:
                        continue

                    difference = abs(times_b[pos] - expected_time_b)
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
        weighted_support = sum(
            hash_weights[match.hash_key]
            for match in verified_matches
        )
        distinct_hash_count = len({
            match.hash_key for match in verified_matches
        })

        candidate = OffsetCandidate(
            offset=float(verified_offset),
            matches=verified_matches,
            match_count=len(verified_matches),
            vote_count=raw_offset_votes[offset_bin],
            weighted_support=weighted_support,
            distinct_hash_count=distinct_hash_count,
        )
        _attach_sequence_metrics(candidate)
        candidates.append(candidate)

    # Prefer coherent chronological sequences before raw fingerprint volume.
    candidates.sort(
        key=lambda candidate: (
            candidate.longest_chain_duration,
            candidate.total_chain_duration,
            candidate.sequence_anchor_count,
            candidate.weighted_support,
            candidate.distinct_hash_count,
            candidate.match_count,
        ),
        reverse=True,
    )

    distinct_candidates = []

    for candidate in candidates:
        duplicate = any(
            abs(candidate.offset - existing.offset) < candidate_separation
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
        offset_histogram=dict(raw_offset_votes),
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
