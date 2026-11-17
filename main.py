
import argparse
import math
from pathlib import Path

from src.sonictrace.audio_loader import load_audio
from src.sonictrace.preprocessing import preprocess_audio
from src.sonictrace.features import extract_features
from src.sonictrace.fingerprint import detect_peaks, generate_fingerprints
from src.sonictrace.matcher import match_fingerprint_candidates
from src.sonictrace.segments import detect_matching_segments
from src.sonictrace.scoring import calculate_coverage, calculate_similarity_report
from src.sonictrace.verification import (
    verify_segments,
    detect_chroma_intervals,
    verify_mel_intervals,
)


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes}:{remaining_seconds:05.2f}"
    
def generate_fingerprint_candidates(features_a, features_b, sample_rate):
    """Generate local, sequence-backed alignment hypotheses."""

    streams = {
        "full-mix": (
            features_a.mel_spectrogram,
            features_b.mel_spectrogram,
        ),
        "harmonic": (
            features_a.harmonic_mel_spectrogram,
            features_b.harmonic_mel_spectrogram,
        ),
    }

    local_candidates = []

    for source, (mel_a, mel_b) in streams.items():
        peaks_a = detect_peaks(mel_a, sample_rate)
        peaks_b = detect_peaks(mel_b, sample_rate)

        fingerprints_a = generate_fingerprints(
            peaks_a,
            sample_rate=sample_rate,
        )
        fingerprints_b = generate_fingerprints(
            peaks_b,
            sample_rate=sample_rate,
        )

        result = match_fingerprint_candidates(
            fingerprints_a,
            fingerprints_b,
            offset_tolerance=0.10,
            max_candidates=30,
        )

        for offset_candidate in result.candidates[:20]:
            # Each chain is an independently supported local alignment.
            # Do not treat the whole offset candidate as one global match.
            for chain in offset_candidate.sequence_chains:
                local_candidates.append({
                    "offset": chain.median_offset,
                    "matches": chain.matches,
                    "match_count": len(chain.matches),
                    "support": offset_candidate.weighted_support,
                    "sources": {source},
                    "chain": chain,
                    "chain_duration": chain.duration,
                    "chain_anchors": chain.anchor_count,
                    "chain_distinct_hashes": chain.distinct_hash_count,
                    "chain_offset_mad": chain.offset_mad,
                })

    # Sort before limiting so long, well-supported chains survive.
    local_candidates.sort(
        key=lambda item: (
            item["chain_duration"],
            item["chain_anchors"],
            item["chain_distinct_hashes"],
            -item["chain_offset_mad"],
        ),
        reverse=True,
    )

    # Full-mix and harmonic streams can produce duplicate chains. Keep a
    # single representative when both A and B ranges substantially overlap.
    deduplicated = []
    for candidate in local_candidates:
        chain = candidate["chain"]
        duplicate = None

        for existing in deduplicated:
            other = existing["chain"]
            overlap_a = overlap_ratio(
                chain.start_a, chain.end_a,
                other.start_a, other.end_a,
            )
            overlap_b = overlap_ratio(
                chain.start_b, chain.end_b,
                other.start_b, other.end_b,
            )
            if overlap_a >= 0.60 and overlap_b >= 0.60:
                duplicate = existing
                break

        if duplicate is None:
            deduplicated.append(candidate)
        else:
            duplicate["sources"].update(candidate["sources"])

    return deduplicated[:80]


def local_fingerprint_support(interval, candidate):
    """
    Count unique fingerprint anchor timestamps inside an interval.

    A match must fall inside both recordings and agree with the
    interval's own offset.
    """

    interval_offset = interval.start_b - interval.start_a
    anchors = set()

    for match in candidate["matches"]:
        if not (
            interval.start_a <= match.time_a <= interval.end_a
        ):
            continue

        if not (
            interval.start_b <= match.time_b <= interval.end_b
        ):
            continue

        if abs(match.offset - interval_offset) > 0.25:
            continue

        anchors.add(round(match.time_a, 2))

    return len(anchors)


def required_local_support(duration):
    """
    Provisional corroboration requirement:
    at least 3 unique anchors and approximately 0.5 anchors per second.
    """

    return max(3, math.ceil(duration * 0.5))


def calculate_temporal_consistency(matches, max_pair_gap=2.0):
    """
    Measure whether matching fingerprint anchors preserve chronological
    order and relative timing in both recordings.

    This is diagnostic evidence for now; it does not reject matches.
    """

    anchor_times = {}

    # Group fingerprints sharing approximately the same A timestamp.
    for match in matches:
        time_a = round(match.time_a, 2)
        anchor_times.setdefault(time_a, []).append(match.time_b)

    points = sorted(
        (
            time_a,
            sum(times_b) / len(times_b),
        )
        for time_a, times_b in anchor_times.items()
    )

    evaluated_pairs = 0
    ordered_pairs = 0
    consistent_pairs = 0

    for previous, current in zip(points, points[1:]):
        time_a_1, time_b_1 = previous
        time_a_2, time_b_2 = current

        delta_a = time_a_2 - time_a_1
        delta_b = time_b_2 - time_b_1

        # Ignore very close anchors and large gaps.
        if delta_a <= 0.02 or delta_a > max_pair_gap:
            continue

        evaluated_pairs += 1

        if delta_b > 0.02:
            ordered_pairs += 1

            if abs(delta_a - delta_b) <= 0.25:
                consistent_pairs += 1

    if evaluated_pairs == 0:
        return {
            "pairs": 0,
            "order_ratio": 0.0,
            "spacing_ratio": 0.0,
        }

    return {
        "pairs": evaluated_pairs,
        "order_ratio": ordered_pairs / evaluated_pairs,
        "spacing_ratio": consistent_pairs / evaluated_pairs,
    }


def evaluate_candidate(
    candidate,
    features_a,
    features_b,
    duration_a,
    duration_b,
):
    """Verify one local sequence chain at its own time offset."""

    chain = candidate["chain"]
    detected_intervals = detect_chroma_intervals(
        features_a=features_a,
        features_b=features_b,
        offset=candidate["offset"],
        similarity_threshold=0.80,
        min_duration_seconds=2.0,
        max_gap_seconds=0.5,
    )

    # The chroma scanner examines the full recordings. Restrict its output
    # to the time ranges independently supported by this fingerprint chain.
    intervals = []
    for interval in detected_intervals:
        intersection_a = max(
            0.0,
            min(interval.end_a, chain.end_a)
            - max(interval.start_a, chain.start_a),
        )
        intersection_b = max(
            0.0,
            min(interval.end_b, chain.end_b)
            - max(interval.start_b, chain.start_b),
        )
        interval_coverage_a = intersection_a / max(interval.duration, 1e-9)
        interval_coverage_b = intersection_b / max(interval.duration, 1e-9)
        interval_offset = interval.start_b - interval.start_a

        if (
            interval_coverage_a >= 0.60
            and interval_coverage_b >= 0.60
            and abs(interval_offset - chain.median_offset) <= 0.30
        ):
            intervals.append(interval)

    mel_results = verify_mel_intervals(
        intervals=intervals,
        features_a=features_a,
        features_b=features_b,
    )

    report = calculate_similarity_report(
        chroma_intervals=intervals,
        mel_results=mel_results,
        duration_a=duration_a,
        duration_b=duration_b,
    )

    candidate_duration = (
        duration_a * report.candidate_coverage_a / 100.0
    )
    evidence_integral = candidate_duration * report.weighted_similarity

    temporal = calculate_temporal_consistency(candidate["matches"])
    ranking = (
        report.strong_duration_a,
        chain.duration,
        chain.anchor_count,
        evidence_integral,
        report.weighted_similarity,
    )

    return {
        "candidate": candidate,
        "intervals": intervals,
        "mel_results": mel_results,
        "report": report,
        "temporal": temporal,
        "ranking": ranking,
    }


def is_strong_interval(interval, mel_result):
    return (
        interval.duration >= 5.0
        and interval.mean_similarity >= 0.80
        and mel_result.mean_similarity >= 0.75
    )


def overlap_ratio(start_a, end_a, start_b, end_b):
    intersection = max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )

    shorter = min(end_a - start_a, end_b - start_b)

    if shorter <= 0:
        return 0.0

    return intersection / shorter


def consolidate_intervals(evaluations):
    """
    Keep only intervals with local fingerprint corroboration.

    Remove duplicate detections only when they overlap in both
    recordings. Preserve repeated occurrences at different positions.
    """

    candidates = []
    rejected_count = 0

    for evaluation in evaluations:
        mel_by_id = {
            id(result.interval): result
            for result in evaluation["mel_results"]
        }

        for interval in evaluation["intervals"]:
            mel_result = mel_by_id.get(id(interval))

            if mel_result is None:
                continue

            support = local_fingerprint_support(
                interval,
                evaluation["candidate"],
            )

            required = required_local_support(interval.duration)

            if support < required:
                rejected_count += 1
                continue

            combined_score = (
                0.40 * interval.mean_similarity
                + 0.60 * mel_result.mean_similarity
            )

            candidates.append({
                "interval": interval,
                "mel_result": mel_result,
                "score": combined_score,
                "offset": evaluation["candidate"]["offset"],
                "fingerprint_support": support,
                "required_support": required,
            })

    # Prefer the strongest corroborated version of overlapping detections.
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["fingerprint_support"],
            item["interval"].duration,
        ),
        reverse=True,
    )

    retained = []

    for candidate in candidates:
        interval = candidate["interval"]
        duplicate = False

        for existing in retained:
            other = existing["interval"]

            overlap_a = overlap_ratio(
                interval.start_a,
                interval.end_a,
                other.start_a,
                other.end_a,
            )

            overlap_b = overlap_ratio(
                interval.start_b,
                interval.end_b,
                other.start_b,
                other.end_b,
            )

            if overlap_a >= 0.50 and overlap_b >= 0.50:
                duplicate = True
                break

        if not duplicate:
            retained.append(candidate)

    retained.sort(
        key=lambda item: item["interval"].start_a
    )

    intervals = [
        item["interval"]
        for item in retained
    ]

    offsets = []

    for item in retained:
        offset = item["offset"]

        if not any(
            abs(offset - existing) < 0.25
            for existing in offsets
        ):
            offsets.append(offset)

    return intervals, offsets, retained, rejected_count


def supported_intervals_for_evaluation(evaluation):
    """Return strong intervals corroborated by this exact local chain."""

    mel_by_id = {
        id(result.interval): result
        for result in evaluation["mel_results"]
    }
    supported = []

    for interval in evaluation["intervals"]:
        mel_result = mel_by_id.get(id(interval))
        if mel_result is None or not is_strong_interval(interval, mel_result):
            continue

        support = local_fingerprint_support(
            interval, evaluation["candidate"]
        )
        required = required_local_support(interval.duration)
        if support < required:
            continue

        score = (
            0.40 * interval.mean_similarity
            + 0.60 * mel_result.mean_similarity
        )
        supported.append({
            "interval": interval,
            "mel_result": mel_result,
            "fingerprint_support": support,
            "score": score,
        })

    return supported


def build_piecewise_paths(
    evaluations,
    max_gap_seconds=90.0,
    overlap_tolerance=1.0,
    max_paths=5,
):
    """
    Stitch verified local chains into order-preserving piecewise paths.

    A transition is allowed when both recordings move forward and the gap
    in each recording is no greater than max_gap_seconds. Offset changes
    are allowed because edits can insert or remove material.

    The function extracts the highest-weight non-overlapping path first,
    then repeats on remaining nodes to expose alternative alignments.
    These paths are alternatives, not evidence to be added together.
    """

    nodes = []
    for evaluation in evaluations:
        evidence = supported_intervals_for_evaluation(evaluation)
        if not evidence:
            continue

        chain = evaluation["candidate"]["chain"]
        evidence_duration = sum(
            item["interval"].duration for item in evidence
        )
        mean_score = sum(
            item["score"] * item["interval"].duration
            for item in evidence
        ) / max(evidence_duration, 1e-9)

        # Duration dominates; similarity and anchor count provide modest
        # tie-breaking support without letting raw fingerprint volume win.
        weight = (
            evidence_duration * mean_score
            + min(chain.anchor_count, 300) * 0.005
        )
        nodes.append({
            "evaluation": evaluation,
            "chain": chain,
            "evidence": evidence,
            "evidence_duration": evidence_duration,
            "mean_score": mean_score,
            "weight": weight,
        })

    remaining = sorted(
        nodes,
        key=lambda node: (
            node["chain"].start_a,
            node["chain"].start_b,
        ),
    )
    paths = []

    def can_follow(previous, current):
        gap_a = current["chain"].start_a - previous["chain"].end_a
        gap_b = current["chain"].start_b - previous["chain"].end_b
        return (
            gap_a >= -overlap_tolerance
            and gap_b >= -overlap_tolerance
            and gap_a <= max_gap_seconds
            and gap_b <= max_gap_seconds
        )

    while remaining and len(paths) < max_paths:
        best_scores = []
        previous_indices = []

        for index, node in enumerate(remaining):
            best_score = node["weight"]
            best_previous = None

            for prior_index in range(index):
                if not can_follow(remaining[prior_index], node):
                    continue

                proposed = best_scores[prior_index] + node["weight"]
                if proposed > best_score:
                    best_score = proposed
                    best_previous = prior_index

            best_scores.append(best_score)
            previous_indices.append(best_previous)

        endpoint = max(
            range(len(best_scores)),
            key=lambda index: best_scores[index],
        )
        selected_indices = []
        cursor = endpoint
        while cursor is not None:
            selected_indices.append(cursor)
            cursor = previous_indices[cursor]
        selected_indices.reverse()

        path_nodes = [remaining[index] for index in selected_indices]
        paths.append({
            "nodes": path_nodes,
            "evaluations": [node["evaluation"] for node in path_nodes],
            "score": sum(node["weight"] for node in path_nodes),
            "evidence_duration": sum(
                node["evidence_duration"] for node in path_nodes
            ),
        })

        selected_ids = {id(node) for node in path_nodes}
        remaining = [
            node for node in remaining
            if id(node) not in selected_ids
        ]

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="SonicTrace Audio Similarity Detector"
    )
    parser.add_argument("--audio-a", required=True)
    parser.add_argument("--audio-b", required=True)
    args = parser.parse_args()

    print("\n========== SONICTRACE ==========")
    print("Preparing audio files...")

    audio_a, sr_a = load_audio(args.audio_a)
    audio_b, sr_b = load_audio(args.audio_b)

    processed_a = preprocess_audio(audio_a, sr_a)
    processed_b = preprocess_audio(audio_b, sr_b)

    print("Extracting audio features...")

    features_a = extract_features(
        processed_a.audio,
        processed_a.sample_rate,
    )
    features_b = extract_features(
        processed_b.audio,
        processed_b.sample_rate,
    )

    print("Generating fingerprint hypotheses...")

    candidates = generate_fingerprint_candidates(
        features_a,
        features_b,
        processed_a.sample_rate,
    )

    print(f"Testing {len(candidates)} local sequence hypotheses...")

    evaluations = []

    for candidate in candidates:
        evaluations.append(
            evaluate_candidate(
                candidate,
                features_a,
                features_b,
                processed_a.processed_duration,
                processed_b.processed_duration,
            )
        )

    # Keep only chains with strong intervals corroborated by fingerprints
    # belonging to those same chains.
    viable = [
        evaluation for evaluation in evaluations
        if supported_intervals_for_evaluation(evaluation)
    ]

    # Fallback is diagnostic only: retain the best chain if no strong
    # locally corroborated interval survives verification.
    if not viable and evaluations:
        viable = [max(evaluations, key=lambda item: item["ranking"])]

    # Stitch compatible local chains into piecewise, order-preserving paths.
    # Only the primary path contributes to the headline coverage/score;
    # alternative paths are printed separately to avoid double-counting.
    paths = build_piecewise_paths(viable)
    if paths:
        selected = paths[0]["evaluations"]
    else:
        selected = viable

    primary = (
        max(selected, key=lambda item: item["ranking"])
        if selected else None
    )

    # Detect fingerprint segments independently for each alignment.
    segments_by_alignment = []

    for evaluation in selected:
        segments = detect_matching_segments(
            evaluation["candidate"]["matches"],
            max_gap_seconds=1.5,
            max_offset_deviation=0.25,
            min_duration_seconds=5.0,
            min_matches=10,
        )

        segments_by_alignment.append(segments)

    segment_count = sum(
        len(segments)
        for segments in segments_by_alignment
    )

    verification_results = []

    for segments in segments_by_alignment:
        verification_results.extend(
            verify_segments(
                segments=segments,
                features_a=features_a,
                features_b=features_b,
            )
        )

    all_intervals, retained_offsets, retained, rejected_count = (
        consolidate_intervals(selected)
    )

    # Reverify the consolidated intervals.
    mel_results = verify_mel_intervals(
        intervals=all_intervals,
        features_a=features_a,
        features_b=features_b,
    )

    report = calculate_similarity_report(
        chroma_intervals=all_intervals,
        mel_results=mel_results,
        duration_a=processed_a.processed_duration,
        duration_b=processed_b.processed_duration,
    )

    print("\n============= RESULTS =============")
    print("FILES")
    print(
        f"Audio A: {Path(args.audio_a).name} "
        f"({format_time(processed_a.processed_duration)})"
    )
    print(
        f"Audio B: {Path(args.audio_b).name} "
        f"({format_time(processed_b.processed_duration)})"
    )

    print("\nMATCHING")
    print("\nLOCAL SEQUENCE CHAINS")
    print(
        f"{'Offset':>9} {'A range':>17} {'B range':>17} "
        f"{'Length':>8} {'Anchors':>8} {'MAD':>7}"
    )
    print("-" * 83)

    for evaluation in selected:
        candidate = evaluation["candidate"]
        chain = candidate["chain"]
        print(
            f"{candidate['offset']:>+8.2f}s "
            f"{format_time(chain.start_a)}-{format_time(chain.end_a):<11} "
            f"{format_time(chain.start_b)}-{format_time(chain.end_b):<11} "
            f"{chain.duration:>7.2f}s "
            f"{chain.anchor_count:>8} "
            f"{chain.offset_mad:>6.3f}"
        )
    print("\nPIECEWISE ALIGNMENT PATHS")
    if paths:
        for path_index, path in enumerate(paths, start=1):
            print(
                f"Path {path_index}: {len(path['nodes'])} chain(s), "
                f"{path['evidence_duration']:.2f}s corroborated, "
                f"weighted support {path['score']:.2f}"
            )
            for node_index, node in enumerate(path["nodes"], start=1):
                chain = node["chain"]
                offset = chain.median_offset
                if node_index == 1:
                    transition = "start"
                else:
                    previous = path["nodes"][node_index - 2]["chain"]
                    gap_a = chain.start_a - previous.end_a
                    gap_b = chain.start_b - previous.end_b
                    transition = f"gap A {gap_a:.1f}s / B {gap_b:.1f}s"
                print(
                    f"  {node_index}. A {format_time(chain.start_a)}-"
                    f"{format_time(chain.end_a)} -> B "
                    f"{format_time(chain.start_b)}-"
                    f"{format_time(chain.end_b)} | "
                    f"offset {offset:+.2f}s | "
                    f"{node['evidence_duration']:.2f}s evidence | {transition}"
                )
    else:
        print("No corroborated piecewise paths found.")

    print(f"Offset hypotheses tested:  {len(candidates)}")
    print(f"Alignments retained:       {len(retained_offsets)}")
    print(f"Fingerprint segments:      {segment_count}")
    print(f"Intervals rejected:        {rejected_count}")
    print(f"Corroborated intervals:    {len(retained)}")

    if primary is None:
        print("Primary time offset:       Not found")
    else:
        primary_offset = primary["candidate"]["offset"]

        direction = (
            "B passage occurs later"
            if primary_offset > 0
            else "B passage occurs earlier"
            if primary_offset < 0
            else "Same passage position"
        )

        print(
            f"Primary local offset:      {primary_offset:+.2f}s "
            f"({direction})"
        )

    if retained_offsets:
        print("Detected offsets:")
        for offset in retained_offsets:
            print(f"  {offset:+.2f}s")

    print(f"Candidate intervals:       {report.verified_interval_count}")

    print("\nSIMILARITY")
    print("Metrics below describe the primary path only; alternative paths are not added together.")
    print(
        f"Primary-path score:        "
        f"{report.weighted_similarity * 100:.2f}%"
    )
    print(f"Strong-evidence intervals: {report.strong_interval_count}")
    print(
        f"Strong matching duration:  "
        f"{report.strong_duration_a:.2f}s"
    )

    print("\nPRIMARY-PATH COVERAGE")
    print(
        f"Candidate coverage:        "
        f"A {report.candidate_coverage_a:.2f}% | "
        f"B {report.candidate_coverage_b:.2f}%"
    )
    print(
        f"Strong-evidence coverage:  "
        f"A {report.strong_coverage_a:.2f}% | "
        f"B {report.strong_coverage_b:.2f}%"
    )

    if report.interval_scores:

        if retained:
            print("\nALIGNMENT DIAGNOSTICS")
            print(
                f"{'Offset':>9} {'Intervals':>10} "
                f"{'Anchors':>10} {'Duration':>10}"
            )
            print("-" * 45)

            alignment_groups = {}

            for item in retained:
                key = round(item["offset"], 2)

                if key not in alignment_groups:
                    alignment_groups[key] = {
                        "intervals": 0,
                        "anchors": 0,
                        "duration": 0.0,
                    }

                group = alignment_groups[key]
                group["intervals"] += 1
                group["anchors"] += item["fingerprint_support"]
                group["duration"] += item["interval"].duration

            for offset, group in sorted(
                alignment_groups.items(),
                key=lambda entry: entry[1]["duration"],
                reverse=True,
            ):
                print(
                    f"{offset:>+8.2f}s "
                    f"{group['intervals']:>10} "
                    f"{group['anchors']:>10} "
                    f"{group['duration']:>9.2f}s"
                )

            print("\nTOP CORROBORATED INTERVALS")
            print(
                f"{'#':>2} {'Offset':>9} {'Audio A':<17} "
                f"{'Audio B':<17} {'Length':>8} "
                f"{'Score':>8} {'Anchors':>8}"
            )
            print("-" * 88)

            ranked = sorted(
                retained,
                key=lambda item: (
                    item["score"],
                    item["fingerprint_support"],
                    item["interval"].duration,
                ),
                reverse=True,
            )

            for index, item in enumerate(ranked[:30], start=1):
                interval = item["interval"]

                audio_a_range = (
                    f"{format_time(interval.start_a)}-"
                    f"{format_time(interval.end_a)}"
                )
                audio_b_range = (
                    f"{format_time(interval.start_b)}-"
                    f"{format_time(interval.end_b)}"
                )

                print(
                    f"{index:>2} "
                    f"{item['offset']:>+8.2f}s "
                    f"{audio_a_range:<17} "
                    f"{audio_b_range:<17} "
                    f"{interval.duration:>7.2f}s "
                    f"{item['score'] * 100:>7.2f}% "
                    f"{item['fingerprint_support']:>8}"
                )

        else:
            print("\nMATCHED INTERVALS")
            print("No intervals passed local fingerprint corroboration.")

    print("\n===================================")
    print("Analysis complete.")
    print(
        "Similarity scores are heuristic measurements, "
        "not probabilities or legal plagiarism determinations."
    )


if __name__ == "__main__":
    main()
