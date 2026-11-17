
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
    """Generate offset hypotheses from full-mix and harmonic fingerprints."""

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

    combined = []

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

        for candidate in result.candidates[:15]:
            existing = next(
                (
                    item for item in combined
                    if abs(item["offset"] - candidate.offset) < 0.25
                ),
                None,
            )

            if existing is None:
                combined.append({
                    "offset": candidate.offset,
                    "matches": candidate.matches,
                    "match_count": candidate.match_count,
                    "support": candidate.match_count,
                    "sources": {source},
                })
            else:
                existing["support"] += candidate.match_count
                existing["sources"].add(source)

                if candidate.match_count > existing["match_count"]:
                    existing["offset"] = candidate.offset
                    existing["matches"] = candidate.matches
                    existing["match_count"] = candidate.match_count

    combined.sort(
        key=lambda item: item["support"],
        reverse=True,
    )

    return combined[:30]


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


def evaluate_candidate(
    candidate,
    features_a,
    features_b,
    duration_a,
    duration_b,
):
    """Evaluate one offset using chroma, Mel, and local fingerprints."""

    intervals = detect_chroma_intervals(
        features_a=features_a,
        features_b=features_b,
        offset=candidate["offset"],
        similarity_threshold=0.80,
        min_duration_seconds=2.0,
        max_gap_seconds=0.5,
    )

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

    evidence_integral = (
        candidate_duration * report.weighted_similarity
    )

    ranking = (
        report.strong_duration_a,
        evidence_integral,
        report.weighted_similarity,
        candidate["support"],
    )

    return {
        "candidate": candidate,
        "intervals": intervals,
        "mel_results": mel_results,
        "report": report,
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

    print(f"Testing {len(candidates)} offset hypotheses...")

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

    # Require at least one strong interval with local fingerprint support.
    viable = []

    for evaluation in evaluations:
        mel_by_id = {
            id(result.interval): result
            for result in evaluation["mel_results"]
        }

        has_corroborated_strong_interval = False

        for interval in evaluation["intervals"]:
            mel_result = mel_by_id.get(id(interval))

            if mel_result is None:
                continue

            support = local_fingerprint_support(
                interval,
                evaluation["candidate"],
            )

            required = required_local_support(interval.duration)

            if (
                is_strong_interval(interval, mel_result)
                and support >= required
            ):
                has_corroborated_strong_interval = True
                break

        if has_corroborated_strong_interval:
            viable.append(evaluation)

    # Fallback: retain the best alignment for diagnostics.
    if viable:
        selected = viable
    elif evaluations:
        selected = [
            max(evaluations, key=lambda item: item["ranking"])
        ]
    else:
        selected = []

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
            "B starts later"
            if primary_offset > 0
            else "B starts earlier"
            if primary_offset < 0
            else "Same starting position"
        )

        print(
            f"Primary time offset:       {primary_offset:+.2f}s "
            f"({direction})"
        )

    if retained_offsets:
        print("Detected offsets:")
        for offset in retained_offsets:
            print(f"  {offset:+.2f}s")

    print(f"Candidate intervals:       {report.verified_interval_count}")

    print("\nSIMILARITY")
    print(
        f"Duration-weighted score:   "
        f"{report.weighted_similarity * 100:.2f}%"
    )
    print(f"Strong-evidence intervals: {report.strong_interval_count}")
    print(
        f"Strong matching duration:  "
        f"{report.strong_duration_a:.2f}s"
    )

    print("\nCOVERAGE")
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
