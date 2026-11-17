# SonicTrace

**Audio Similarity Detection and Partial-Match Analysis**

SonicTrace is a Python-based audio analysis tool designed to identify shared audio content between two recordings, even when they have been edited differently.

It focuses on detecting reused music and audio passages in recordings containing different edits, background dialogue, sound effects, and changes in timing.

Unlike basic audio comparison tools, SonicTrace attempts to locate matching passages independently rather than requiring the recordings to have a single continuous alignment.

## Key Features

- **Audio preprocessing:** Converts recordings to mono, standardizes the sample rate, trims silence, and normalizes audio levels.
- **Audio fingerprinting:** Extracts spectrogram peaks and generates fingerprints based on frequency relationships and temporal differences.
- **Multiple feature representations:** Uses full-mix and harmonic spectrograms to generate matching hypotheses.
- **Sequence matching:** Identifies groups of fingerprint matches that follow a consistent chronological pattern.
- **Local alignment detection:** Searches for matching passages at different offsets, allowing for edits, cuts, and rearranged sections.
- **Chroma verification:** Compares harmonic pitch-class features to evaluate candidate matches.
- **Mel-spectrogram verification:** Provides an additional spectral similarity check.
- **Partial-match localization:** Reports matching timestamps in both recordings.
- **Coverage analysis:** Estimates how much of each recording is covered by detected matching intervals.

## How It Works

SonicTrace uses a multi-stage audio analysis pipeline.

### 1. Audio Loading and Preprocessing

Both recordings are loaded and converted into a consistent audio representation.

Preprocessing includes:
- Mono conversion
- Resampling to 22,050 Hz
- Silence trimming
- RMS normalization with peak limiting

### 2. Feature Extraction

The system extracts multiple representations of each recording:

- MFCC features
- Chroma features
- Mel spectrograms
- Harmonic Mel spectrograms

These representations provide complementary information about the audio.

### 3. Audio Fingerprinting

SonicTrace detects prominent peaks in the spectrogram and generates fingerprints using pairs of frequency peaks and their temporal separation.

These fingerprints provide a compact representation that can be compared across recordings.

### 4. Candidate Alignment Detection

Fingerprint matches are used to generate possible time-offset hypotheses.

The sequence matcher looks for chronological groups of matches that maintain a reasonably consistent offset between the recordings.

### 5. Local Similarity Verification

Candidate passages are independently evaluated using:

- Chroma similarity
- Mel-spectrogram similarity
- Local fingerprint support

This additional verification helps reject candidate matches that are not supported by multiple feature representations.

### 6. Piecewise Alignment and Coverage

Matching passages can occur at different offsets because of cuts, inserted material, or rearranged edits.

SonicTrace therefore evaluates local matching sequences and constructs piecewise alignment paths rather than assuming one global time offset.

The final report includes matching timestamps, alignment offsets, similarity measurements, and estimated coverage.

## Project Structure

```text
sonictrace/
├── data/
│   ├── input/
│   └── processed/
├── src/
│   └── sonictrace/
│       ├── __init__.py
│       ├── audio_loader.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── fingerprint.py
│       ├── matcher.py
│       ├── segments.py
│       ├── scoring.py
│       └── verification.py
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

## Installation

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

### Setup

Clone the repository:

```powershell
git clone https://github.com/Papeet2785/sonictrace.git
cd sonictrace
```

Install dependencies:

```powershell
uv sync
```

## Usage

Place the two audio recordings in an accessible directory.

Run SonicTrace:

```powershell
uv run python main.py --audio-a "path\to\audio_a.mp3" --audio-b "path\to\audio_b.mp3"
```

For example:

```powershell
uv run python main.py --audio-a "data\input\audio_a.mp3" --audio-b "data\input\audio_b.mp3"
```

SonicTrace processes both recordings and prints a similarity report in the terminal.

## Example Output

The following is an example from a development test involving two differently edited videos containing the same underlying song.

| Measurement | Result |
|---|---:|
| Audio A duration | 5:42.24 |
| Audio B duration | 5:33.43 |
| Local sequence chains | 5 |
| Piecewise alignment paths | 2 |
| Corroborated intervals | 5 |
| Primary-path score | 87.01% |
| Strong matching duration | 109.16 seconds |
| Coverage of Audio A | 31.89% |
| Coverage of Audio B | 32.74% |

Example detected passages:

| Audio A timestamp | Audio B timestamp | Duration | Similarity |
|---|---|---:|---:|
| 3:28.45–3:38.78 | 2:21.15–2:31.49 | 10.33 sec | 93.28% |
| 0:35.74–0:52.20 | 0:35.46–0:51.92 | 16.46 sec | 90.49% |
| 4:07.11–4:33.11 | 3:28.98–3:54.99 | 26.01 sec | 86.59% |
| 1:51.18–2:19.60 | 1:50.36–2:18.79 | 28.42 sec | 85.39% |
| 0:54.40–1:22.34 | 0:53.94–1:21.87 | 27.93 sec | 84.65% |

*These are development results from one test pair, not benchmark accuracy measurements.*

## Technologies Used

- Python
- NumPy
- SciPy
- Librosa
- SoundFile
- Matplotlib
- Pytest

## Limitations

SonicTrace is an experimental audio similarity analysis project.

Current limitations include:

- The detection thresholds and scoring weights are heuristic and have not been calibrated against a large benchmark dataset.
- Similar musical passages may produce false-positive matches.
- Heavy background dialogue, sound effects, distortion, and aggressive audio processing can reduce detection reliability.
- Changes in playback speed or pitch may affect matching performance.
- Detected similarity does not establish the source, ownership, intent, or legal status of reused audio.
- Reported similarity scores are not probabilities or legal plagiarism determinations.

The system should be treated as an analytical aid rather than a definitive plagiarism detector.

## Future Improvements

Potential areas for further development include:

- Larger labeled evaluation datasets
- Precision, recall, and false-positive benchmarking
- Improved handling of playback-speed and pitch changes
- More robust matching under heavy background interference
- Exportable timestamp and alignment reports

## Author

Developed as an independent project exploring digital signal processing, audio fingerprinting, feature extraction, and sequence alignment in Python.