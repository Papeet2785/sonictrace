from pathlib import Path

import librosa
import numpy as np


SUPPORTED_FORMATS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
}


def load_audio(file_path: str, sample_rate: int = 22050):
    """
    Load an audio file and standardize it.

    Returns:
        audio: Mono audio signal
        sample_rate: Target sample rate
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    audio, sr = librosa.load(
        path,
        sr=sample_rate,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    return audio, sr


def get_audio_info(audio, sample_rate):
    duration = len(audio) / sample_rate

    return {
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "samples": len(audio),
        "peak_amplitude": round(float(np.max(np.abs(audio))), 4),
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4),
    }
