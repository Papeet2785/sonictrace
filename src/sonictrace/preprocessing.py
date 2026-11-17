from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class PreprocessedAudio:
    audio: np.ndarray
    sample_rate: int
    original_duration: float
    processed_duration: float
    trim_start_seconds: float
    trim_end_seconds: float
    gain_db: float


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = 22050,
    trim_silence: bool = True,
    top_db: float = 35.0,
    target_rms_db: float = -20.0,
    max_gain_db: float = 20.0,
    peak_ceiling: float = 0.98,
) -> PreprocessedAudio:

    # 1. Convert to floating-point representation
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio signal is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    if audio.ndim == 2:
        # SoundFile-style input: (samples, channels)
        audio = librosa.to_mono(audio.T)

    elif audio.ndim != 1:
        raise ValueError("Audio must be mono or stereo.")

    if sample_rate <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")

    original_duration = len(audio) / sample_rate

    # 2. Convert to target sample rate
    if sample_rate != target_sr:
        audio = librosa.resample(
            y=audio,
            orig_sr=sample_rate,
            target_sr=target_sr,
        )

    sample_rate = target_sr

    # 3. Reject silent or nearly silent recordings
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < 1e-8:
        raise ValueError("Audio is silent or nearly silent.")

    # 4. Remove leading and trailing silence
    trim_start = 0.0
    trim_end = 0.0

    if trim_silence:
        resampled_length = len(audio)
        audio, indices = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=2048,
            hop_length=512,
        )

        start_index, end_index = indices

        trim_start = start_index / sample_rate
        trim_end = (resampled_length - end_index) / sample_rate

    if len(audio) == 0:
        raise ValueError("No usable audio remains after trimming.")

    # 5. RMS-based volume normalization
    current_rms = float(np.sqrt(np.mean(audio ** 2)))

    target_rms = 10 ** (target_rms_db / 20)

    gain = target_rms / max(current_rms, 1e-8)

    # Limit amplification to avoid excessive noise amplification
    max_gain = 10 ** (max_gain_db / 20)
    gain = min(gain, max_gain)

    audio = audio * gain

    # 6. Prevent clipping
    peak = float(np.max(np.abs(audio)))

    if peak > peak_ceiling:
        peak_gain = peak_ceiling / peak
        audio = audio * peak_gain
        gain *= peak_gain

    gain_db = 20 * np.log10(max(gain, 1e-12))

    processed_duration = len(audio) / sample_rate

    return PreprocessedAudio(
        audio=audio,
        sample_rate=sample_rate,
        original_duration=original_duration,
        processed_duration=processed_duration,
        trim_start_seconds=trim_start,
        trim_end_seconds=trim_end,
        gain_db=float(gain_db),
    )
