
import librosa
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioFeatures:
    mfcc: np.ndarray
    chroma: np.ndarray
    mel_spectrogram: np.ndarray
    harmonic_mel_spectrogram: np.ndarray
    frame_times: np.ndarray


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> AudioFeatures:

    if audio.size == 0:
        raise ValueError("Cannot extract features from empty audio.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    if len(audio) < n_fft:
        raise ValueError(f"Audio is too short. Need at least {n_fft} samples.")

    # Original audio features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_chroma=12,
    )

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        power=2.0,
        fmin=50.0,
        fmax=sample_rate / 2,
    )

    mel_db = librosa.power_to_db(mel_power, ref=np.max)

    # Harmonic-percussive source separation
    harmonic_audio = librosa.effects.harmonic(
        y=audio,
        margin=3.0,
    )

    # Mel spectrogram from the harmonic component
    harmonic_mel_power = librosa.feature.melspectrogram(
        y=harmonic_audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        power=2.0,
        fmin=50.0,
        fmax=sample_rate / 2,
    )

    harmonic_mel_db = librosa.power_to_db(
        harmonic_mel_power,
        ref=np.max,
    )

    frame_count = mfcc.shape[1]

    frame_times = librosa.frames_to_time(
        np.arange(frame_count),
        sr=sample_rate,
        hop_length=hop_length,
    )

    return AudioFeatures(
        mfcc=mfcc,
        chroma=chroma,
        mel_spectrogram=mel_db,
        harmonic_mel_spectrogram=harmonic_mel_db,
        frame_times=frame_times,
    )
