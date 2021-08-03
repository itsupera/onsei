import contextlib
import logging
import os
import subprocess
import tempfile
from typing import List, Tuple, Iterable

import numpy as np
import parselmouth
from PySegmentKit import PySegmentKit


logging.basicConfig(level=logging.INFO)
logging.getLogger("sox").setLevel(logging.ERROR)


def get_noise_from_intensity(intensity: parselmouth.Intensity) -> float:
    """
    Compute noise level by sampling the beginning and end of the intensity signal
    """
    ts_noise = 0.1
    samples = []
    xs = intensity.xs()
    duration = xs[-1]
    for x, y in zip(xs, intensity.values[0, :]):
        if x < ts_noise or x > duration - ts_noise:
            samples.append(y)
    logging.debug(f"Computing noise level using {len(samples)} samples")
    return np.mean(samples)


def detect_voice_activity_from_intensity(intensity: parselmouth.Intensity) -> tuple:
    noise = get_noise_from_intensity(intensity)
    logging.debug(f"Noise level: {noise:.2f} dB")
    sig = intensity.values[0, :]
    mean = np.mean(sig)
    logging.debug(f"Mean volume: {mean:.2f} dB")
    vad = np.array(
        [(1.0 if x > (noise + ((mean - noise) / 1.5)) else 0.0) for x in sig])
    begin_idx, end_idx = vad.argmax(), len(vad) - vad[::-1].argmax() - 1
    begin_ts, end_ts = intensity.xs()[begin_idx], intensity.xs()[end_idx]
    return vad, begin_idx, end_idx, begin_ts, end_ts


def detect_voice_activity_from_pitch(pitch: parselmouth.Pitch):
    pitch_freq = pitch.selected_array['frequency']
    vad = pitch_freq > 0
    begin_idx, end_idx = vad.argmax(), len(vad) - vad[::-1].argmax() - 1
    begin_ts, end_ts = pitch.xs()[begin_idx], pitch.xs()[end_idx]
    logging.debug(f"Voice detected between {begin_ts:.2f}s and {end_ts:.2f}s")
    return vad, begin_ts, end_ts


def ts_sequences_to_index(ts_seq1, ts_seq2):
    """ For each ts in ts_seq1, find the idx in ts_seq2 that is the closest """
    index = []
    for ts in ts_seq1:
        index.append(abs(ts_seq2 - ts).argmin())
    return index


def replacing_zero_by_nan(x):
    x_copy = x.copy()
    x_copy[x_copy == 0] = np.nan
    return x_copy


def znormed(x):
    return (x - np.mean(x)) / np.std(x)


def segment_speech(
        wav_filename: str,
        transcript: str,
        begin_ts: float,
        end_ts: float,
        ) -> List[Tuple[float, float, str]]:
    """
    Find the phonemes using the audio file and a transcript in hiragana.
    Return the start time, end time and phoneme for each detected phoneme.
    """
    # PySegmentKit expects the audio and transcript to be under the same directory,
    # so we copy / create temporary files to handle this.
    with tempfile.TemporaryDirectory() as tmpdir:
        save_cropped_audio(wav_filename, os.path.join(tmpdir, "tmp.wav"), begin_ts,
                           end_ts)

        with open(os.path.join(tmpdir, "tmp.txt"), "w") as f:
            f.write(transcript)

        sk = PySegmentKit(tmpdir,
                          disable_silence_at_ends=True,
                          leave_dict=False,
                          debug=True,
                          triphone=False,
                          input_mfcc=False)

        # Hide the prints
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            segmented = sk.segment()

        basenames = list(segmented.keys())
        # We copied only one wav file and its transcript,
        # so we should only get one result.
        assert len(basenames) == 1, basenames
        # Add begin_ts such that the timestamps correspond to the audio before cropping
        result = [
            (pho_beg + begin_ts, pho_end + begin_ts, pho)
            for pho_beg, pho_end, pho in segmented[basenames[0]]
        ]
        return result


def save_cropped_audio(src_wav_filename: str, dst_wav_filename: str, begin_ts: float, end_ts: float):
    p = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(begin_ts), "-to", str(end_ts),
         "-i", src_wav_filename, dst_wav_filename])
    p.wait()


def parse_wav_file_to_sound_obj(wav_filename: str) -> parselmouth.Sound:
    snd = parselmouth.Sound(wav_filename)
    if snd.sampling_frequency != 16000:
        raise ValueError(f"Sampling frequency should be 16kHz, "
                         f"not {snd.sampling_frequency}Hz !")
    if snd.n_channels != 1:
        raise ValueError(f"Should only have 1 audio channel, "
                         f"not {snd.n_channels} !")
    return snd


def check_wav_content_is_valid(content: bytes):
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_filename = os.path.join(tmp_dir, 'tmp.wav')
        with open(wav_filename, 'wb') as tmp_file:
            tmp_file.write(content)
        # Try parsing to Sound object, will raise an Exception if anything is wrong
        parse_wav_file_to_sound_obj(wav_filename)


def phonemes_to_step_function(
    phonemes: List[Tuple[float, float, str]],
    timestamps: Iterable[float]
) -> List[float]:
    """
    In order to align speech recordings based on phoneme positions,
    create a step function that we can be used for DTW.
    The value is 0 before the first phoneme, 1 during the first phoneme, and so on.
    """
    xs = []
    for ts in timestamps:
        if ts < phonemes[0][0]:
            x = 0
        else:
            for idx, pho in enumerate(phonemes):
                if ts < pho[1]:
                    x = idx + 1
                    break
            else:
                x = len(phonemes) + 1

        xs.append(x)

    return xs


def convert_audio(original_audio_filepath: str, converted_wav_filepath: str) -> None:
    p = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", original_audio_filepath, "-ar", "16000", "-ac", "1",
         converted_wav_filepath])
    p.wait()