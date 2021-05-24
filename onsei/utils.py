import contextlib
import logging
import os
import tempfile
from functools import cached_property
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import sox

# Hide prints
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    from dtw import dtw, rabinerJuangStepPattern

from onsei.vad import detect_voice_with_webrtcvad

PITCH_TIME_STEP = 0.02
MINIMUM_PITCH = 100.0


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class SpeechRecord:

    def __init__(
        self,
        wav_filename: str,
        transcript: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.wav_filename = wav_filename
        self.transcript = transcript
        self.name = name

        self.snd = parselmouth.Sound(self.wav_filename)
        if self.snd.sampling_frequency != 16000:
            raise ValueError(f"Sampling frequency should be 16kHz, "
                             f"not {self.snd.sampling_frequency}Hz !")
        if self.snd.n_channels != 1:
            raise ValueError(f"Should only have 1 audio channel, "
                             f"not {self.snd.n_channels} !")

        self.pitch = self.snd.to_pitch(time_step=PITCH_TIME_STEP)

        self.pitch_freq = self.pitch.selected_array['frequency']
        self.pitch_freq[self.pitch_freq == 0] = np.nan

        self.pitch_freq_filtered, self.mean_pitch_freq, self.std_pitch_freq = \
            cleanup_pitch_freq(self.pitch_freq)

        self.intensity = self.snd.to_intensity(MINIMUM_PITCH)

        # Run a simple voice detection algorithm to find
        # where the speech starts and ends
        self.vad_ts, self.vad_is_speech, self.begin_ts, self.end_ts = \
            detect_voice_with_webrtcvad(self.wav_filename)
        self.begin_idx, self.end_idx = ts_sequences_to_index(
            [self.begin_ts, self.end_ts],
            self.intensity.xs()
            )
        logging.debug(f"Voice detected from {self.begin_ts}s to {self.end_ts}s")

        self.phonemes = None
        if self.transcript:
            self.phonemes = segment_speech(self.wav_filename, self.transcript,
                                           self.begin_ts, self.end_ts)
            logging.debug(f"Phonemes segmentation for teacher: {self.phonemes}")

        # Initialize alignment related attributes
        self.ref_rec = None
        self.align_ts = None
        self.align_index = None
        self.pitch_diffs_ts = None
        self.pitch_diffs = None

    def plot_pitch_and_spectro(
        self,
        window_length=0.03,
        maximum_frequency=500,
        time_step=0.01,
        frequency_step=10.0,
        plot_maximum_frequency=500,
    ):
        title = self.name if self.name else None

        # If desired, pre-emphasize the sound fragment
        # before calculating the spectrogram
        pre_emphasized_snd = self.snd.copy()
        pre_emphasized_snd.pre_emphasize()
        spectrogram = pre_emphasized_snd.to_spectrogram(
            window_length=window_length,
            maximum_frequency=maximum_frequency,
            time_step=time_step,
            frequency_step=frequency_step
        )
        draw_spectrogram(spectrogram, maximum_frequency=plot_maximum_frequency)

        pitch_ts = self.pitch.xs()
        y = self.pitch_freq_filtered.copy()
        y[y == 0] = np.nan
        plt.plot(pitch_ts, y, 'o', markersize=5, color='w')
        plt.plot(pitch_ts, y, 'o', markersize=2)
        plt.ylim(0, plot_maximum_frequency)
        plt.ylabel("fundamental frequency [Hz]")

        plt.twinx()
        draw_intensity(self.intensity)

        xmin = self.snd.xmin if self.begin_ts is None else self.begin_ts
        xmax = self.snd.xmax if self.end_ts is None else self.end_ts

        if self.phonemes:
            plot_phonemes(self.phonemes, xmin=xmin, xmax=xmax)

        plt.xlim([xmin, xmax])

        if title:
            plt.title(title)

    def align_with(self, ref_rec: "SpeechRecord"):
        self.ref_rec = ref_rec

        # Aliases for clarity
        student_rec = self
        teacher_rec = ref_rec

        # x is the query (which we will "warp") and y the reference
        x = student_rec.znormed_intensity
        y = teacher_rec.znormed_intensity
        step_pattern = rabinerJuangStepPattern(4, "c", smoothed=True)
        align = dtw(x, y, keep_internals=True, step_pattern=step_pattern)

        student_rec.align_index = align.index1
        teacher_rec.align_index = align.index2

        # Timestamp for each point in the alignment
        student_rec.align_ts = student_rec.intensity.xs()[
                               student_rec.begin_idx:student_rec.end_idx][align.index1]
        teacher_rec.align_ts = teacher_rec.intensity.xs()[
                               teacher_rec.begin_idx:teacher_rec.end_idx][align.index2]

    @cached_property
    def znormed_intensity(self):
        return znormed(
            self.intensity.values[0, self.begin_idx:self.end_idx])

    @cached_property
    def aligned_pitch(self):
        # Intensity and pitch computed by parselmouth do not have the same timestamps,
        # so we mean to find the frames in the pitch signal using the aligned timestamps
        align_idx_pitch = ts_sequences_to_index(self.align_ts, self.pitch.xs())
        pitch = self.pitch_freq_filtered[align_idx_pitch]
        return pitch

    @cached_property
    def norm_aligned_pitch(self):
        return (self.aligned_pitch - self.mean_pitch_freq) / self.std_pitch_freq

    def compare_pitch(self):
        self.pitch_diffs_ts = []
        self.pitch_diffs = []
        for idx, (teacher, student) in enumerate(
                zip(self.ref_rec.norm_aligned_pitch, self.norm_aligned_pitch)):
            if not np.isnan(teacher) and not np.isnan(student):
                self.pitch_diffs_ts.append(self.ref_rec.align_ts[idx])
                self.pitch_diffs.append(teacher - student)

        distances = [abs(p) for p in self.pitch_diffs]
        mean_distance = np.mean(distances)
        return mean_distance

    def plot_aligned_intensities(self):
        plt.plot(self.ref_rec.align_ts,
                 self.ref_rec.znormed_intensity[self.ref_rec.align_index],
                 'b-')
        plt.plot(self.ref_rec.align_ts,
                 self.znormed_intensity[self.align_index],
                 'g-')
        plt.title("Aligned student intensity (green) to match teacher (blue)")

    def plot_aligned_pitches(self):
        plt.plot(self.ref_rec.align_ts, self.ref_rec.norm_aligned_pitch, 'b.')
        plt.plot(self.ref_rec.align_ts, self.norm_aligned_pitch, 'g.')
        plt.title("Applying the same alignment on normalized pitch")

    def plot_pitch_errors(self):
        plot_pitch_errors(self.pitch_diffs_ts, self.pitch_diffs)
        if self.ref_rec.phonemes:
            plot_phonemes(self.ref_rec.phonemes, y=0, color="black")


def draw_spectrogram(spectrogram, dynamic_range=70, maximum_frequency=None):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    if not maximum_frequency:
        maximum_frequency = spectrogram.ymax
    plt.ylim([spectrogram.ymin, maximum_frequency])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim([0, max(intensity.values[0, :])])
    plt.ylabel("intensity [dB]")


def draw_pitch(pitch: parselmouth.Pitch, maximum_frequency=None):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = replacing_zero_by_nan(pitch.selected_array['frequency'])
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    if not maximum_frequency:
        maximum_frequency = pitch.ceiling
    plt.ylim(0, maximum_frequency)
    plt.ylabel("fundamental frequency [Hz]")


def cleanup_pitch_freq(pitch_freq):
    """
    Remove some outliers from the pitch frequencies
    """
    mean = np.nanmean(pitch_freq)
    std = np.nanstd(pitch_freq)

    min_cut = mean - std * 2.5
    max_cut = mean + std * 2.5
    logging.debug(f"Pitch frequencies cleanup: mean is {mean:.2f} Hz, "
                  f"keeping values within [{min_cut:.2f} Hz, {max_cut:.2f} Hz]")

    new_sig = []
    for x in pitch_freq:
        if not np.isnan(x) and (min_cut <= x <= max_cut):
            new_sig.append(x)
        else:
            new_sig.append(np.nan)

    new_sig = np.array(new_sig)

    return new_sig, mean, std


def plot_phonemes(
        phonemes: List[Tuple[float, float, str]],
        y: float = 10,
        color: str = 'white',
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
):
    # Setup font configuration of matplotlib to plot Japanese text
    # from matplotlib import rcParams
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio',
    #                                'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic',
    #                                'Noto Sans CJK JP']

    font_dict = {
        'color': color,
        'size': 24,
        }
    sep_font_dict = {
        'color': 'gray',
        'size': 24,
        }

    plt_xmin, plt_xmax = plt.xlim()
    if xmin is None:
        xmin = plt_xmin
    if xmin is None:
        xmax = plt_xmax

    def is_within_xlims(x):
        return (xmin is None or x >= xmin) and (xmax is None or x <= xmax)

    for pho_beg, pho_end, pho in phonemes:
        ts = pho_beg + (pho_end - pho_beg) / 2
        if is_within_xlims(pho_beg):
            plt.text(pho_beg, y, "|", fontdict=sep_font_dict)
        if is_within_xlims(pho_end):
            plt.text(pho_end, y, "|", fontdict=sep_font_dict)
        if is_within_xlims(ts):
            plt.text(ts, y, pho, fontdict=font_dict)


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


def plot_pitch_errors(pitch_diffs_ts, pitch_diffs):
    cc = ['colors'] * len(pitch_diffs)
    for n, val in enumerate(pitch_diffs):
        if abs(val) < 1:
            cc[n] = 'green'
        elif abs(val) < 2:
            cc[n] = 'yellow'
        elif abs(val) < 3:
            cc[n] = 'orange'
        else:
            cc[n] = 'red'

    plt.bar(pitch_diffs_ts, pitch_diffs, width=0.008, color=cc)
    plt.title('Pitch "error"')


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
    from PySegmentKit import PySegmentKit

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
        result = segmented[basenames[0]]
        return result


def save_cropped_audio(src_wav_filename, dst_wav_filename, begin_ts, end_ts):
    with open(dst_wav_filename, 'w') as f:
        tfm = sox.Transformer()
        tfm.trim(begin_ts, end_ts)
        # Hide the warnings, a bit ugly but setting SOX's verbosity did not work
        with open(os.devnull, "w") as g, contextlib.redirect_stderr(g):
            tfm.build_file(src_wav_filename, f.name)
