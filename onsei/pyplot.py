"""
Plotting utils using matplotlib
"""
import logging
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import parselmouth

from onsei.speech_record import SpeechRecord
from onsei.utils import replacing_zero_by_nan

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_pitch_and_spectro(
        rec: SpeechRecord,
        window_length=0.03,
        maximum_frequency=500,
        time_step=0.01,
        frequency_step=10.0,
        plot_maximum_frequency=500,
        ):
    title = rec.name if rec.name else None

    # If desired, pre-emphasize the sound fragment
    # before calculating the spectrogram
    pre_emphasized_snd = rec.snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(
        window_length=window_length,
        maximum_frequency=maximum_frequency,
        time_step=time_step,
        frequency_step=frequency_step
        )
    draw_spectrogram(spectrogram, maximum_frequency=plot_maximum_frequency)

    pitch_ts = rec.pitch.xs()
    y = rec.pitch_freq.copy()
    y[y == 0] = np.nan
    plt.plot(pitch_ts, y, 'o', markersize=5, color='w')
    plt.plot(pitch_ts, y, 'o', markersize=2)
    plt.ylim(0, plot_maximum_frequency)
    plt.ylabel("fundamental frequency [Hz]")

    plt.twinx()
    draw_intensity(rec.intensity)

    xmin = rec.snd.xmin if rec.begin_ts is None else rec.begin_ts
    xmax = rec.snd.xmax if rec.end_ts is None else rec.end_ts

    if rec.phonemes:
        plot_phonemes(rec.phonemes, xmin=xmin, xmax=xmax)

    plt.xlim([xmin, xmax])

    if title:
        plt.title(title)


def plot_aligned_intensities(rec: SpeechRecord):
    plt.plot(rec.ref_rec.align_ts,
             rec.ref_rec.znormed_intensity[rec.ref_rec.align_index],
             'b-')
    plt.plot(rec.ref_rec.align_ts,
             rec.znormed_intensity[rec.align_index],
             'g-')
    plt.title("Aligned student intensity (green) to match teacher (blue)")


def plot_aligned_pitches(rec: SpeechRecord):
    plt.plot(rec.ref_rec.align_ts, rec.ref_rec.norm_aligned_pitch, 'b.')
    plt.plot(rec.ref_rec.align_ts, rec.norm_aligned_pitch, 'g.')
    plt.title("Applying the same alignment on normalized pitch")


def plot_pitch_errors(rec: SpeechRecord):
    _plot_pitch_errors(rec.pitch_diffs_ts, rec.pitch_diffs)
    if rec.ref_rec.phonemes:
        plot_phonemes(rec.ref_rec.phonemes, y=0, color="black")


def plot_aligned_pitches_and_phonemes(rec: SpeechRecord):
    plt.plot(rec.ref_rec.align_ts, rec.ref_rec.norm_aligned_pitch, 'b-')
    plt.plot(rec.ref_rec.align_ts, rec.norm_aligned_pitch, 'r-')

    if rec.ref_rec.phonemes:
        plot_phonemes(rec.ref_rec.phonemes, y=0, color="black", font_size=18)

    plt.xlabel("Time(s)")
    plt.ylabel("Normalized Pitch")
    plt.legend(["Reference audio", "Your recording"])


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


def plot_phonemes(
        phonemes: List[Tuple[float, float, str]],
        y: float = 10,
        color: str = 'white',
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        font_size=24,
):
    # Setup font configuration of matplotlib to plot Japanese text
    # from matplotlib import rcParams
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio',
    #                                'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic',
    #                                'Noto Sans CJK JP']

    font_dict = {
        'color': color,
        'size': font_size,
        }
    sep_font_dict = {
        'color': 'gray',
        'size': font_size,
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


def _plot_pitch_errors(pitch_diffs_ts, pitch_diffs):
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
