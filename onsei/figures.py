"""
bqplot figures for the notebook
"""
from typing import Tuple

import numpy as np
from bqplot import LinearScale, Lines, Axis, Figure, Label

from onsei.utils import segment_speech, SpeechRecord


def update_label_with_phonemes(
    label: Label,
    phonemes: Tuple[float, float, str],
    xmin: float,
    xmax: float,
    y: float = 0.5
):
    # Position on the figure is a percentage (0.0 for left most position, 1.0 for right
    # most), so we need to convert the ts into a ratio using the provided xmin and xmax
    def ts_to_ratio(ts):
        return (ts - xmin) / (xmax - xmin)

    xs = []
    texts = []
    colors = []
    # Show some separators between phonemes
    for pho_beg, pho_end, _ in phonemes:
        xs.extend([ts_to_ratio(pho_beg), ts_to_ratio(pho_end)])
        texts.extend(["|", "|"])
        colors.extend(["gray", "gray"])
    # Show the phonemes in romaji on top of the separator (may overlap if narrow)
    for pho_beg, pho_end, pho in phonemes:
        ts = pho_beg + (pho_end - pho_beg) / 2
        x = ts_to_ratio(ts)
        xs.append(x)
        texts.append(pho)
        colors.append("orange")

    label.x = xs
    label.y = [y for _ in range(len(xs))]
    label.text = texts
    label.colors = colors


class ViewRecordFigure(Figure):
    """
    Visualize a single recording
    """

    def __init__(self, **kwargs):
        scale_ts = LinearScale()
        scale_pitch = LinearScale()
        scale_intensity = LinearScale()

        self.line_pitch = Lines(
            x=[],
            y=[],
            scales={'x': scale_ts, 'y': scale_pitch},
            labels=["Pitch"],
            colors=["dodgerblue"],
            display_legend=True
            )
        self.line_intensity = Lines(
            x=[],
            y=[],
            scales={'x': scale_ts, 'y': scale_intensity},
            labels=["Intensity"],
            colors=["lightgreen"],
            fill="bottom",
            display_legend=True
        )
        self.line_vad_intensity = Lines(x=[], y=[],
                                        scales={'x': scale_ts, 'y': scale_intensity},
                                        labels=["Detected Speech"], colors=["red"],
                                        fill="bottom", display_legend=True)
        self.ax_ts = Axis(scale=scale_ts, label="Time (s)", grid_lines="solid")
        self.ax_pitch = Axis(scale=scale_intensity, label="Pitch (Hz)",
                             orientation="vertical", grid_lines="solid", side="left")
        self.ax_intensity = Axis(scale=scale_intensity, label="Intensity (dB)",
                                 orientation="vertical", grid_lines="solid",
                                 side="right")
        self.label_transcript = Label(x=[], y=[], text=[], colors=[])

        super().__init__(
            marks=[self.line_intensity, self.line_vad_intensity, self.line_pitch,
                   self.label_transcript],
            axes=[self.ax_ts, self.ax_pitch, self.ax_intensity],
            legend_location="top-right",
            **kwargs,
        )

    def update_data(self, rec: SpeechRecord):
        with self.line_pitch.hold_sync(), self.line_intensity.hold_sync(), self.line_vad_intensity.hold_sync(), self.label_transcript.hold_sync():
            y = rec.pitch_freq_filtered.copy()
            y[y == 0] = np.nan
            self.line_pitch.x = rec.pitch.xs()
            self.line_pitch.y = y

            self.line_intensity.x = rec.intensity.xs()
            self.line_intensity.y = rec.intensity.values.T

            self.line_vad_intensity.x = rec.intensity.xs()[rec.begin_idx:rec.end_idx]
            self.line_vad_intensity.y = rec.intensity.values.T[
                                        rec.begin_idx:rec.end_idx]

            phonemes = segment_speech(rec.wav_filename, rec.transcript, rec.begin_ts,
                                      rec.end_ts)
            xmin, xmax = self.line_pitch.x[0], self.line_pitch.x[-1]
            update_label_with_phonemes(self.label_transcript, phonemes, xmin, xmax)


class CompareFigure(Figure):

    def __init__(self):
        scale_ts = LinearScale()
        scale_cmp_ts = LinearScale()
        scale_norm_pitch = LinearScale()

        self.line_cmp_pitch_teacher = Lines(
            x=[],
            y=[],
            scales={'x': scale_ts, 'y': scale_norm_pitch},
            labels=["Teacher Norm Pitch"],
            colors=["blue"],
            display_legend=True
            )
        self.line_cmp_pitch_student = Lines(
            x=[],
            y=[],
            scales={'x': scale_ts, 'y': scale_norm_pitch},
            labels=["Student Norm Pitch"],
            colors=["red"],
            display_legend=True
            )
        self.ax_cmp_ts = Axis(scale=scale_cmp_ts, label="Time (s)", grid_lines="solid")
        self.ax_cmp_pitch = Axis(
            scale=scale_norm_pitch,
            label="Normalized Pitch",
            orientation="vertical",
            grid_lines="solid",
            side="left"
            )

        super().__init__(
            marks=[self.line_cmp_pitch_teacher, self.line_cmp_pitch_student],
            axes=[self.ax_cmp_ts, self.ax_cmp_pitch],
            legend_location="top-right",
            title="Pitch comparison"
            )

    def update_data(self, teacher_rec: SpeechRecord, student_rec: SpeechRecord):
        with self.line_cmp_pitch_student.hold_sync(), self.line_cmp_pitch_teacher.hold_sync():
            self.line_cmp_pitch_student.x = teacher_rec.align_ts
            self.line_cmp_pitch_student.y = student_rec.norm_aligned_pitch
            self.line_cmp_pitch_teacher.x = teacher_rec.align_ts
            self.line_cmp_pitch_teacher.y = teacher_rec.norm_aligned_pitch

    def clear(self):
        with self.line_cmp_pitch_student.hold_sync(), self.line_cmp_pitch_teacher.hold_sync():
            self.line_cmp_pitch_student.x = []
            self.line_cmp_pitch_student.y = []
            self.line_cmp_pitch_teacher.x = []
            self.line_cmp_pitch_teacher.y = []
