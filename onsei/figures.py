"""
bqplot figures for the notebook
"""
from typing import Tuple, List

import numpy as np
from bqplot import LinearScale, Lines, Axis, Figure, Label

from onsei.utils import segment_speech, SpeechRecord


def update_label_with_phonemes(label: Label, phonemes: List[Tuple[float, float, str]]):
    xs = []
    texts = []
    colors = []
    # Show some separators between phonemes
    for pho_beg, pho_end, _ in phonemes:
        xs.extend([pho_beg, pho_end])
        texts.extend(["|", "|"])
        colors.extend(["gray", "gray"])
    # Show the phonemes in romaji on top of the separator (may overlap if narrow)
    for pho_beg, pho_end, pho in phonemes:
        x = pho_beg + (pho_end - pho_beg) / 2
        xs.append(x)
        texts.append(pho)
        colors.append("orange")

    label.x = xs
    label.y = [0.5 for _ in range(len(xs))]  # center of (default) y axis
    label.text = texts
    label.colors = colors


class ViewRecordFigure(Figure):
    """
    Visualize a single recording
    """

    def __init__(self, **kwargs):
        self.scale_ts = LinearScale()
        self.scale_pitch = LinearScale()
        self.scale_intensity = LinearScale()

        self.line_pitch = Lines(
            x=[],
            y=[],
            scales={'x': self.scale_ts, 'y': self.scale_pitch},
            labels=["Pitch"],
            colors=["dodgerblue"],
            display_legend=True
            )
        self.line_intensity = Lines(
            x=[],
            y=[],
            scales={'x': self.scale_ts, 'y': self.scale_intensity},
            labels=["Intensity"],
            colors=["lightgreen"],
            fill="bottom",
            display_legend=True
            )
        self.line_vad_intensity = Lines(x=[], y=[],
                                        scales={'x': self.scale_ts, 'y': self.scale_intensity},
                                        labels=["Detected Speech"], colors=["red"],
                                        fill="bottom", display_legend=True)
        self.ax_ts = Axis(scale=self.scale_ts, label="Time (s)", grid_lines="solid")
        self.ax_pitch = Axis(scale=self.scale_pitch, label="Pitch (Hz)",
                             orientation="vertical", grid_lines="solid", side="left")
        self.ax_intensity = Axis(scale=self.scale_intensity, label="Intensity (dB)",
                                 orientation="vertical", grid_lines="dashed",
                                 side="right")
        self.label_transcript = Label(x=[], y=[], scales={'x': self.scale_ts})

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

            update_label_with_phonemes(self.label_transcript, rec.phonemes)

    def clear(self):
        with self.line_pitch.hold_sync(), self.line_intensity.hold_sync(), self.line_vad_intensity.hold_sync(), self.label_transcript.hold_sync():
            self.line_pitch.x = []
            self.line_pitch.y = []
            self.line_intensity.x = []
            self.line_intensity.y = []
            self.line_vad_intensity.x = []
            self.line_vad_intensity.y = []
            self.label_transcript.x = []
            self.label_transcript.y = []


class CompareFigure(Figure):

    def __init__(self):
        self.scale_ts = LinearScale()
        self.scale_norm_pitch = LinearScale()

        self.line_cmp_pitch_teacher = Lines(
            x=[],
            y=[],
            scales={'x': self.scale_ts, 'y': self.scale_norm_pitch},
            labels=["Teacher Pitch"],
            colors=["blue"],
            display_legend=True
            )
        self.line_cmp_pitch_student = Lines(
            x=[],
            y=[],
            scales={'x': self.scale_ts, 'y': self.scale_norm_pitch},
            labels=["Student Pitch"],
            colors=["red"],
            display_legend=True
            )
        self.label_transcript = Label(
            x=[],
            y=[],
            text=[],
            colors=[],
            scales={'x': self.scale_ts, 'y': self.scale_norm_pitch},
        )

        self.ax_ts = Axis(
            scale=self.scale_ts,
            label="Time (s)",
            grid_lines="solid",
            )
        self.ax_pitch = Axis(
            scale=self.scale_norm_pitch,
            label="Normalized Pitch",
            orientation="vertical",
            grid_lines="solid",
            side="left",
            )

        super().__init__(
            marks=[
                self.line_cmp_pitch_teacher,
                self.line_cmp_pitch_student,
                self.label_transcript,
                ],
            axes=[self.ax_ts, self.ax_pitch],
            legend_location="top-right",
            title="Pitch comparison on aligned signals",
            )

    def update_data(self, teacher_rec: SpeechRecord, student_rec: SpeechRecord):
        with self.line_cmp_pitch_student.hold_sync(), self.line_cmp_pitch_teacher.hold_sync(), self.label_transcript.hold_sync():
            self.line_cmp_pitch_student.x = teacher_rec.align_ts
            self.line_cmp_pitch_student.y = student_rec.norm_aligned_pitch
            self.line_cmp_pitch_teacher.x = teacher_rec.align_ts
            self.line_cmp_pitch_teacher.y = teacher_rec.norm_aligned_pitch

            update_label_with_phonemes(self.label_transcript, teacher_rec.phonemes)

    def clear(self):
        with self.line_cmp_pitch_student.hold_sync(), self.line_cmp_pitch_teacher.hold_sync():
            self.line_cmp_pitch_student.x = []
            self.line_cmp_pitch_student.y = []
            self.line_cmp_pitch_teacher.x = []
            self.line_cmp_pitch_teacher.y = []
            self.label_transcript.x = []
            self.label_transcript.y = []