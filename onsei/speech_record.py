"""
SpeechRecord handles the processing of sentence audio recording
"""
import contextlib
import logging
import os
from enum import Enum, auto
from functools import cached_property
from typing import Optional, Union

import numpy as np

from onsei.sentence import Sentence
from onsei.utils import parse_wav_file_to_sound_obj, ts_sequences_to_index, segment_speech, znormed, \
    phonemes_to_step_function

# Hide prints
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    from dtw import dtw, rabinerJuangStepPattern

from onsei.vad import detect_voice_with_webrtcvad

PITCH_TIME_STEP = 0.005
INTENSITY_TIME_STEP = 0.005
MINIMUM_PITCH = 100.0


logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sox").setLevel(logging.ERROR)


class AlignmentMethod(str, Enum):
    phonemes = "phonemes"
    intensity = "intensity"


class SpeechRecord:

    def __init__(
        self,
        wav_filename: str,
        sentence: Optional[Union[Sentence, str]] = None,
        name: Optional[str] = None,
    ):
        self.wav_filename = wav_filename

        if isinstance(sentence, Sentence):
            self.sentence = sentence
        elif isinstance(sentence, str):
            self.sentence = Sentence(sentence)
        else:
            self.sentence = None

        self.name = name

        self.snd = parse_wav_file_to_sound_obj(self.wav_filename)

        self.pitch = self.snd.to_pitch(time_step=PITCH_TIME_STEP).kill_octave_jumps().smooth()

        self.pitch_freq = self.pitch.selected_array['frequency']
        self.pitch_freq[self.pitch_freq == 0] = np.nan
        self.mean_pitch_freq = np.nanmean(self.pitch_freq)
        self.std_pitch_freq = np.nanstd(self.pitch_freq)

        self.intensity = self.snd.to_intensity(MINIMUM_PITCH, time_step=INTENSITY_TIME_STEP)

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
        if self.sentence:
            self.phonemes = segment_speech(self.wav_filename, self.sentence.julius_transcript,
                                           self.begin_ts, self.end_ts)
            logging.debug(f"Phonemes segmentation for teacher: {self.phonemes}")

        # Initialize alignment related attributes
        self.ref_rec = None
        self.align_ts = None
        self.align_index = None
        self.pitch_diffs_ts = None
        self.pitch_diffs = None

    def align_with(self, ref_rec: "SpeechRecord", method: AlignmentMethod = AlignmentMethod.phonemes):
        self.ref_rec = ref_rec

        # Aliases for clarity
        student_rec = self
        teacher_rec = ref_rec

        # x is the query (which we will "warp") and y the reference
        if method == AlignmentMethod.phonemes:
            if not teacher_rec.phonemes:
                raise NoPhonemeSegmentationError(teacher_rec)
            if not student_rec.phonemes:
                raise NoPhonemeSegmentationError(student_rec)
            x = phonemes_to_step_function(student_rec.phonemes, student_rec.timestamps_for_alignment)
            y = phonemes_to_step_function(teacher_rec.phonemes, teacher_rec.timestamps_for_alignment)
        elif method == AlignmentMethod.intensity:
            x = student_rec.znormed_intensity
            y = teacher_rec.znormed_intensity
        else:
            raise ValueError(f"Unknown method {method} !")

        step_pattern = rabinerJuangStepPattern(4, "c", smoothed=True)
        try:
            align = dtw(x, y, keep_internals=True, step_pattern=step_pattern)
        except ValueError as exc:
            raise AlignmentError(exc)

        student_rec.align_index = align.index1
        teacher_rec.align_index = align.index2

        # Timestamp for each point in the alignment
        student_rec.align_ts = student_rec.timestamps_for_alignment[align.index1]
        teacher_rec.align_ts = teacher_rec.timestamps_for_alignment[align.index2]

    @cached_property
    def timestamps_for_alignment(self):
        return self.intensity.xs()[self.begin_idx:self.end_idx]

    @cached_property
    def znormed_intensity(self):
        return znormed(
            self.intensity.values[0, self.begin_idx:self.end_idx])

    @property
    def aligned_pitch(self):
        # Intensity and pitch computed by parselmouth do not have the same timestamps,
        # so we mean to find the frames in the pitch signal using the aligned timestamps
        align_idx_pitch = ts_sequences_to_index(self.align_ts, self.pitch.xs())
        pitch = self.pitch_freq[align_idx_pitch]
        return pitch

    @property
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


class AlignmentError(Exception):
    pass


class NoPhonemeSegmentationError(Exception):
    def __init__(self, record: SpeechRecord):
        self.record = record