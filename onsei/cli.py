import contextlib
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer

from onsei.sentence import Sentence
from onsei.utils import SpeechRecord

app = typer.Typer()


@app.command()
def view(wav_filename: str, sentence: Optional[str] = None) -> None:
    """
    Visualize a recording
    """
    record = SpeechRecord(wav_filename, sentence)

    plt.figure()
    record.plot_pitch_and_spectro()
    plt.show()


@app.command()
def compare(teacher_wav_filename: str, student_wav_filename: str,
            show_graphs: bool = True, notebook: bool = False,
            sentence: Optional[str] = None) -> float:
    """
    Compare a teacher and student recording of the same sentence
    """

    print(f"Comparing {teacher_wav_filename} with {student_wav_filename}")

    teacher_rec = SpeechRecord(teacher_wav_filename, sentence, name="Teacher")
    student_rec = SpeechRecord(student_wav_filename, sentence, name="Student")

    if show_graphs:
        plt.figure()
        plt.subplot(211)
        teacher_rec.plot_pitch_and_spectro()
        plt.subplot(212)
        student_rec.plot_pitch_and_spectro()
        plt.show(block=False)

    student_rec.align_with(teacher_rec)
    mean_distance = student_rec.compare_pitch()
    print(f"Mean distance: {mean_distance:.2f} "
          f"(smaller means student speech is closer to teacher)")

    # Plot the warped intensity and pitches
    if show_graphs:
        plt.figure()
        plt.subplot(311)
        student_rec.plot_aligned_intensities()
        plt.subplot(312)
        student_rec.plot_aligned_pitches()
        plt.subplot(313)
        student_rec.plot_pitch_errors()
        plt.show(block=False)

        if not notebook:
            input("Press Enter to close graphs and quit")

    return mean_distance


@app.command()
def benchmark(teacher_base_folder: str, student_base_folder: str,
              stats_filepath: Optional[str] = None):
    """
    Compute stats by comparing many sentences.
    The `teacher_base_folder` and `student_base_folder` must be organized with one
    folder for each sentence, e.g., base_folder/sentenceX/recordingY.wav.
    Only sentence folders that are both in `teacher_base_folder` and
    `student_base_folder` will be compared.
    """
    teacher_sentence_folders = set(os.listdir(teacher_base_folder))
    student_sentence_folders = set(os.listdir(student_base_folder))
    sentences = teacher_sentence_folders & student_sentence_folders
    print(f"Sentence folders in common: {sentences}")

    sample_paths = defaultdict(lambda: defaultdict(list))
    for sentence in sentences:
        sentence_folder_path = os.path.join(teacher_base_folder, sentence)
        sample_paths[sentence]["teacher"] = [os.path.join(sentence_folder_path, fname)
                                             for fname in
                                             os.listdir(sentence_folder_path)]
        sentence_folder_path = os.path.join(student_base_folder, sentence)
        sample_paths[sentence]["student"] = [os.path.join(sentence_folder_path, fname)
                                             for fname in
                                             os.listdir(sentence_folder_path)]

    distances_by_sentence = defaultdict(list)
    for sentence in sample_paths:
        for teacher_wav_filename in sample_paths[sentence]["teacher"]:
            for student_wav_filename in sample_paths[sentence]["student"]:
                # In case we want to compare a source with itself
                if teacher_wav_filename != student_wav_filename:
                    try:
                        # Hide the prints
                        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                            distance = compare(teacher_wav_filename,
                                               student_wav_filename,
                                               show_graphs=False)
                        # print(f"teacher_wav_filename={teacher_wav_filename} "
                        #       f"student_wav_filename={student_wav_filename} "
                        #       f"distance={distance}")
                        if np.isnan(distance):
                            sys.stderr.write(f"Invalid distance for {teacher_wav_filename} and "
                                             f"{student_wav_filename} (could match the signals)")
                        else:
                            distances_by_sentence[sentence].append(distance)
                    except:
                        sys.stderr.write(
                            f"Error when comparing {teacher_wav_filename} "
                            f"and {student_wav_filename}\n")
                        pass

    print("===============================================================")
    print("SUMMARY\n")
    print(f"Compared {teacher_base_folder} (teacher) "
          f"and {student_base_folder} (student)")
    print(f"Sentences found for both sources: {len(sentences)}")

    distances = []
    for values in distances_by_sentence.values():
        distances.extend(values)
    print("Distances stats:")
    print(f"mean: {np.mean(distances)}, std: {np.std(distances)}")

    if stats_filepath:
        stats = {
            "teacher_base_folder": teacher_base_folder,
            "student_base_folder": student_base_folder,
            "distances_by_sentence": distances_by_sentence,
        }
        with open(stats_filepath, 'a') as f:
            f.write(json.dumps(stats))


if __name__ == "__main__":
    app()
