import contextlib
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import typer
from dtw import dtw, rabinerJuangStepPattern

from onsei.utils import cleanup_pitch_freq, \
    plot_pitch_and_spectro, \
    draw_intensity, ts_sequences_to_index, replacing_zero_by_nan, \
    znormed
from onsei.vad import detect_voice_with_webrtcvad

app = typer.Typer()


@app.command()
def view(wav_filename: str) -> None:
    """
    Visualize a recording
    """
    # Load the wav file and retrieve the pitch and intensity using Praat bindings
    snd = parselmouth.Sound(wav_filename)
    pitch = snd.to_pitch()
    pitch_freq = pitch.selected_array['frequency']
    pitch_freq_filtered, mean_pitch_freq, std_pitch_freq = cleanup_pitch_freq(
        pitch_freq)
    intensity = snd.to_intensity()

    # Run a simple voice detection algorithm to find where the speech starts and ends
    # vad, begin_ts, end_ts = detect_voice_activity_from_pitch(pitch)
    vad_ts, vad_is_speech, begin_ts, end_ts = detect_voice_with_webrtcvad(wav_filename)

    plt.figure()
    plt.subplot(211)
    plot_pitch_and_spectro(snd, pitch.xs(),
                           pitch_freq_filtered)
    plt.subplot(212)
    # draw_intensity(intensity)
    # plt.plot(pitch.xs(), vad * pitch_freq)
    plt.plot(vad_ts, vad_is_speech)
    plt.xlim(pitch.xs()[0], pitch.xs()[-1])
    plt.title("VAD Teacher")
    plt.show()


@app.command()
def compare(teacher_wav_filename: str, student_wav_filename: str,
            show_graphs: bool = True, notebook: bool = False) -> float:
    """
    Compare a teacher and student recording of the same sentence
    """

    print(f"Comparing {teacher_wav_filename} with {student_wav_filename}")

    # Load the wav files and retrieve the pitch and intensity using Praat bindings
    snd_teacher = parselmouth.Sound(teacher_wav_filename)
    snd_student = parselmouth.Sound(student_wav_filename)

    pitch_teacher = snd_teacher.to_pitch()
    pitch_student = snd_student.to_pitch()

    pitch_freq_teacher = pitch_teacher.selected_array['frequency']
    pitch_freq_student = pitch_student.selected_array['frequency']

    pitch_freq_teacher_filtered, mean_pitch_freq_teacher, std_pitch_freq_teacher = cleanup_pitch_freq(
        pitch_freq_teacher)
    pitch_freq_student_filtered, mean_pitch_freq_student, std_pitch_freq_student = cleanup_pitch_freq(
        pitch_freq_student)

    intensity_teacher = snd_teacher.to_intensity()
    intensity_student = snd_student.to_intensity()

    # Run a simple voice detection algorithm to find where the speech starts and ends
    #vad_teacher, begin_ts_teacher, end_ts_teacher = detect_voice_activity_from_pitch(
    #    pitch_teacher)
    vad_ts_teacher, vad_is_speech_teacher, begin_ts_teacher, end_ts_teacher = detect_voice_with_webrtcvad(teacher_wav_filename)
    begin_idx_teacher, end_idx_teacher = ts_sequences_to_index(
        [begin_ts_teacher, end_ts_teacher], intensity_teacher.xs())

    # vad_student, begin_ts_student, end_ts_student = detect_voice_activity_from_pitch(
    #     pitch_student)
    vad_ts_student, vad_is_speech_student, begin_ts_student, end_ts_student = detect_voice_with_webrtcvad(
        student_wav_filename)
    begin_idx_student, end_idx_student = ts_sequences_to_index(
        [begin_ts_student, end_ts_student], intensity_student.xs())

    if show_graphs:
        plt.figure()
        plt.subplot(211)
        plot_pitch_and_spectro(snd_teacher, pitch_teacher.xs(),
                               pitch_freq_teacher_filtered, title="Teacher")
        plt.subplot(212)
        plot_pitch_and_spectro(snd_student, pitch_student.xs(),
                               pitch_freq_student_filtered, title="Student")
        plt.show(block=False)

        plt.figure()
        plt.subplot(211)
        draw_intensity(intensity_teacher)
        plt.plot(vad_ts_teacher,
                 vad_is_speech_teacher * np.max(intensity_teacher))
        # plt.plot(pitch_teacher.xs(),
        #          vad_teacher * np.max(intensity_teacher))
        plt.title("VAD Teacher")
        plt.subplot(212)
        draw_intensity(intensity_student)
        plt.plot(vad_ts_student,
                 vad_is_speech_student * np.max(intensity_student))
        # plt.plot(pitch_student.xs(),
        #          vad_student * np.max(intensity_student))
        plt.title("VAD Student")
        plt.show(block=False)

    # Align speech sequence using a DTW on intensity

    # x is the query (which we will "warp") and y the reference
    x = znormed(intensity_student.values[0, begin_idx_student:end_idx_student])
    y = znormed(intensity_teacher.values[0, begin_idx_teacher:end_idx_teacher])

    # Align the Rabiner-Juang type VI-c unsmoothed recursion
    # step_pattern = rabinerJuangStepPattern(6, "c", smoothed=True)
    step_pattern = rabinerJuangStepPattern(4, "c", smoothed=True)
    # step_pattern = "symmetric2"
    align = dtw(x, y, keep_internals=True, step_pattern=step_pattern)
    # align.plot(type="threeway")  # Display the warping curve, i.e. the alignment curve

    # Timestamp for each point in the alignment
    align_ts_student = intensity_student.xs()[begin_idx_student:end_idx_student][
        align.index1]
    align_ts_teacher = intensity_teacher.xs()[begin_idx_teacher:end_idx_teacher][
        align.index2]

    # Intensity and pitch computed by parselmouth do not have the same timestamps,
    # so we mean to find the frames in the pitch signal using the aligned timestamps
    align_idx_pitch_student = ts_sequences_to_index(align_ts_student,
                                                    pitch_student.xs())
    align_idx_pitch_teacher = ts_sequences_to_index(align_ts_teacher,
                                                    pitch_teacher.xs())

    # Align the pitch signals, using the same alignment as for intensity
    aligned_pitch_teacher = replacing_zero_by_nan(
        pitch_freq_teacher_filtered[align_idx_pitch_teacher])
    aligned_pitch_student = replacing_zero_by_nan(
        pitch_freq_student_filtered[align_idx_pitch_student])

    # Compute a distance based on z-score normalized aligned pitches
    zscore_norm_pitch_teacher = (aligned_pitch_teacher - mean_pitch_freq_teacher) \
                                / std_pitch_freq_teacher
    zscore_norm_pitch_student = (aligned_pitch_student - mean_pitch_freq_student) \
                                / std_pitch_freq_student

    distances_ts = []
    distances = []
    for idx, (teacher, student) in enumerate(zip(zscore_norm_pitch_teacher, zscore_norm_pitch_student)):
        if not np.isnan(teacher) and not np.isnan(student):
            distances_ts.append(align_ts_teacher[idx])
            distances.append(abs(teacher - student))
    mean_distance = np.mean(distances)
    print(f"Mean distance: {mean_distance:.2f} "
          f"(smaller means student speech is closer to teacher)")

    # Plot the warped intensity and pitches
    if show_graphs:
        plt.figure()
        plt.subplot(311)
        plt.plot(align_ts_teacher, y[align.index2], 'b-')
        plt.plot(align_ts_teacher, x[align.index1], 'g-')
        plt.title("Aligned student intensity (green) to match teacher (blue)")
        plt.subplot(312)
        # plt.plot(align_ts_teacher, aligned_pitch_teacher, 'b.')
        # plt.plot(align_ts_teacher, aligned_pitch_student, 'g.')
        plt.plot(align_ts_teacher, zscore_norm_pitch_teacher, 'b.')
        plt.plot(align_ts_teacher, zscore_norm_pitch_student, 'g.')
        plt.title("Applying the same alignment on normalized pitch")
        plt.subplot(313)
        plt.plot(distances_ts, distances, 'r.')
        plt.title('Pitch "error"')
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
