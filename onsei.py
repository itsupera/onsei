import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import typer
from dtw import dtw, rabinerJuangStepPattern

from utils import cleanup_pitch_freq, detect_voice_activity_from_intensity, \
    plot_pitch_and_spectro, \
    draw_intensity, ts_sequences_to_index, replacing_zero_by_nan

app = typer.Typer()


@app.command()
def compare(teacher_wav_filename: str, student_wav_filename: str,
            show_graphs: bool = True) -> None:

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
    vad_teacher, begin_idx_teacher, end_idx_teacher, begin_ts_teacher, end_ts_teacher = detect_voice_activity_from_intensity(
        intensity_teacher)
    vad_student, begin_idx_student, end_idx_student, begin_ts_student, end_ts_student = detect_voice_activity_from_intensity(
        intensity_student)

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
        plt.plot(intensity_teacher.xs(),
                 vad_teacher * np.max(intensity_teacher.values[0, :]))
        plt.title("VAD Teacher")
        plt.subplot(212)
        draw_intensity(intensity_student)
        plt.plot(intensity_student.xs(),
                 vad_student * np.max(intensity_student.values[0, :]))
        plt.title("VAD Student")
        plt.show(block=False)

    # Align speech sequence using a DTW on intensity

    # x is the query (which we will "warp") and y the reference
    x = intensity_student.values[0, begin_idx_student:end_idx_student]
    y = intensity_teacher.values[0, begin_idx_teacher:end_idx_teacher]

    # Align the Rabiner-Juang type VI-c unsmoothed recursion
    align = dtw(x, y, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
    # align.plot(type="threeway")  # Display the warping curve, i.e. the alignment curve

    # Plot the warped query along with reference
    if show_graphs:
        plt.figure()
        plt.plot(y, 'b-')
        plt.plot(align.index2, x[align.index1], 'g-')
        plt.title("Teacher and warped student intensity (teacher blue)")
        plt.show(block=False)

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
    distances = []
    for teacher, student in zip(zscore_norm_pitch_teacher, zscore_norm_pitch_student):
        if not np.isnan(teacher) and not np.isnan(student):
            distances.append(abs(teacher - student))
    mean_distance = np.mean(distances)
    print(f"Mean distance: {mean_distance:.2f} "
          f"(smaller means student speech is closer to teacher)")

    # Plot the warped intensity and pitches
    if show_graphs:
        plt.figure()
        plt.subplot(211)
        plt.plot(align_ts_teacher, y[align.index2], 'b-')
        plt.plot(align_ts_teacher, x[align.index1], 'g-')
        plt.title("Aligned student intensity (green) to match teacher (blue)")
        plt.subplot(212)
        plt.plot(align_ts_teacher, aligned_pitch_teacher, 'b.')
        plt.plot(align_ts_teacher, aligned_pitch_student, 'g.')
        plt.title("Applying the same alignment on pitch")
        plt.show(block=False)

        input("Press Enter to close graphs and quit")

    return mean_distance


if __name__ == "__main__":
    app()
