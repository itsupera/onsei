import matplotlib.pyplot as plt
import numpy as np
import parselmouth


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


def draw_spec_pitch(snd: parselmouth.Sound, window_length=0.03, maximum_frequency=500, time_step=0.002, frequency_step=20.0):
    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=window_length, maximum_frequency=maximum_frequency, time_step=time_step, frequency_step=frequency_step)
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch, maximum_frequency)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show()  # or plt.savefig("spectrogram_0.03.pdf")

# draw_spec_pitch(snd, window_length=0.03, maximum_frequency=500, time_step=0.01, frequency_step=10.0)


def cleanup_pitch_freq(pitch_sig):
    """
    Remove some outliers from the pitch frequencies
    """
    valid_freqs = pitch_sig[pitch_sig != 0]
    mean = np.mean(valid_freqs)
    std = np.std(valid_freqs)

    min_cut = mean - std * 2.5
    max_cut = mean + std * 2.5
    print(f"Pitch frequencies cleanup: mean is {mean:.2f} Hz, "
          f"keeping values within [{min_cut:.2f} Hz, {max_cut:.2f} Hz]")

    new_sig = []
    for x in pitch_sig:
        if x == 0 or (min_cut <= x <= max_cut):
            new_sig.append(x)
        else:
            new_sig.append(0)

    new_sig = np.array(new_sig)

    # Normalize
    # new_sig = (new_sig - min(new_sig)) / (max(new_sig) - min(new_sig))

    return new_sig, mean, std


def plot_pitch_and_spectro(
    snd: parselmouth.Sound,
    pitch_ts: np.ndarray,
    pitch_freq_filtered: np.ndarray,
    window_length=0.03,
    maximum_frequency=500,
    time_step=0.01,
    frequency_step=10.0,
    plot_maximum_frequency=500,
    title=None,
):
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=window_length,
                                                    maximum_frequency=maximum_frequency,
                                                    time_step=time_step,
                                                    frequency_step=frequency_step)
    draw_spectrogram(spectrogram, maximum_frequency=plot_maximum_frequency)

    y = pitch_freq_filtered.copy()
    y[y == 0] = np.nan
    plt.plot(pitch_ts, y, 'o', markersize=5, color='w')
    plt.plot(pitch_ts, y, 'o', markersize=2)
    plt.ylim(0, plot_maximum_frequency)
    plt.ylabel("fundamental frequency [Hz]")

    plt.twinx()
    intensity = snd.to_intensity()
    draw_intensity(intensity)

    plt.xlim([snd.xmin, snd.xmax])

    if title:
        plt.title(title)


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
    print(f"Computing noise level using {len(samples)} samples")
    return np.mean(samples)


def detect_voice_activity_from_intensity(intensity: parselmouth.Intensity) -> tuple:
    noise = get_noise_from_intensity(intensity)
    print(f"Noise level: {noise:.2f} dB")
    sig = intensity.values[0, :]
    mean = np.mean(sig)
    print(f"Mean volume: {mean:.2f} dB")
    vad = np.array([(1.0 if x > (noise + ((mean - noise) / 1.5)) else 0.0) for x in sig])
    begin_idx, end_idx = vad.argmax(), len(vad) - vad[::-1].argmax() - 1
    begin_ts, end_ts = intensity.xs()[begin_idx], intensity.xs()[end_idx]
    return vad, begin_idx, end_idx, begin_ts, end_ts


def detect_voice_activity_from_pitch(pitch: parselmouth.Pitch):
    pitch_freq = pitch.selected_array['frequency']
    vad = pitch_freq > 0
    begin_idx, end_idx = vad.argmax(), len(vad) - vad[::-1].argmax() - 1
    begin_ts, end_ts = pitch.xs()[begin_idx], pitch.xs()[end_idx]
    print(f"Voice detected between {begin_ts:.2f}s and {end_ts:.2f}s")
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