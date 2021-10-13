import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm
from Constants import *

from Spike import Spike


def get_waveforms(samples, peaks=None):
    """
    Extract the waveforms that within the samples, aligned around the
    positive peaks (maximum points).
    :param samples: The data from which extract the peaks.
    :param peaks: (optional) Array of peaks from prior knowledge, e.g. from
                  prior smoothing of the data.
    :return: Array of arrays, which one is a waveform.
    """
    if peaks is None:
        peaks = find_peaks(samples, height=THRESHOLD)[0]
        peaks = peaks[np.logical_and(POINTS_BEFORE_PEAK <= peaks,
                                     peaks < len(samples) - POINTS_AFTER_PEAK)]
    spikes = np.empty((0, SPIKE_LENGTH))
    for peak in peaks:
        spikes = np.append(spikes, [
            samples[peak - POINTS_BEFORE_PEAK:peak + POINTS_AFTER_PEAK + 1]],
                           axis=0)
    return spikes, peaks


def get_waveforms_from_batch(samples_batch):
    """
    Extract the waveforms from batch of samples.
    :return: Array of arrays, which one is a waveform.
    """
    all_spikes = np.empty((0, POINTS_BEFORE_PEAK + POINTS_AFTER_PEAK + 1))
    all_peaks = np.empty(0)
    for samples in samples_batch:
        spikes, peaks = get_waveforms(samples)
        all_spikes = np.append(all_spikes, spikes, axis=0)
        all_peaks = np.append(all_peaks, peaks)
    return all_spikes, all_peaks


def does_contain_synthetic(cluster):
    """
    Tells if the cluster contains at least one synthetic waveforms, i.e
    waveform that is created by overlapping two single-unit waveforms or
    waveform which is a single-unit waveform itself.
    :return: True if a synthetic waveform exists in the cluster, False else.
    """
    for spike in cluster:
        if spike.type is not WaveForms.overlap:
            return True
    return False


def find_best_match(spike: Spike, cluster, minimal_corr=MIN_CORR):
    """
    Finds the synthetic waveform in the cluster that best match a given overlap
    waveform in the cluster.
    :param spike: The overlap waveform to find the best match to it.
    :param cluster: The cluster of the spike.
    :param minimal_corr: The minimal correlation needed to recognize to spike as similar
    :return: True if found a synthetic spike with positive correlation with
             the given spike, False otherwise.
    """
    best_match = None
    best_corr = minimal_corr

    for curr_spike in cluster:
        if curr_spike.peak_time != spike.peak_time and \
                curr_spike.type is not WaveForms.overlap:
            curr_corr = np.corrcoef(spike.waveform, curr_spike.waveform)[0][1]
            if curr_corr > best_corr:
                best_match = curr_spike
                best_corr = curr_corr

    if best_match is None:
        return False
    if best_match.type is WaveForms.synthetic:
        spike.single_unit1 = best_match.single_unit1
        spike.single_unit2 = best_match.single_unit2
    else:  # The best match is a single unit
        spike.single_unit1 = best_match
        spike.single_unit2 = None
    return True


def corr_helper(spike1, spike2):
    """
    Takes two overlap waveforms that identified with the same single-unit
    waveform and indicates which of them has higher correlation with the
    single-unit.
    :param spike1: The first spike to compare.
    :param spike2: Te second spike to compare.
    :return: True if the first spike has higher correlation with the common
             single-unit, False otherwise.
    """
    if spike1.single_unit1.code != spike2.single_unit1.code:
        raise ValueError("The two spikes should come from the same "
                         "single unit")
    corr1 = np.corrcoef(spike1.waveform, spike1.single_unit1.waveform)[0][1]
    corr2 = np.corrcoef(spike2.waveform, spike2.single_unit1.waveform)[0][1]
    return corr1 > corr2


def reconstruct_single_unit_record(single_unit_waveform, peaks, record_len=ms * kHz):
    """
    Builds a record of firing of a single neuron.
    :param single_unit_waveform: The samples which creates the shape of the single-unit's neuron.
    :param peaks: The peaks indices of the single-unit.
    :param record_len: The amount of samples of the extracellular record.
    :return: Samples of estimated firing of a single neuron.
    """
    samples = np.zeros(record_len)
    samples[peaks] = 1
    samples = np.convolve(samples, single_unit_waveform, mode="same")
    return samples


def loss(src_peaks, output_peaks, tol=1, record_len=ms * kHz):
    """
    A loss function for the algorithm: around every peak af spike it draw a
    gaussian (normalized to have 1 at the peak), and sum the absolute values
    of the differences between the between the peaks of the output, and the
    peaks (padded with gaussian) of the source data. The loss is this sum
    divided by the number of the output peaks.
    :param src_peaks: The peaks of the source.
    :param output_peaks: The peak of the output of the program.
    :param tol: Tolerance. A number in the range (0,1] that determine the
           gaussian's width. As it grows, 2 peaks in the src and the output
           that close to each other in time will get more points of accuracy.
    :param record_len: The amount of samples of the extracellular record.
    :return: Two numbers in the range [0,1], which are the FPR (the number of
             spikes in the output that doesn't appears in src, weighted with
             the gaussian) and the FNR (appears in src but not in output).
    """
    if not 0 < tol <= 1:
        raise ValueError("Tolerance should be in the range (0,1]")
    gaussian_width = REFRACTORY_PERIOD * kHz // 2 - 1
    x = np.linspace(norm.ppf(0.00001), norm.ppf(0.99999), gaussian_width)
    y = norm.pdf(x, scale=tol) / norm.pdf(x, scale=tol)[gaussian_width // 2]

    if len(output_peaks) > 0 and len(src_peaks) > 0:
        src = np.zeros(record_len)
        src[src_peaks] = 1
        src = np.convolve(src, y, mode="same")
        FPR = np.sum(1 - src[output_peaks]) / len(output_peaks)

        output = np.zeros(record_len)
        output[output_peaks] = 1
        output = np.convolve(output, y, mode="same")
        FNR = np.sum(1 - output[src_peaks]) / len(src_peaks)

    elif len(output_peaks) == 0:
        if len(src_peaks) == 0:
            FPR, FNR = 0, 0
        else:
            FPR, FNR = 0, 1
    else:
        FPR, FNR = 1, 0

    if FPR < 0 or FPR > 1 or FNR < 0 or FNR > 1:
        raise ValueError("All errors should be in the range [0,1]")

    return FPR, FNR


def find_overlaps_sizes(spike_times1, spike_times2, record=ms*kHz):
    """
    For every spike in both single-units, count the number of points of overlap
    with another spike from the other single-unit.
    :param spike_times1: Indices of the times of spikes of the first single units.
    :param spike_times2: Indices of the times of spikes of the second single units.
    :return: Two arrays corresponds to the input arrays, in each slot the
             number of records of overlap between a spike and other spikes from
             the another single-units.
    """
    intervals1 = np.full(record, 0)  # will have 1 in the range of a spike
    intervals2 = np.full(record, 0)

    for idx in spike_times1:
        intervals1[idx:idx + SPIKE_LENGTH] = 1
    for idx in spike_times2:
        intervals2[idx:idx + SPIKE_LENGTH] = 1

    intervals = intervals1 + intervals2
    overlap_sizes_unit1 = np.empty(len(spike_times1))  # count the overlap pts
    overlap_sizes_unit2 = np.empty(len(spike_times2))
    for i in range(len(spike_times1)):
        overlap_sizes_unit1[i] = \
            np.count_nonzero(intervals[spike_times1[i]:
                                       spike_times1[i] + SPIKE_LENGTH] == 2)
    for i in range(len(spike_times2)):
        overlap_sizes_unit2[i] = \
            np.count_nonzero(intervals[spike_times2[i]:
                                       spike_times2[i] + SPIKE_LENGTH] == 2)

    return overlap_sizes_unit1, overlap_sizes_unit2

# delete: manual convolution
# for time in src_peaks:
#     if time < SPIKE_LENGTH // 2:
#         src[0:time + SPIKE_LENGTH//2 + 1] = np.maximum(
#             src[0:time + SPIKE_LENGTH//2 + 1],
#             y[SPIKE_LENGTH//2 - time:])
#     elif time > ms * kHz - SPIKE_LENGTH//2 - 1:
#         src[time - SPIKE_LENGTH//2:ms * kHz] = np.maximum(
#             src[time - SPIKE_LENGTH//2:ms * kHz],
#             y[:SPIKE_LENGTH//2 + (ms * kHz - time)])
#     else:
#         src[time - SPIKE_LENGTH//2:time + SPIKE_LENGTH//2 + 1] = np.maximum(
#             src[time - SPIKE_LENGTH//2:time + SPIKE_LENGTH//2 + 1], y)
