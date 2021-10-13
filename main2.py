import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import HelperFunctions as hf
from MatlabData import load_data_from_npy_files
from Simulator import generate_simulated_data
from Clustering import cluster, fix_refractory_violations, divide_to_single_units
from SpikeFactory import create_all_synthetic_spikes, create_overlap_spikes
from Spike import Spike
from Visualization import dummy, smart
from Constants import *


# ############################# #
# ###     Main function     ### #
# ############################# #

def run(record, single_neuron_waveforms, src_peaks=None, _plot=False):
    if src_peaks is not None and len(single_neuron_waveforms) != len(src_peaks):
        raise ValueError("The amount of waveforms should correspond to"
                         " the amount of peaks arrays")

    # Create an array of Spike objects - single neurons, synthetic spikes from those single neurons and spikes detected
    # from the record (overlap spikes)
    single_unit_spikes = list(
        map(lambda waveform, code: Spike(waveform, code=code), single_neuron_waveforms, SingleUnits))
    overlap_waveforms, overlaps_peaks = hf.get_waveforms(record)
    overlap_spikes = create_overlap_spikes(overlap_waveforms, overlaps_peaks)
    synthetic_spikes = create_all_synthetic_spikes(single_unit_spikes)
    all_spikes = np.concatenate((single_unit_spikes, synthetic_spikes, overlap_spikes))
    all_waveforms = np.array([spike.waveform for spike in all_spikes])  # 2D array of the waveforms of the spikes only
    print(f"Number of spikes: {len(all_waveforms)}\n")

    # Cluster the spikes
    clusters = cluster(all_waveforms, all_spikes)
    print("OK: Clustering")

    # Divide the overlap spikes according to the most dominant single-unit that creates them
    single_unit_clusters, multi_unit_cluster = divide_to_single_units(clusters)
    print("OK: Matching single units")

    # Check for refractory period violations, and remove spikes that too close to other spikes
    fix_refractory_violations(single_unit_clusters, multi_unit_cluster)
    print("OK: removing refractory violations")

    # Reconstruct the single unit clusters
    output_peaks = [np.array([spike.peak_time for spike in clu]) for clu in single_unit_clusters]
    # When using the default true data, it is recommended to execute the next lines:
    # output_peaks[0] += 5
    # output_peaks[2] += 10

    if src_peaks is None:
        return output_peaks

    if _plot:
        src = list(
            map(hf.reconstruct_single_unit_record, single_neuron_waveforms, src_peaks, [len(record)] * len(record)))
        output = list(
            map(hf.reconstruct_single_unit_record, single_neuron_waveforms, output_peaks, [len(record)] * len(record)))
        print("OK: Reconstruct single-units record")
        for i in range(len(single_neuron_waveforms)):
            smart([src[i]], title=f"Original unit {i + 1}")
            smart([output[i]], title=f"Estimated unit {i + 1}")

    # Calculate the losses (FPR and FNR)
    losses = list(map(hf.loss, src_peaks, output_peaks, [1] * len(src_peaks), [len(record)] * len(src_peaks)))
    print("\nResults:")
    for i in range(len(single_neuron_waveforms)):
        print(f"unit {i + 1}:\n    "
              f"original spikes {len(src_peaks[i])}, "
              f"estimated {len(output_peaks[i])}\n    "
              f"FPR {losses[i][0]}, FNR {losses[i][1]}")
    if len(single_neuron_waveforms) == 2:
        print(f"Correlation = {np.corrcoef(single_neuron_waveforms[0], single_neuron_waveforms[1])[0][1]}")

    return losses


record, single_unit_waveforms, peak_times = generate_simulated_data()
run(record, single_unit_waveforms, peak_times, _plot=True)


# ############################## #
# ###   Performances tests   ### #
# ############################## #

def fpr_fnr_as_func_of_noise():
    """
    (!) NOTICE - THIS FUNCTION IS NOT UPDATE AND IS NEEDED TO BE FIXED (!)
    """
    noise = np.linspace(0, 1, 15)
    y = list(map(generate_simulated_data, [SIMULATOR_SINGLE_UNITS_WAVEFORMS_PATHS2] * len(noise), noise))
    x = np.array(list(map(run, [s[0] for s in y], [s[1] for s in y], [s[2] for s in y])), dtype=float)
    times = 10
    for i in range(times):
        y = list(map(generate_simulated_data, [SIMULATOR_SINGLE_UNITS_WAVEFORMS_PATHS2] * len(noise), noise))
        x += np.array(list(map(run, [s[0] for s in y], [s[1] for s in y], [s[2] for s in y])), dtype=float)
        print(i)
    x /= times
    print(x)
    np.save("numpy_arrays\\errors_as_func_of_noise_high_correlation", x)
    dummy(x[:, 0], x=noise, title="FPR of neuron1, as a function of noise variance")
    dummy(x[:, 1], x=noise, title="FNR of neuron1, as a function of noise variance")
    dummy(x[:, 2], x=noise, title="FPR of neuron2, as a function of noise variance")
    dummy(x[:, 3], x=noise, title="FNR of neuron2, as a function of noise variance")


def fnr_as_func_of_overlap(_save=False, _plot=True):
    """
    (!) NOTICE - THIS FUNCTION IS FITTED ONLY FOR 2 UNITS; ADJUSTMENT IS NEEDED FOR GENERAL AMOUNT OF NEURONS (!)
    Calculate the FNR of the each neuron in a simulated data, as a function of the size of overlap
    with spikes of another neurons.
    :param _save: If true, run the algorithm and save the results. Otherwise, load the results from file.
    :param _plot: If true, plot the data.
    :return: A tuple of two arrays, one for each neuron.
    """
    noise = np.linspace(0.05, 0.5, 10)
    if _save:
        fnr_unit1 = np.empty([0, SPIKE_LENGTH + 1])
        fnr_unit2 = np.empty([0, SPIKE_LENGTH + 1])
        times = 10
        for n in noise:
            print(f"noise = {n}")  # for tests only
            fnr1 = np.zeros(SPIKE_LENGTH + 1)
            fnr2 = np.zeros(SPIKE_LENGTH + 1)
            for i in range(times):
                print(f"    time {i}")  # for tests only
                record, single_unit_waveforms, src_peaks = generate_simulated_data(noise_sd=n)
                output_peaks = run(record, single_unit_waveforms)
                overlap_sizes_unit1, overlap_sizes_unit2 = hf.find_overlaps_sizes(src_peaks[0], src_peaks[1])
                fnr1 += np.array(list(map(hf.loss, [src_peaks[0][overlap_sizes_unit1 == i]
                                                    for i in range(SPIKE_LENGTH + 1)],
                                          [output_peaks[0]] * (SPIKE_LENGTH + 1))))[:, 1]
                fnr2 += np.array(list(map(hf.loss, [src_peaks[1][overlap_sizes_unit2 == i]
                                                    for i in range(SPIKE_LENGTH + 1)],
                                          [output_peaks[1]] * (SPIKE_LENGTH + 1))))[:, 1]
            fnr1 /= times
            fnr2 /= times
            fnr_unit1 = np.append(fnr_unit1, [fnr1], axis=0)
            fnr_unit2 = np.append(fnr_unit2, [fnr2], axis=0)

        np.save("numpy_arrays\\fnr_of_unit1_low_correlation", fnr_unit1)
        np.save("numpy_arrays\\fnr_of_unit2_low_correlation", fnr_unit2)

    else:
        fnr_unit1 = np.load("numpy_arrays\\fnr_of_unit1_high_correlation.npy")
        fnr_unit2 = np.load("numpy_arrays\\fnr_of_unit2_high_correlation.npy")

    if _plot:
        labels = [f"noise sd  = {n}" for n in noise]
        fig = make_subplots(rows=2, cols=1, subplot_titles=["spike22200", "artifact1"])
        for fnr_1, label in list(zip(fnr_unit1, labels)):
            fig.append_trace(go.Scatter(x=list(range(41)), y=fnr_1, name=label, showlegend=True), row=1, col=1)
        for fnr_2, label in list(zip(fnr_unit2, labels)):
            fig.append_trace(go.Scatter(x=list(range(41)), y=fnr_2, name=label, showlegend=True), row=2, col=1)
        fig.update_layout(title_text=f"FNR as a function of overlap size")
        fig.show()

    return fnr_unit1, fnr_unit2


def false_positive_as_func_of_overlap(_save=False, _plot=True):
    """
    (!) NOTICE - THIS FUNCTION IS FITTED ONLY FOR 2 UNITS; ADJUSTMENT IS NEEDED FOR GENERAL AMOUNT OF NEURONS (!)
    Calculate the false positive of the each neuron in a true data, as a function of the size of overlap
    with spikes of another neurons.
    :param _save: If true, run the algorithm and save the results. Otherwise, load the results from file.
    :param _plot: If true, plot the false positive of each neuron as a function of the overlap percentage.
    :return: None
    """
    if _save:
        record, waveforms, src_peak_times = load_data_from_npy_files()
        output_peaks_times = run(record, waveforms, _plot=False)
        output_peaks_times[0] += 5
        output_peaks_times[2] += 10
        src_peak_times = np.sort(np.concatenate(src_peak_times))  # Flatten the times
        ans = np.zeros((2, 11))  # An array of the false positive spikes of neuron 1 and neuron 3 from the true data
        no_overlap_a = np.empty((0, SPIKE_LENGTH))  # The waveforms of the false positive spikes from the first
        # neuron that have no overlap with true spikes
        overlap_a = np.empty((0, SPIKE_LENGTH))  # The waveforms of the false positive spikes form the first neuron
        # that have 30%-50% overlap with true spikes

        no_overlap_b = np.empty((0, SPIKE_LENGTH))
        overlap_b = np.empty((0, SPIKE_LENGTH))
        for i in range(len(output_peaks_times[0])):
            print(f"i = {i}")
            if np.intersect1d(range(output_peaks_times[0][i] - 5, output_peaks_times[0][i] + 5),
                              src_peak_times).size == 0:
                # so this peak is false positive
                if np.intersect1d(
                        range(output_peaks_times[0][i] - SPIKE_LENGTH, output_peaks_times[0][i] + SPIKE_LENGTH),
                        src_peak_times).size == 0:
                    ans[0][0] += 1
                    no_overlap_a = np.append(no_overlap_a, [
                        record[
                        output_peaks_times[0][i] - SPIKE_LENGTH // 2:output_peaks_times[0][i] + SPIKE_LENGTH // 2 + 1]],
                                             axis=0)
                    continue
                for j in range(1, 11):
                    intersection = np.intersect1d(
                        [*range(output_peaks_times[0][i] - int(((11 - j) / 10) * SPIKE_LENGTH),
                                output_peaks_times[0][i] - int(((10 - j) / 10) * SPIKE_LENGTH)),
                         *range(output_peaks_times[0][i] + int(((10 - j) / 10) * SPIKE_LENGTH),
                                output_peaks_times[0][i] + int(((11 - j) / 10) * SPIKE_LENGTH))],
                        src_peak_times).size
                    if intersection > 0:
                        ans[0][j] += intersection
                        if j in [3, 4, 5]:
                            overlap_a = np.append(overlap_a, [record[output_peaks_times[0][i] - SPIKE_LENGTH // 2:
                                                                     output_peaks_times[0][
                                                                         i] + SPIKE_LENGTH // 2 + 1]],
                                                  axis=0)
        for i in range(len(output_peaks_times[2])):
            print(f"j = {i}")
            if np.intersect1d(range(output_peaks_times[2][i] - 5, output_peaks_times[2][i] + 5),
                              src_peak_times).size == 0:
                # so this peak is false positive
                if np.intersect1d(
                        range(output_peaks_times[2][i] - SPIKE_LENGTH, output_peaks_times[2][i] + SPIKE_LENGTH),
                        src_peak_times).size == 0:
                    ans[1][0] += 1
                    no_overlap_b = np.append(no_overlap_b, [record[output_peaks_times[2][i] - SPIKE_LENGTH // 2:
                                                                   output_peaks_times[2][i] + SPIKE_LENGTH // 2 + 1]],
                                             axis=0)
                    continue
                for j in range(1, 11):
                    intersection = np.intersect1d(
                        [*range(output_peaks_times[2][i] - int(((11 - j) / 10) * SPIKE_LENGTH),
                                output_peaks_times[2][i] - int(((10 - j) / 10) * SPIKE_LENGTH)),
                         *range(output_peaks_times[2][i] + int(((10 - j) / 10) * SPIKE_LENGTH),
                                output_peaks_times[2][i] + int(((11 - j) / 10) * SPIKE_LENGTH))],
                        src_peak_times).size
                    if intersection > 0:
                        ans[1][j] += intersection
                        if j in [3, 4, 5]:
                            overlap_b = np.append(overlap_b, [record[output_peaks_times[2][i] - SPIKE_LENGTH // 2:
                                                                     output_peaks_times[2][i] + SPIKE_LENGTH // 2 + 1]],
                                                  axis=0)
        np.save("numpy_arrays\\false_positive_of_unit1", ans[0])
        np.save("numpy_arrays\\false_positive_of_unit2", ans[1])
        np.save("numpy_arrays\\no_overlap_a", no_overlap_a)
        np.save("numpy_arrays\\no_overlap_b", no_overlap_b)
        np.save("numpy_arrays\\overlap_a", overlap_a)
        np.save("numpy_arrays\\overlap_b", overlap_b)

    else:
        ans = np.empty((0, 11))
        unit1 = np.load("numpy_arrays\\false_positive_of_unit1.npy")
        unit2 = np.load("numpy_arrays\\false_positive_of_unit2.npy")
        ans = np.append(ans, [unit1], axis=0)
        ans = np.append(ans, [unit2], axis=0)
        no_overlap_a = np.load("numpy_arrays\\no_overlap_a.npy")
        no_overlap_b = np.load("numpy_arrays\\no_overlap_b.npy")
        overlap_a = np.load("numpy_arrays\\overlap_a.npy")
        overlap_b = np.load("numpy_arrays\\overlap_b.npy")

    if _plot:
        smart([ans[0], ans[1]], "Number of false positives as a function of overlap size")
        smart(overlap_a, "All the false positive spikes in unit 1 that overlap 30%-50% with a true spike")
        smart(overlap_b, "All the false positive spikes in unit 3 that overlap 30%-50% with a true spike")
        smart(no_overlap_a, "All the false positive spikes in unit 1 that has no overlap with a true spike")
        smart(no_overlap_b, "All the false positive spikes in unit 3 that has no overlap with a true spike")


def cross_correlation_function():
    """
    Plot the CCF (cross correlation function) between every two neurons in the true data, once according to the "true"
    spike times, and another time according to the spike times in the output of the program.
    :return: None
    """
    record, single_unit_waveforms, src_peak_times = load_data_from_npy_files()
    output_peak_times = run(record, single_unit_waveforms)
    for i in range(len(src_peak_times) - 1):
        for j in range(i+1, len(src_peak_times)):
            src_spikes1 = np.zeros(len(record))
            src_spikes1[src_peak_times[i]] = 1
            src_spikes2 = np.zeros(len(record))
            src_spikes2[src_peak_times[j]] = 1

            src_corr = correlate(src_spikes1, src_spikes2, mode="same")
            millisecond = len(src_corr) // (2*kHz)
            plt.plot(np.linspace(-millisecond, millisecond, len(src_corr)), src_corr)
            plt.title(f"Cross correlation between unit {i+1} and unit {j+1}, from source")
            plt.xlabel("time relative to spike (milliseconds)")
            plt.ylabel("firing rate (Hz)")
            plt.show()

            output_spikes1 = np.zeros(len(record))
            output_spikes1[output_peak_times[i]] = 1
            output_spikes2 = np.zeros(len(record))
            output_spikes2[output_peak_times[j]] = 1

            output_corr = correlate(output_spikes1, output_spikes2, mode="same")
            millisecond = len(output_corr) // (2 * kHz)
            plt.plot(np.linspace(-millisecond, millisecond, len(output_corr)), output_corr)
            plt.title(f"Cross correlation between unit {i + 1} and unit {j + 1}, from output")
            plt.xlabel("time relative to spike (milliseconds)")
            plt.ylabel("firing rate (Hz)")
            plt.show()

# ### Not in use ### #
# def subtract_waveforms(record, waveforms, peak_times):
#     if len(waveforms) != len(peak_times):
#         raise ValueError("The amount of waveforms should correspond to the amount of peaks arrays")
#
#     for i, g in zip(range(len(peak_times)), [5, 6, 10, 3]):
#         for peak in peak_times[i]:
#             record[peak - POINTS_BEFORE_PEAK:peak + POINTS_AFTER_PEAK + 1 - g] -= waveforms[i][g:]
#     return record
