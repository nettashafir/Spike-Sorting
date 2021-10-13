import numpy as np
import scipy.io as spio

from Constants import *


def find_waveform(record, peaks, before, after):
    """
    Extract all the spike waveforms around given peaks.
    :param record: The extracellular record to extract the waveforms from.
    :param peaks: The peaks of the spikes, in the middle of the waveforms.
    :param before: The data points to take before each peak.
    :param after: The data points to take after each peak.
    :return: An array of array, each one represents a waveform os a spike/
    """
    spikes = np.empty((0, SPIKE_LENGTH))
    for i in range(len(peaks)):
        peak = int(peaks[i] * kHz)
        spikes = np.append(spikes, [record[peak - before:peak + after + 1]], axis=0)
    return spikes.mean(axis=0)


def load_data_from_matlab_file(path="true_data\\data4adi.mat"):
    """
    :param path: Path of the matlab file contains the record and the peaks.
    :return: A 1D numpy array of the record, 2D numpy array of the waveforms and 1D numpy array of the peak times.
    """
    m = spio.loadmat(path)
    recording1 = m['dall'][0][0]
    record = recording1[0][0]  # (4,546,017, )
    spike_times = recording1[3][0][0]  # a tuple of 4 arrays in the shape of (1, many)

    peaks1 = (spike_times[0][0] * kHz).astype(int)  # (2477, )
    peaks2 = (spike_times[1][0] * kHz).astype(int)   # (1927, )
    peaks3 = (spike_times[2][0] * kHz).astype(int)   # (1075, )
    peaks4 = (spike_times[3][0] * kHz).astype(int)   # (120, )
    peak_times = [peaks1, peaks2, peaks3, peaks4]

    waveform1 = find_waveform(record, peaks1, 37, 27)
    waveform2 = find_waveform(record, peaks2, 38, 26)
    waveform3 = find_waveform(record, peaks3, 42, 22)
    waveform4 = find_waveform(record, peaks4, 35, 29)
    single_unit_waveforms = [waveform1, waveform2, waveform3, waveform4]

    return record, single_unit_waveforms, peak_times,


def load_data_from_npy_files(record_path=TRUE_RECORD_PATH,
                             single_unit_waveform_paths=TRUE_SINGLE_UNITS_WAVEFORMS_PATHS,
                             peak_times_paths=TRUE_PEAK_TIMES_PATHS):
    """
    :param record_path: The extracellular record, in a length of ms*kHz
    :param single_unit_waveform_paths: The shape of the spike of the single neurons behind the record.
    :param peak_times_paths: in ms
    :return: A 1D numpy array of the record, 2D numpy array of the waveforms and 1D numpy array of the peak times.
    """
    record = np.array(np.load(record_path), dtype=float)
    single_unit_waveforms = list(map(np.load, single_unit_waveform_paths))
    peak_times = [(np.array(x) * kHz).astype(int) for x in list(map(np.load, peak_times_paths))]

    return record, single_unit_waveforms, peak_times


# data = np.load("data.npy")
#
# # Plot the data conveniently
# fig = go.Figure(layout=go.Layout(
#     title=r"$\text{Samples}$",
#     xaxis=dict(title=r"$\text{ms}$"),
#     yaxis=dict(title=r"$\text{mV}$")))
# fig.add_traces([go.Scatter(y=data)])
# fig.update_layout(width=1200, height=600)
# fig.show()
#
# spikes, peaks = hf.get_waveforms(data)  # 77 spikes with threshold of 20,000
# for i in range(20, 30, 2):
#     plt.plot(spikes[i])
#     plt.plot(spikes[i+1])
#     plt.grid()
#     plt.show()