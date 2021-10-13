import numpy as np

from Spike import Spike
from Constants import *


def filter_refractory_violations(spike_time_intervals):
    """
    Filter out the peak times that violate the reasonable refractory periods
    that is known from literature.
    :param spike_time_intervals: Array-like of unfiltered peak times.
    :return: Array of filtered peak times.
    """
    refractory_filter = []
    for i in range(len(spike_time_intervals)):
        if spike_time_intervals[i] > REFRACTORY_PERIOD:
            refractory_filter.append(i)
        elif i < len(spike_time_intervals) - 1:
            spike_time_intervals[i + 1] += spike_time_intervals[i]
    return refractory_filter


def generate_single_unit_record(spikes_times, spike_shape, noise_sd):
    """
    :param spikes_times: The indices of times of the start of the spikes (not the peaks).
    :param spike_shape: Array of the waveform of the spike.
    :param noise_sd: The standard deviation of the noise in the spike.
    :return: Array with the complete record of the single unit.
    """
    record = np.full(ms * kHz, RESTING_POTENTIAL)
    for idx in spikes_times:
        noise = min(max(1-noise_sd, np.random.normal(1, noise_sd**2)),
                    1 + noise_sd)
        noised_spike = spike_shape * noise
        record[idx:idx+len(spike_shape)] = noised_spike[:len(record) - idx]
    return record


def generate_single_unit(lam, noise_sd, spike_shape):
    """
    Create fire of a given neuron in a Poisson process.
    :param lam: The parameter of the a Poisson process.
    :param noise_sd: The standard deviation of the noise of the spikes.
    :param spike_shape: The waveform of a single spike.
    :return: The single-unit record, and the times of the peaks of the spikes
    """
    spike_time_intervals = np.random.exponential(lam, NUM_OF_SPIKES_TO_TAKE)

    # Check that the spikes fulls the ms that recorded
    counter = 1
    spike_times = np.add.accumulate(spike_time_intervals)
    refractory_filter = filter_refractory_violations(spike_time_intervals)

    while np.sum(spike_time_intervals) < ms:
        if counter > 10:
            raise RuntimeError(f"Not enough spikes to fill {ms} "
                               f"ms. Pick higher lambda, or take more spikes")
        spike_time_intervals = np.random.exponential(lam, NUM_OF_SPIKES_TO_TAKE)
        refractory_filter = filter_refractory_violations(spike_time_intervals)
        counter += 1

    flt_spike_times = spike_times[refractory_filter]  # filter out the refractory violations
    flt_spike_times = (flt_spike_times * kHz).astype(int)  # Convert the milliseconds to indexes
    flt_spike_times = flt_spike_times[flt_spike_times + POINTS_BEFORE_PEAK < ms * kHz]  # Filter out out-of-range spikes
    record = generate_single_unit_record(flt_spike_times, spike_shape, noise_sd)

    return record, flt_spike_times + POINTS_BEFORE_PEAK


def generate_simulated_data(single_unit_waveforms_paths=SIMULATOR_SINGLE_UNITS_WAVEFORMS_PATHS1, noise_sd=0.2):
    """
    Create simulated data.
    :param single_unit_waveforms_paths: Array of paths to numpy arrays which waveforms of single neurons.
    :param noise_sd: Standard deviation of the noise in the spike amplitude. should be in the range [0,1]
    :return: The simulated record, 2D array of waveforms (each one is array) and 2D array of peak times, one
             for each single unit.
    """
    if noise_sd < 0 or noise_sd > 1:
        raise ValueError("Standard deviation must be between 0 and 1")
    if len(single_unit_waveforms_paths) != len(LAMBDAS):
        raise ValueError("The amount of waveforms isn't legal")

    single_unit_waveforms = list(map(np.load, single_unit_waveforms_paths))

    src = list(map(generate_single_unit, LAMBDAS, [noise_sd]*len(single_unit_waveforms), single_unit_waveforms))
    record = np.sum([s[0] for s in src], axis=0)
    src_peaks = [s[1] for s in src]

    return record, single_unit_waveforms, src_peaks


# ########    Not in use    ########## #

# ##  Brian Simulator  ## #

# from brian2 import *
#
# def generate_spikes(threshold=1, factor=1, ref=0, _plot=False):
#     start_scope()
#     sigma = 0.5
#
#     eqs = '''
#     dv/dt = (I-v)/tau : 1 (unless refractory)
#     I : 1
#     tau : second
#     '''
#
#     G = NeuronGroup(2, eqs, threshold='v>-50*threshold', reset='v = -70', refractory=ref * ms, method='exact')
#     G.v = [-70, -70]
#     G.I = [3*factor, 4*factor]
#     G.tau = [10, 10]*ms
#
#     # Comment these two lines out to see what happens without Synapses
#     # S = Synapses(G, G, on_pre='v_post += 0.2')
#     # S.connect(i=0, j=1)
#
#     M = StateMonitor(G, 'v', record=True)
#     S = SpikeMonitor(G)
#
#     run(50*ms)
#
#     if _plot:
#         plot(M.t/ms, M.v[0], label='Neuron 0')
#         plot(M.t/ms, M.v[1], label='Neuron 1')
#         xlabel('Time (ms)')
#         ylabel('v')
#         legend()
#         show()
#
#     return M.v, M.t/ms, S.t/ms
