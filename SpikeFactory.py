import numpy as np
from itertools import combinations
from Spike import Spike


def create_synthetic_spikes(unit1, unit2):
    """
    Generates all the synthetic waveforms from 2 single units.
    :param unit1: The first single unit (Spike object).
    :param unit2: The second single unit (Spike object).
    :return: A numpy array with Spike objects of all the overlap waveforms
             between the two single units.
    """
    if len(unit1) != len(unit2):
        raise ValueError("The length of the two units should be equal")

    synth_waveforms = np.array([Spike(waveform=(unit1 + unit2),
                                      single_unit1=unit1, single_unit2=unit2)])

    for i in range(1, len(unit1)):
        z = np.zeros(i)
        overlap1 = np.concatenate((unit2.waveform[i:], z)) + unit1.waveform
        overlap2 = np.concatenate((z, unit2.waveform[:-i])) + unit1.waveform
        overlap3 = np.concatenate((unit1.waveform[i:], z)) + unit2.waveform
        overlap4 = np.concatenate((z, unit1.waveform[:-i])) + unit2.waveform

        synth_waveforms = np.append(synth_waveforms, np.array(
            [Spike(waveform=overlap1, single_unit1=unit1),
             Spike(waveform=overlap2, single_unit1=unit1),
             Spike(waveform=overlap3, single_unit1=unit2),
             Spike(waveform=overlap4, single_unit1=unit2)]))
    return synth_waveforms


def create_all_synthetic_spikes(units):
    """
    Generates Spike objects of all the synthetic waveforms from any amount
    of single units.
    :param units: Array-like of Spike objects representing single units.
    :return: An array with Spike obj. of all the synthetic overlap waveforms
                between any 2 different single units in the array.
    """
    all_synth_waveforms = np.empty(0)
    all_combinations = np.array(list(combinations(units, 2)))
    for unit1, unit2 in all_combinations:
        all_synth_waveforms = np.append(all_synth_waveforms,
                                        create_synthetic_spikes(unit1, unit2))
    return all_synth_waveforms


def create_overlap_spikes(spikes_samples, peaks_info):
    """
    Generates Spike objects of overlap waveforms from batch of samples.
    :param spikes_samples: Array of array of samples
    :param peaks_info: Array of peak times, corresponds with the spikes_samples
    :return: An array of Spike obj. of all the waveform samples that was given.
    """
    if len(spikes_samples) != len(peaks_info):
        raise ValueError("The peaks array doesn't match the spikes' samples "
                         "array")
    overlap_waveforms = np.array(
        [Spike(spikes_samples[i], peak_time=peaks_info[i])
         for i in range(len(spikes_samples))])
    return overlap_waveforms
