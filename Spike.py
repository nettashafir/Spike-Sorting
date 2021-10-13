import numpy as np

from Constants import *


class Spike:
    """
    Class that represents a spike of a neuron.
    Three options for a spike:
        (1) single unit - spike that represent fire of a single neuron. The waveform is given as an input to the program
        (2) synthetic waveform - spike that it's waveform has created by the program, by overlapping two waveforms of
            two single neurons in some amount of data points, and sum the two waveforms.
        (3) overlapping waveform - a spike that recognized in the input record, which is suspected to be a sum of two
            overlapping waveforms of two single neurons.
    """
    def __init__(self, waveform, single_unit1=None, single_unit2=None,
                 peak_time=None, code=None):
        self.waveform = np.array(waveform)

        # For synthetic waveforms, or overlap waveforms after matching stage
        self.single_unit1 = single_unit1
        self.single_unit2 = single_unit2

        # For overlapping waveforms only; unique value for each waveform
        self.peak_time = peak_time

        # For single units only; the code of the single unit in the
        # SingleUnits enum
        self.code = code

        # The type of the spike
        if self.code is not None:
            self.type = WaveForms.single_unit
        elif self.peak_time is not None:
            self.type = WaveForms.overlap
        else:
            self.type = WaveForms.synthetic

    def __str__(self):
        return str(self.waveform)

    def __add__(self, other):
        return self.waveform + other.waveform

    def __len__(self):
        return len(self.waveform)
