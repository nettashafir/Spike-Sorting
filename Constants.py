from enum import Enum

# Constants and magic numbers
kHz = 32  # Thousands of samples per second, i.e. samples per 1 millisecond
ms = 100  # Milliseconds that has been measured (in the simulated data)
RESTING_POTENTIAL = -70  # The voltage in the cell while not firing
THRESHOLD = 20000  # The threshold for peak detection
POINTS_BEFORE_PEAK = 20
POINTS_AFTER_PEAK = 40
SPIKE_LENGTH = POINTS_BEFORE_PEAK + POINTS_AFTER_PEAK + 1
REFRACTORY_PERIOD = 2  # In ms
LAMBDAS = [2, 4]  # Parameters of the poisson processes of the simulated single-units
NUM_OF_SPIKES_TO_TAKE = int(ms / min(LAMBDAS)) + 1  # should be higher than (ms/L_i)
MIN_CORR = 0  # The minimal correlation needed to recognize to spike as similar

# paths
SIMULATOR_SINGLE_UNITS_WAVEFORMS_PATHS1 = ["simulator_data\\spike22300.npy", "simulator_data\\spike27000.npy"]
SIMULATOR_SINGLE_UNITS_WAVEFORMS_PATHS2 = ["simulator_data\\spike22200.npy", "simulator_data\\artifact1.npy"]
TRUE_RECORD_PATH = "true_data\\record.npy"
TRUE_SINGLE_UNITS_WAVEFORMS_PATHS = ["true_data\\waveform1.npy",
                                     "true_data\\waveform2.npy",
                                     "true_data\\waveform3.npy",
                                     "true_data\\waveform4.npy"]
TRUE_PEAK_TIMES_PATHS = ["true_data\\peak_times1.npy",
                         "true_data\\peak_times2.npy",
                         "true_data\\peak_times3.npy",
                         "true_data\\peak_times4.npy"]


# enums
class WaveForms(Enum):
    single_unit = 0
    overlap = 1
    synthetic = 2


class SingleUnits(Enum):
    neuron1 = 0
    neuron2 = 1
