import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from Constants import *
import HelperFunctions as hf
from Spike import Spike


def cluster(waveforms, spike_objects, _plot=False):
    """
    Divide the the given spikes into clusters.
    :param waveforms: A m*d matrix of m waveforms in size of d points.
    :param spike_objects: A vector of Spike objects of m size; the i'th Spike
                         corresponds to the i'th waveform in 'samples'.
    :param _plot: If true, the clusters are plotted with their means.
    :return: An array of arrays, each one is a cluster of Spike objects.
    """
    pca = PCA(n_components=3).fit(waveforms)
    features = pca.transform(waveforms)
    # features = np.concatenate((features, np.array([np.linalg.norm(data, ord=1, axis=1)]).T), axis=1)
    # features = np.concatenate((features, np.array([data[:, POINTS_BEFORE_PEAK]]).T), axis=1)
    # features = np.concatenate((features, abs(fft(data, axis=1))), axis=1)  # shape (., 36)

    k_means = KMeans(n_clusters=15, random_state=0).fit(features)
    clustering = k_means.labels_
    clusters = []

    for cluster_num in set(clustering):
        cluster = spike_objects[clustering == cluster_num]
        clusters.append(cluster)
        if _plot:
            samples_cluster = waveforms[clustering == cluster_num]
            c_mean = np.mean(samples_cluster, axis=0)
            for row in range(len(samples_cluster)):
                plt.plot(samples_cluster[row])
            plt.plot(range(len(c_mean)), c_mean, '-k', linewidth=2)
            plt.title(f"Cluster {cluster_num}")
            plt.show()
    return clusters


def divide_to_single_units(clusters):
    """
    Divide the clusters to single-unit clusters, when every Spike object is
    sent to the cluster of the single-unit which was matched to it before.
    :param clusters: Array of arrays of Spike objects.
    :param single_units_amount: Prior information about the number of single
                                units behind the original multi-unit activity.
    :return: Array of arrays of Spike objects which are the single-unit
             clusters, and array of the multi unit-cluster with all of the
             Spikes that wasn't any synthetic waveform in their cluster.
    """
    single_units_clusters = [[] for unit in range(len(SingleUnits))]
    multi_unit_cluster = []
    for i in range(len(clusters)):
        if not hf.does_contain_synthetic(clusters[i]):
            multi_unit_cluster.extend(clusters[i])
            continue
        for spike in clusters[i]:
            if spike.type is not WaveForms.overlap:
                continue
            if not hf.find_best_match(spike, clusters[i]):
                multi_unit_cluster.append(spike)
                continue
            single_units_clusters[spike.single_unit1.code.value].append(spike)
            if spike.single_unit2 is not None:
                single_units_clusters[spike.single_unit2.code.value]. \
                    append(Spike(waveform=spike.waveform,
                                 single_unit1=spike.single_unit2,
                                 peak_time=spike.peak_time))
    return single_units_clusters, multi_unit_cluster


def fix_refractory_violations(single_units_clusters, multi_unit_cluster):
    """
    Filter out from a cluster waveforms that isn't reasonable in the matter
    of refractory periods, i.e. remove waveforms that are too much close to
    their neighbors.
    :param single_units_clusters: The clusters to clear from violations.
    :param multi_unit_cluster: The multi-unit cluster to move all the
                               violations to.
    :return: None
    """
    for i in range(len(single_units_clusters)):
        move_to_multi_unit = set()
        single_units_clusters[i].sort(key=lambda spike: spike.peak_time)
        k = 0
        for j in range(1, len(single_units_clusters[i])):
            curr_spike, next_spike = single_units_clusters[i][k], single_units_clusters[i][j]
            if next_spike.peak_time - curr_spike.peak_time < REFRACTORY_PERIOD * kHz:
                if hf.corr_helper(curr_spike, next_spike):
                    move_to_multi_unit.add(j)
                else:
                    move_to_multi_unit.add(k)
                    k = j
            else:
                k = j

        single_units_clusters[i] = [single_units_clusters[i][j] for j in
                                    range(len(single_units_clusters[i]))
                                    if j not in move_to_multi_unit]
        multi_unit_cluster.extend([single_units_clusters[i][j] for j in
                                   range(len(single_units_clusters[i]))
                                   if j in move_to_multi_unit])


# ########    Not in use    ########## #

# ##  Data from Brian ## #

# data1, times1, true_spikes1 = generate_spikes(_plot=False)
# data2, times2, true_spikes2 = generate_spikes(threshold=0.4, factor=1.2, ref=3, _plot=False)
# samples = data1[0] + data1[1] + data2[0] + data2[1]


# ## Moving average ### #

# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w
#
# smoothed = np.concatenate((np.ones(3), moving_average(samples, 9)))
# fig = make_subplots(rows=1, cols=1)
# fig.add_traces([go.Scatter(x=list(range(len(samples))), y=samples),
#                 go.Scatter(x=list(range(len(smoothed))), y=smoothed)])
# fig.show()


# ### Using Klustakwik ### #

# import klustakwik2 as kk
#
# np.savetxt(fname="data.n", X=data, fmt="%f")
# np.savetxt(fname="name.fet.n", X=features, fmt="%f", header=str(len(features[0])), comments="")
# mask = np.full(features.shape, 1)
# np.savetxt(fname="name.fmask.n", X=mask, fmt="%f", header=str(len(features[0])), comments="")
#
# # --> Now run klustakwik from terminal with: kk2_legacy name n
#
# data = np.loadtxt("data.n")
# clusters = np.loadtxt("name.clu.n")[:-1]


# #### Information from Plexon #### #

# df = pd.read_excel(r'C:\Users\owner\Desktop\netta\ElectrodeInformation\Elct.xlsx')
# df.to_pickle(r"C:\Users\owner\Desktop\netta\ElectrodeInformation\Elect.pkl")
#
# df = pd.read_pickle(r"C:\Users\owner\Desktop\netta\ElectrodeInformation\Elect.pkl")
# df.columns = range(df.shape[1])
#
# cluster2 = df[np.logical_and(df[0] == 1, df[1] == 2)].iloc[:, :]  # shape (12977, 202x)
# for row in range(len(cluster2)):
#     plt.plot(cluster2.iloc[row:row+1])
# plt.show()
#
# data = df[df[0] == 1].iloc[:, 6:]  # shape (92963, 196)
#
# pca = PCA(n_components=3).fit(data)  # Explained variance [0.64691183 0.15397376 0.03079512]
# features = pca.transform(data)  # (92963, 3)
# print("OK 1: PCA")
# features = np.concatenate((features, np.array([np.linalg.norm(data, ord=1, axis=1)]).T), axis=1)
# print("OK 2: Area")
# features = np.concatenate((features, abs(fft(data, axis=1))), axis=1)
# print("OK 3: FFT")
