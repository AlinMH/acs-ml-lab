# Tudor Berariu, 2016


import math
from random import randint
from sys import argv
from zipfile import ZipFile

import matplotlib.markers
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy
from scipy.spatial.distance import euclidean


def getArchive():
    archive_url = "http://www.uni-marburg.de/fb12/datenbionik/downloads/FCPS"
    local_archive = "FCPS.zip"
    from os import path

    if not path.isfile(local_archive):
        import urllib

        print("downloading...")
        urllib.urlretrieve(archive_url, filename=local_archive)
        assert path.isfile(local_archive)
        print("got the archive")
    return ZipFile(local_archive)


def getDataSet(archive, dataSetName):
    path = "FCPS/01FCPSdata/" + dataSetName

    lrnFile = path + ".lrn"
    with archive.open(lrnFile, "r") as f:  # open .lrn file
        N = int(f.readline().decode("UTF-8").split()[1])  # number of examples
        D = int(f.readline().decode("UTF-8").split()[1]) - 1  # number of columns
        f.readline()  # skip the useless line
        f.readline()  # skip columns' names
        Xs = np.zeros([N, D])
        for i in range(N):
            data = f.readline().decode("UTF-8").strip().split("\t")
            assert len(data) == (D + 1)  # check line
            assert int(data[0]) == (i + 1)
            Xs[i] = np.array(list(map(float, data[1:])))

    clsFile = path + ".cls"
    with archive.open(clsFile, "r") as f:  # open.cls file
        labels = np.zeros(N).astype("uint")

        line = f.readline().decode("UTF-8")
        while line.startswith("%"):  # skip header
            line = f.readline().decode("UTF-8")

        i = 0
        while line and i < N:
            data = line.strip().split("\t")
            assert len(data) == 2
            assert int(data[0]) == (i + 1)
            labels[i] = int(data[1])
            line = f.readline().decode("UTF-8")
            i = i + 1

        assert i == N

    return Xs, labels  # return data and correct classes


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def dummy(Xs):
    (N, D) = Xs.shape
    Z = np.zeros((N - 1, 4))
    lastIndex = 0
    for i in range(N - 1):
        Z[i, 0] = lastIndex
        Z[i, 1] = i + 1
        Z[i, 2] = 0.1 + i
        Z[i, 3] = i + 2
        lastIndex = N + i
    return Z


def singleLinkage(Xs):
    (N, D) = Xs.shape
    Z = np.zeros((N - 1, 4))
    clusters = []
    i = 0

    #  init
    for x in Xs:
        clusters.append([i, [x]])
        i += 1

    for i in range(N - 1):
        dmin = math.inf
        clust_idx1 = -1
        clust_idx2 = -1
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster[1]:
                for cluster_idx2, cluster2 in enumerate(clusters[(cluster_idx + 1) :]):
                    for point2 in cluster2[1]:
                        if euclidean(point, point2) < dmin:
                            dmin = euclidean(point, point2)
                            clust_idx1 = cluster_idx
                            clust_idx2 = cluster_idx2 + cluster_idx + 1

        idx1 = clusters[clust_idx1][0]
        idx2 = clusters[clust_idx2][0]
        clust1_points = clusters[clust_idx1][1]
        clust2_points = clusters[clust_idx2][1]
        clust1_points.extend(clust2_points)
        clusters[clust_idx1][0] = N + i
        clusters.pop(clust_idx2)
        Z[i, 0] = idx1
        Z[i, 1] = idx2
        Z[i, 2] = dmin
        Z[i, 3] = len(clust1_points)

    return Z


def completeLinkage(Xs):
    (N, D) = Xs.shape
    Z = np.zeros((N - 1, 4))
    clusters = []
    i = 0

    #  init
    for x in Xs:
        clusters.append([i, [x]])
        i += 1

    for i in range(N - 1):
        dmin = math.inf
        clust_idx1 = -1
        clust_idx2 = -1
        for cluster_idx, cluster in enumerate(clusters):
            dmax = -math.inf
            for point in cluster[1]:
                for cluster_idx2, cluster2 in enumerate(clusters[(cluster_idx + 1) :]):
                    for point2 in cluster2[1]:
                        dist = euclidean(point, point2)
                        if dist < dmin:
                            dmin = dist
                            if dist > dmax:
                                dmax = dist
                                clust_idx1 = cluster_idx
                                clust_idx2 = cluster_idx2 + cluster_idx + 1

        idx1 = clusters[clust_idx1][0]
        idx2 = clusters[clust_idx2][0]
        clust1_points = clusters[clust_idx1][1]
        clust2_points = clusters[clust_idx2][1]
        clust1_points.extend(clust2_points)
        clusters[clust_idx1][0] = N + i
        clusters.pop(clust_idx2)
        Z[i, 0] = idx1
        Z[i, 1] = idx2
        Z[i, 2] = dmin
        Z[i, 3] = len(clust1_points)

    return Z


def groupAverageLinkage(Xs):
    (N, D) = Xs.shape
    Z = np.zeros((N - 1, 4))
    clusters = []
    i = 0

    #  init
    for x in Xs:
        clusters.append([i, [x]])
        i += 1

    for i in range(N - 1):
        dmin = math.inf
        clust_idx1 = -1
        clust_idx2 = -1
        dist = {}

        for cluster_idx, cluster in enumerate(clusters):
            dist[cluster_idx] = {}
            for cluster_idx2, cluster2 in enumerate(clusters[(cluster_idx + 1) :]):
                dist[cluster_idx][cluster_idx + cluster_idx2 + 1] = 0

        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster[1]:
                for cluster_idx2, cluster2 in enumerate(clusters[(cluster_idx + 1) :]):
                    for point2 in cluster2[1]:
                        dist[cluster_idx][cluster_idx + cluster_idx2 + 1] += (
                            1 / (len(cluster) * len(cluster2)) * euclidean(point, point2)
                        )

        for cluster_idx, cluster in enumerate(clusters):
            for cluster_idx2, cluster2 in enumerate(clusters[(cluster_idx + 1) :]):
                d = dist[cluster_idx][cluster_idx + cluster_idx2 + 1]
                if d < dmin:
                    dmin = d
                    clust_idx1 = cluster_idx
                    clust_idx2 = cluster_idx + cluster_idx2 + 1

        idx1 = clusters[clust_idx1][0]
        idx2 = clusters[clust_idx2][0]
        clust1_points = clusters[clust_idx1][1]
        clust2_points = clusters[clust_idx2][1]
        clust1_points.extend(clust2_points)
        clusters[clust_idx1][0] = N + i
        clusters.pop(clust_idx2)
        Z[i, 0] = idx1
        Z[i, 1] = idx2
        Z[i, 2] = dmin
        Z[i, 3] = len(clust1_points)

    return Z


def extractClusters(Xs, Z):
    (N, D) = Xs.shape
    assert Z.shape == (N - 1, 4)

    # TODO 4

    # return 1, np.zeros(N)
    return 1, np.zeros(N).astype(int)


def randIndex(clusters, labels):
    assert labels.size == clusters.size
    N = clusters.size

    a = 0.0
    b = 0.0

    for (i, j) in [(i, j) for i in range(N) for j in range(i + 1, N) if i < j]:
        if (
            (clusters[i] == clusters[j])
            and (labels[i] == labels[j])
            or (clusters[i] != clusters[j])
            and (labels[i] != labels[j])
        ):
            a = a + 1
        b = b + 1

    return float(a) / float(b)


def plot(Xs, labels, K, clusters):
    labelsNo = np.max(labels)
    markers = []  # get the different markers
    while len(markers) < labelsNo:
        markers.extend(list(matplotlib.markers.MarkerStyle.filled_markers))
    colors = plt.cm.rainbow(np.linspace(0, 1, K + 1))

    if Xs.shape[1] == 2:
        x = Xs[:, 0]
        y = Xs[:, 1]
        for (_x, _y, _c, _l) in zip(x, y, clusters, labels):
            plt.scatter(_x, _y, s=200, c=[colors[_c]], marker=markers[_l])
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:, 0]
        y = Xs[:, 1]
        z = Xs[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=[colors[_c]], marker=markers[_l])
        plt.show()


if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: " + argv[0] + " dataset_name")
        exit()

    Xs, labels = getDataSet(getArchive(), argv[1])  # Xs is NxD, labels is Nx1
    # Z = singleLinkage(Xs)
    Z = completeLinkage(Xs)
    # Z = groupAverageLinkage(Xs)
    # plt.figure()
    dn = hierarchy.dendrogram(Z)
    # plt.show()

    K, clusters = extractClusters(Xs, Z)
    print("randIndex: ", randIndex(clusters, labels))

    plot(Xs, labels, K, clusters)
