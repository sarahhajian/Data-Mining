import numpy
from sklearn.cluster import KMeans


def k_means(X_scaled):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_scaled)
    labels = kmeans.labels_
    C1 = numpy.where(labels == 1)[0]
    C2 = numpy.where(labels == 0)[0]
    print('C1: ', len(C1), '\n', 'C2: ', len(C2),'\n')
    print('C1: ', C1, '\n', 'C2: ', C2)
