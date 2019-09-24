#!/usr/bin/env python
"""Generates samples for a 2-d Gaussian distribution

Users specify how many samples to draw from each distribution. Each
distribution is specified by passing the cluster center (x, y),
sigma_x and sigma_y
"""
import sys
from random import gauss

try:
    xrange
except NameError:
    xrange = range

try:
    from itertools import izip
except:
    izip = zip

USAGE = """
Usage: {0} output_file N mu_x1 mu_y1 sigma_x1 sigma_y1 ...
    Expects 2 initial arguments: output_file and N (number of samples to draw
    per distribution). The following arguments are expected in groups of 4
    (mu_x, mu_y, sigma_x, sigma_y) and specify the parameters for each
    cluster. At least one group must be specified.

    Example: Running the following
        {0} test_clusters 50 1.0 0 0.5 0.5 2.0 2.0 1.0 1.0
    would generate 50 samples each from two clusters, one with mean (1.0, 0.0)
    and sigmas (0.5, 0.5) and another with mean (2.0, 2.0) and sigmas (1.0, 1.0).
    The data would be written to a file named 'test_clusters'.

    The file has 3 columns: x, y, and cluster_index. The columns are separated
    by a tab character.
"""


def sample(mean, sd):
    return (gauss(mean[0], sd[0]), gauss(mean[1], sd[1]))


def main():
    if len(sys.argv) < 7 or (len(sys.argv) % 4) != 3:
        print(USAGE.format(sys.argv[0]))
        sys.exit(1)

    output_file = sys.argv[1]
    N = int(sys.argv[2])

    cluster_params = sys.argv[3:]

    means = []
    sd = []
    for mx, my, sx, sy in izip(cluster_params[0::4],
                               cluster_params[1::4],
                               cluster_params[2::4],
                               cluster_params[3::4]):
        means.append((float(mx), float(my)))
        sd.append((float(sx), float(sy)))

    with open(output_file, "w") as f:
        for cluster, (mean, sd) in enumerate(izip(means, sd)):
            for i in range(N):
                x, y = sample(mean, sd)
                f.write("{0:.6f}\t{1:.6f}\t{2}\n".format(x, y, cluster))

if __name__ == "__main__":
    main()
