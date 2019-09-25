import sys, csv, random
import numpy as np
import matplotlib.pyplot as plt


# find square distances
def dsquare(pt1, pt2, dim):
    d = 0
    for i in range(dim):
        d += (float(pt1[i]) - float(pt2[i])) ** 2
    return d


def assign_to_c(pt, dim, c):
    # initializing d_min and ci before the loop with i=0 case, start on i=1 in loop
    # min distance squared
    d_min = dsquare(c[0], pt, dim)
    # cluster index of d_min
    ci = c[0][dim]

    # start loop at cluster index i=1
    for i in c[1:]:
        di = dsquare(i, pt, dim)
        if di < d_min:
            d_min = di
            ci = i[dim]  # retrieves cluster index of current cluster compared

    pt[dim] = ci
    return pt


def recenter(c, data, dim):
    re_c = []
    for i in range(len(c)):
        re_c.append([0, 0, -1])  # placeholders to be written over; no cluster exists of index -1

    for i, val in enumerate(c):

        # init runsums for value and count of x
        xi = []  # sum of all x with label k (for each dim)
        for j in range(dim):
            xi.append(0)
        xk = 0  # count of xi with label k

        for j in range(len(data)):
            # if a point's assigned cluster is the cluster being recentered
            if data[j][dim] == val[dim]:
                # increment count of points in cluster
                xk += 1
                # for each dim add the coordinate to the runsum
                for k in range(dim):
                    xi[k] += float(data[j][k])

        # assign new centroid position to c along each dim
        for j in range(dim):
            re_c[i][j] = xi[j] / xk
        re_c[i][dim] = val[dim]

    return re_c


# open tsv, parse into list, close file
def read_tsv():
    data = []
    with open(sys.argv[1], 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
        f.close()
    return data


# initialize k centers of dim dimensions, with cluster index as last list list item
def init_c(k, dim):
    centers = []
    for i in range(k):
        centers.append([])
        for j in range(dim):
            centers[i].append(i + random.randint(0, 10) / 10 * (i + 1))  # rethink random location alg?
        centers[i].append(i)
    return centers


def plot(c, data, dim):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    col = []
    for i in range(len(data)):
        if [float(row[dim]) for row in data][i] == 0:
            col.append('k')
        elif [float(row[dim]) for row in data][i] == 1:
            col.append('r')
        else:
            col.append('b')

    ax1.scatter([float(row[0]) for row in data], [float(row[1]) for row in data], c=col, s=5)

    col2 = []
    for i in range(len(c)):
        if [float(row[dim]) for row in c][i] == 0:
            col2.append('k')
        elif [float(row[dim]) for row in c][i] == 1:
            col2.append('r')
        else:
            col2.append('b')

    ax1.scatter([float(row[0]) for row in c], [float(row[1]) for row in c], c=col2, s=15, marker="s")
    plt.show()


def main():
    if len(sys.argv) < 3:
        print("Usage: {0} <TSV> <int k>".format(sys.argv[0]))
        sys.exit(1)

    # read in tsv data
    data = read_tsv()

    # assign number of dimensions and number of clusters
    dim = len(data[0]) - 1  # -1 as each row comprises of the axes and a final column for the cluster index
    k = int(sys.argv[2])

    # initialize centers at randomized locations
    centers = init_c(k, dim)

    # classify points to nearest centers
    for i in range(len(data)):
        data[i] = assign_to_c(data[i], dim, centers)

    # move each center to centroid of newly labeled points
    re_c = recenter(centers, data, dim)
    iterations = 0

    while centers != re_c and iterations < 20:
        iterations += 1
        sys.stdout.write("Iteration #" + str(iterations) + "\n")

        centers = re_c

        for i in range(len(data)):
            data[i] = assign_to_c(data[i], dim, centers)

        re_c = recenter(centers, data, dim)

    plot(centers, data, dim)


if __name__ == "__main__":
    main()
