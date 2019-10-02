import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import hypertools as hyp

parser = argparse.ArgumentParser()
parser.add_argument('fname', type=str)
parser.add_argument('k_clusters', metavar='k', type=int)
parser.add_argument('export_name', type=str)
parser.add_argument('-fuzzy', action='store_true')
parser.add_argument('-transpose', action='store_true')
args = parser.parse_args()

# other arguments
d_threshold = 10**(-100)  # minimum distance to fix overflow and overlapping point issues, acts like a miniature laplace
i_threshold = 50
# max number of iterations before convergence declared
m = 2  # degree of "fuzziness"; higher -> fuzzier, default of 2


# find square distances
def d_square(pt1, pt2, dim):
    d = 0
    for i in range(dim):
        d += (float(pt1[i]) - float(pt2[i])) ** 2

    if d < d_threshold:
        return d_threshold  # correct for if points are at same location
    else:
        return d
# def assign_to_c(pt, dim, c):
#     # initializing d_min and ci before the loop with i=0 case, start on i=1 in loop
#     # min distance squared
#     d_min = d_square(c[0], pt, dim)
#     # cluster index of d_min
#     ci = c[0][dim]
#
#     # start loop at cluster index i=1
#     for i in c[1:]:
#         di = d_square(i, pt, dim)
#         if di < d_min:  # if a new minimum distance is found
#             d_min = di  # replace with lowest distance
#             ci = i[dim]  # store cluster index of current cluster center
#
#     pt[dim] = ci
#     return pt


def assign_c_prob(pt, dim, c):
    # The general equation for wk(x) is wk = 1 / sigma_j( (d(ck,x) / d(cj,x)) ** (2/m-1) )

    # create array of square distances to each center from point
    d = []
    for ck in c:
        d.append(d_square(ck, pt, dim))

    # prepare to store wk "partition matrix" / probability of belonging to a center
    wk = []
    for i in range(len(c)):
        d_ratio = []
        d_k = d[i]  # current center being looked at

        # create array of ratios to be summed over
        for d_j in d:
            d_ratio.append((d_k / d_j) ** (2/(m-1)))

        # calculate and append final wk for the current center
        wk.append(1 / sum(d_ratio))

    if not args.fuzzy:
        ind = np.where(wk == np.amax(wk))[0][0]
        wk = [0] * len(wk)  # replace all values with zero
        wk[ind] = 1  # insert a one at the index of the closest (max)

    # place new weighted values to each center in array in last index of pt
    pt[dim] = wk
    return pt


def recenter_prob(data, dim):
    re_c = []
    for i in range(args.k_clusters):
        re_c.append([])
        for j in range(dim):
            re_c[i].append(0)
        re_c[i].append(-1)

    if args.fuzzy:
        for i, c_i in enumerate(re_c):
            for j in range(dim):
                # runsums across all points for a given dimension
                numerator = 0
                denominator = 0
                for pt in data:
                    numerator += (pt[dim][i] ** m) * pt[j]
                    denominator += (pt[dim][i] ** m)
                c_i[j] = numerator / denominator
            c_i[dim] = i
        return re_c

    elif not args.fuzzy:
        for i, val in enumerate(re_c):
            # init runsums for value and count of x
            xi = []  # sum of all x with label k (for each dim)
            for j in range(dim):
                xi.append(0)
            xk = 0  # count of xi with label k
            for j in range(len(data)):
                # if a point's assigned cluster is the cluster being recentered
                if np.where(data[j][dim] == np.amax(data[j][dim]))[0][0] == i:
                    # increment count of points in cluster
                    xk += 1
                    # for each dim add the coordinate to the runsum
                    for k in range(dim):
                        xi[k] += data[j][k]
            # assign new centroid position to c along each dim
            for j in range(dim):
                val[j] = xi[j] / xk
            val[dim] = i
        return re_c

#
# def recenter(c, data, dim):
#     re_c = []
#     # I know this looping is funky, only like this as former version seemed to somehow be overwriting original c list
#     for i in range(len(c)):
#         re_c.append([0, 0, -1])  # placeholders to be written over; no cluster exists of index -1
#
#     for i, val in enumerate(c):
#
#         # init runsums for value and count of x
#         xi = []  # sum of all x with label k (for each dim)
#         for j in range(dim):
#             xi.append(0)
#         xk = 0  # count of xi with label k
#
#         for j in range(len(data)):
#             # if a point's assigned cluster is the cluster being recentered
#             if data[j][dim] == val[dim]:
#                 # increment count of points in cluster
#                 xk += 1
#                 # for each dim add the coordinate to the runsum
#                 for k in range(dim):
#                     xi[k] += float(data[j][k])
#
#         # assign new centroid position to c along each dim
#         for j in range(dim):
#             re_c[i][j] = xi[j] / xk
#         re_c[i][dim] = val[dim]
#
#     return re_c


# open tsv, parse into list, close file
def read_tsv():
    data = []
    with open(args.fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
        f.close()
    data = np.array(data).astype(float).tolist()
    return data


# initialize k centers of dim dimensions, with cluster index as last list list item
def init_c(data, dim):
    centers = []
    c_index = 0
    for i in np.random.randint(0, len(data), args.k_clusters):
        centers.append(data[i][:dim])
        centers[c_index].append(c_index)
        c_index += 1
    return centers


def plot(c, data, dim):
    col = plt.cm.tab20c(
        np.linspace(0, 1, args.k_clusters)
    )
    plt.figure()
    data = np.array(data)
    for i in range(args.k_clusters):
        a = []
        for row in data:
            a.append(
                np.where(row[dim] == np.amax(row[dim]))[0][0]
            )
        ind = [row == i for row in a]
        plt.scatter(data[ind, 0],
                    data[ind, 1],
                    c=col[i],
                    s=3)
        plt.scatter(c[i][0],
                    c[i][1],
                    s=10,
                    c=col[i],
                    linewidths=1,
                    edgecolors="000")
    plt.savefig(args.export_name+".png")
    plt.show()

def plot_big(data, dim):
    print("bigplot bb")
    col = plt.cm.tab20c(np.linspace(0, 1, args.k_clusters))
    hue = [0]*len(data)
    for i, row in enumerate(data):
        hue[i] = matplotlib.colors.rgb2hex(col[np.where(row[dim] == np.amax(row[dim]))[0][0]])

    dat = np.array([a[:dim] for a in data])
    TSNE = hyp.plot(dat, '.', reduce="TruncatedSVD", hue=hue, save_path=args.export_name+".png")  #reduce="TSNE",


def plot_bars(centers, data, dim):
    counts = []
    for i, k in enumerate(centers):
        counts.append(sum([np.where(b[dim]==np.amax(b[dim]))[0][0]==i for b in data]))
    plt.hist(counts, bins=30)
    plt.savefig(args.export_name+".png")
    plt.show()


def main():
    # read in tsv data
    data = read_tsv()
    orig_data = data  # store off to the side

    if args.transpose:
        data = np.array(data).T.tolist()

    # assign number of dimensions and number of clusters
    dim = len(data[0]) - 1  # -1 as each row comprises of the axes and a final column for the cluster index

    # # SCORE SCORE SCORE SCORE SCORE SCORE
    # file = open(args.export_name+".txt", "w")
    # s = []
    # for z in range(100):
    # initialize centers at randomized locations
    centers = init_c(data, dim)
    for i in range(len(data)):
        data[i] = assign_c_prob(data[i], dim, centers)
    re_c = recenter_prob(data, dim)
    iterations = 1
    while centers != re_c and iterations < i_threshold:
        print(iterations)
        iterations += 1
        centers = re_c
        for i in range(len(data)):
            data[i] = assign_c_prob(data[i], dim, centers)
        re_c = recenter_prob(data, dim)

    #     # SCORE SCORE SCORE SCORE SCORE
    #     s_z = 0
    #     for i in range(args.k_clusters):
    #         for j in data:
    #             if j[dim][i] == 1:
    #                 s_z += d_square(centers[i], j, dim)
    #     s.append(s_z)
    #     file.write(str(sorted(centers,key=lambda l:l[0]))+"\n")
    #     print(str(z)+": "+str(s_z))
    #
    # file.close()
    # print("Best score index: " + str(np.where(s == np.amin(s))[0][0]))



    # for c in centers:
    #     print("Cluster #" + str(c[dim]) + ": " + str(c[0:dim]) + "\n")
    # for d in data:
    #     print(str(d) + "\n")
    #i=1
    # print(data[i][dim])

    plot_big(data, dim)

    print(np.array([a[dim] for a in data]).flatten().max())
    print(np.array([a[dim] for a in data]).flatten().min())
    print(np.array([a[dim] for a in data]).flatten().sum()/len(data))
    print("Fuzzy: " + str(args.fuzzy))
    print("Iterations: " + str(iterations))

    # plot(centers, data, dim)


if __name__ == "__main__":
    main()
