import sys, csv, random
import numpy as np
import matplotlib.pyplot as plt


# find square distances
def dsquare(pt1,pt2,dim):
    d = 0
    for i in range(dim):
        d += (float(pt1[i])-float(pt2[i]))**2
    return d


def assign_to_c(pt,dim,c):
    ## initializing dmin and ci before the loop with i=0 case, start on i=1 in loop
    # min distance squared

    dmin = dsquare(c[0],pt,dim)
    # cluster index of dmin
    ci = c[0][dim]

    # start loop at cluster index i=1
    for i in c[1:]:
        di = dsquare(i, pt, dim)
        if di < dmin:
            dmin = di
            ci = i[dim] # retrieves cluster index of current cluster compared

    pt[dim] = ci
    return pt


def recenter(c,data,dim):
    re_c = []
    for i in range(len(c)):
        re_c.append([0,0,-1])

    for i, val in enumerate(c):

        # init runsums for value and count of x
        xi = [] # sum of all x with label k (for each dim)
        for j in range(dim):
            xi.append(0)
        xk = 0 # count of xi with label k

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
def readtsv():
    data = []
    with open(sys.argv[1], 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
        f.close()
    return data


# initialize k centers of dim dimensions, with cluster index as last list list item
def initc(k,dim):
    centers = []
    for i in range(k):
        centers.append([])
        for j in range(dim):
            centers[i].append(i+random.randint(0,10)/10*(i+1)) #rethink random location alg?
        centers[i].append(i)
    return centers


def plot(c,data,dim):
    # x =  range(max(data[:][0]))
    # y = range(max(data[:][1]))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # col = np.where(data[dim]==0,'k',np.where(data[dim]==1,'b','r'))
    #
    # ax1.scatter(data[:][0], data[:][1], c=col, s=5)
    # plt.show()

    plt.scatter(data[:][0], data[:][1])
    plt.show()


def main():
    if len(sys.argv) < 3:
        print("Usage: {0} <TSV> <int k>".format(sys.argv[0]))
        sys.exit(1)

    # read in tsv data
    data = readtsv()

    # assign number of dimensions and number of clusters
    dim = len(data[0]) - 1 # -1 as each row comprises of the axes and a final column for the cluster index
    k = int(sys.argv[2])

    # initialize centers at randomized locations
    centers = initc(k,dim)

    # classify points to nearest centers
    for i in range(len(data)):
        data[i] = assign_to_c(data[i], dim, centers)

    # move each center to centroid of newly labeled points
    re_c = recenter(centers,data,dim)
    iterations = 0

    while centers != re_c and iterations < 20:
        iterations += 1
        sys.stderr.write("Iteration #" + str(iterations) + "\n")

        centers = re_c

        for i in range(len(data)):
            data[i] = assign_to_c(data[i], dim, centers)

        re_c = recenter(centers, data, dim)

    plot(centers,data,dim)


if __name__ == "__main__":
    main()