import sys, csv, random


# find square distances
def dsquare(pt1,pt2,dim):
    d = 0
    for i in range(dim):
        d += (pt1[i]-pt2[i])**2
    return d


def assign_to_c(pt,dim,*c):
    ## initializing dmin and ci before the loop with i=0 case, start on i=1 in loop
    # min distance squared
    print(c)
    dmin = dsquare(c[0],pt,dim)
    # cluster index of dmin
    ci = c[0][dim]

    # start loop at cluster index i=1
    for i in c[1:]:
        di = dsquare(c[i],pt,dim)
        if di < dmin:
            dmin = di
            ci = c[i][dim] # retrieves cluster index of current cluster compared

    pt[dim] = ci
    return pt


def main():
    if len(sys.argv) < 3:
        print("Usage: {0} <TSV> <int k>".format(sys.argv[0]))
        sys.exit(1)

    data = []
    with open(sys.argv[1], 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
        f.close()

    dim = len(data[0])
    k = int(sys.argv[2])

    # initialize k centers of dim dimensions, with cluster index as last list list item
    centers = []
    for i in range(k):
        centers.append([])
        for j in range(dim):
            centers[i].append(i+random.randint(0,10)/10*(i+1))
        centers[i].append(i)


if __name__ == "__main__":
    main()