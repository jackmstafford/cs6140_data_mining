import json
from math import factorial, log, pi, pow, sqrt, ceil, floor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from random import choice
from sys import maxint

data_dir = 'data/'
data_fields = ['rating', 'price', '# reviews']
colors = ['b', 'r', 'k', 'orange']
markers = ['*', '^', 'D', 'o'] * 4
for i in range(1, 4):
    colors += colors[i:] + colors[:i]
color_and_marker = []
for c, m in zip(colors, markers):
    color_and_marker.append([c, m])

def readDataFromFile(filename):
    return json.load(open(filename, 'r'))

def getDataFilenames(dataDirectory=data_dir, getMin=True):
    filenames = []
    for filename in os.listdir(data_dir):
        sp = filename.split('.')
        if not getMin and len(sp) == 1:
                filenames.append('{}/{}'.format(data_dir, filename))
        elif getMin and len(sp) == 2:
                filenames.append('{}/{}'.format(data_dir, filename))
    return filenames

def dataToBusinessPoints(data, correctZeroPrices=True):
    pts = []
    for d in data:
        pts.append(d.values()[:])
    if correctZeroPrices:
        count = 0.0
        prices = 0.0
        priceIndex = 1
        for p in pts:
            price = p[priceIndex]
            if price != 0:
                count += 1
                prices += price
        avgPrice = prices / count
        for p in pts:
            if p[priceIndex] == 0:
                p[priceIndex] = avgPrice
    return pts

def makeAllBusinessPoints():
    pts = []
    for filename in getDataFilenames():
        data = readDataFromFile(filename)
        pts.extend(dataToBusinessPoints(data))
    return pts

def dataToNeighborhoodPoint(data):
    pt = [0] * len(data[0])
    for p in dataToBusinessPoints(data):
        for i, v in enumerate(p):
            pt[i] += v
    l = float(len(data))
    for i, _ in enumerate(pt):
        pt[i] /= l
    return pt

def makeAllNeighborhoodPoints():
    pts = []
    names = getDataFilenames()
    ptToName = {}
    for filename in names:
        data = readDataFromFile(filename)
        pt = dataToNeighborhoodPoint(data)
        pts.append(pt)
        ptToName[pointForMapping(pt)] = filename.split('/')[2].split('.')[0]
    return pts, names, ptToName

def pointForMapping(point):
    return json.dumps(point)

def normalize(points):
    maxs = []
    for i in range(len(points[0])):
        maxs.append(float(max(p[i] for p in points)))
    for i, p in enumerate(points):
        for j, v in enumerate(p):
            points[i][j] = v / maxs[j]

def binPoints(points, numBins, index=2, binNames=[], updateMapping=False, mapping=[]):
    if len(binNames) == 0:
        binNames = range(numBins)
    lp = len(points)
    binSize = int(floor(lp / float(numBins)))
    rem = lp % numBins
    lows = [0] * numBins
    highs = [0] * numBins
    #print('binSize: {}'.format(binSize))
    # sortByIndex doesn't deep copy so altering sublists is same list => return not necessary
    pts = sortByIndex(points, index)
    i = 0
    for binNum in range(numBins):
        lows[binNum] = pts[i][index]
        size = binSize
        if binNum + 1 <= rem:
            size += 1
        #print('\tbinNum: {}\tsize: {}'.format(binNum, size))
        for _ in range(size):
        #binStart = binNum * binSize
        #for i in range(binStart, min(binStart + binSize, len(pts))):
            #print('\t\ti: {}'.format(i))
            highs[binNum] = pts[i][index]
            if updateMapping:
                before = pointForMapping(pts[i])
            pts[i][index] = binNames[binNum]
            if updateMapping:
                after = pointForMapping(pts[i])
                mapping[after] = mapping.pop(before)
            i += 1
    labels = []
    for low, high in zip(lows, highs):
        labels.append('{:.{a}f} - {:.{a}f}'.format(low, high, a=2))
    return labels
    #return pts

def sortByIndex(points, index):
    return sorted(points, cmp=lambda x,y: cmp(x[index], y[index]))


## Plotting

def plot3(clusters, title, ptToName, do2d=True, isNeighborhood=True, axLabels=[]):
    print(title)
    if isNeighborhood:
        plt3dClstrs, pltClstrs = plotLabelled3dClusters, plotLabelledClusters
    else:
        plt3dClstrs, pltClstrs = plot3dClusters, plotClusters
    df = data_fields
    plt3dClstrs(clusters, title, ptToName, axLabels, *df)
    #if False:
    if do2d:
        for x in range(3):
            for y in range(x + 1, 3):
                pltClstrs(clusters, title, df[x], df[y], x, y, ptToName, [axLabels[x], axLabels[y]])

def clustersToPoints(clusters):
    points = []
    ptLen = len(clusters[0][0])
    for _ in range(ptLen):
        points.append([])
    for clstr in clusters:
        pts = []
        for _ in range(ptLen):
            pts.append([])
        for pt in clstr:
            for i, v in enumerate(pt):
                pts[i].append(v)
        for i, v in enumerate(pts):
            points[i].append(v)
    return points

def plotClusters(clusters, title, xLabel, yLabel, ix=0, iy=1, ptLabels={}, axLabels=[]):
    pts = clustersToPoints(clusters)
    xs = pts[ix]
    ys = pts[iy]
    ax = plt.figure(figsize=(6, 5)).add_subplot(111)
    plt.subplots_adjust(left=.25, top=.94, bottom=.11, right=.98)
    for x, y, m in zip(xs, ys, color_and_marker):
        #ax.plot(x, y, m, mfc='none')
        ax.scatter(x, y, facecolors='none', edgecolors=m[0], marker=m[1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if len(axLabels) > 0:
        for labs, fun in zip(axLabels, [ax.set_xticklabels, ax.set_yticklabels]):
            if len(labs) > 0:
                fun([0] + labs) # TODO maybe figure out what's going wrong with the labels instead of this hack
    plt.show()

def plot3dClusters(clusters, title, ptLabels, axLabels, xLabel, yLabel, zLabel):
    [xs, ys, zs] = clustersToPoints(clusters)
    ax = plt.figure(figsize=(6, 5)).add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, top=.95, bottom=.07, right=.85)
    #ax.view_init(elev=30, azim=-45)
    for x, y, z, m in zip(xs, ys, zs, color_and_marker):
        ax.scatter(x, y, z, c=m[0], marker=m[1])
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    if len(axLabels) > 0:
        for labs, fun, ugh, setlab, axlab in zip(axLabels, 
                [ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels], 
                ['x', 'y', 'z'],
                [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel],
                [xLabel, yLabel, zLabel]
                ):
            if len(labs) > 0:
                ax.tick_params(axis=ugh, pad=22)
                setlab(axlab, labelpad=38)
                fun(labs)
    plt.show()

def plotLabelledClusters(clusters, title, xLabel, yLabel, ix=0, iy=1, ptLabels={}, axLabels=[]):
    [xs, ys, zs] = clustersToPoints(clusters)
    i = 65
    ax = plt.figure(figsize=(9, 5)).add_subplot(111)
    plt.subplots_adjust(left=.16, top=.95, bottom=.1, right=.73)
    for x, y, z, m in zip(xs, ys, zs, colors):
        for a, b, c in zip(x, y, z):
            pt = [a, b, c]
            lab = ptLabels[pointForMapping(pt)]
            px, py = pt[ix], pt[iy]
            plt.scatter(px, py, c=m, marker='${}$'.format(chr(i)), label=lab)
            i += 1
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if len(axLabels) > 0:
        for labs, fun in zip(axLabels, [ax.set_xticklabels, ax.set_yticklabels]):
            if len(labs) > 0:
                fun([0] + labs) # TODO maybe figure out what's going wrong with the labels instead of this hack
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()

def plotLabelled3dClusters(clusters, title, ptLabels, axLabels, xLabel, yLabel, zLabel):
    [xs, ys, zs] = clustersToPoints(clusters)
    ax = plt.figure(figsize=(8.75, 5)).add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, top=.95, bottom=.07, right=.69)
    ax.view_init(elev=30, azim=-45)
    i = 65
    for x, y, z, m in zip(xs, ys, zs, colors):
        for a, b, c in zip(x, y, z):
            lab = ptLabels[pointForMapping([a, b, c])]
            ax.scatter(a, b, c, c=m, marker='${}$'.format(chr(i)), label=lab)
            i += 1
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    if len(axLabels) > 0:
        for labs, fun, ugh, setlab, axlab in zip(axLabels, 
                [ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels], 
                ['x', 'y', 'z'],
                [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel],
                [xLabel, yLabel, zLabel]
                ):
            if len(labs) > 0:
                ax.tick_params(axis=ugh, pad=18)
                setlab(axlab, labelpad=35)
                fun(labs)
    ax.legend(bbox_to_anchor=(1.45, 1))
    plt.show()

def centerToCluster(xToC, points):
    clusters = []
    for i, c in enumerate(xToC):
        while len(clusters) <= c:
            clusters.append([])
        clusters[c].append(points[i])
    return clusters

def distance(a, b):
    sum = 0
    for x, y in zip(a, b):
        sum += pow(x - y, 2)
    return sqrt(sum)


### 1 Hierarchical Clustering

def shortestLink(s1, s2):
    # find the closest points between the sets
    minDist = maxint
    for a in s1:
        for b in s2:
            minDist = min(distance(a, b), minDist)
    return minDist

def longestLink(s1, s2):
    # find the furthest points between the sets
    maxDist = -1
    for a in s1:
        for b in s2:
            maxDist = max(distance(a, b), maxDist)
    return maxDist

def meanLink(s1, s2):
    a1 = [0] * len(s1[0])
    a2 = [0] * len(s1[0])
    for a, s0 in zip([a1, a2], [s1, s2]):
        for i, v in enumerate(a): # loop thru point coords
            for s in s0: # loop thru points
                a[i] += s[i]
            a[i] /= float(len(s0))
    return distance(a1, a2)

def hierarchicalClustering(points, distFunc, k=4):
    # k = number of clusters to pare down to
    clusters = [[p] for p in points]
    if distFunc == 's':
        distFunc = shortestLink
    elif distFunc == 'l':
        distFunc = longestLink
    else:
        distFunc = meanLink
    while len(clusters) > k:
        minDist = maxint 
        x, y = 0, 1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = distFunc(clusters[i], clusters[j])
                if d < minDist:
                    minDist = d
                    x, y = i, j
        xy = clusters.pop(x) + clusters.pop(y - 1)
        clusters.append(xy)
    return clusters

def hc(points, distFunc, k=4):
    return hierarchicalClustering(points, distFunc, k)

def plotHC(clusters, title, ptToName, isNeighborhood=True, axLabels=[]):
    # using HC on C1 so assume x and y only
    plot3(clusters, title, ptToName, do2d=True, isNeighborhood=isNeighborhood, axLabels=axLabels)

def fullHC(points, distFunc='m', k=4, title='Mean-Link', ptToName={}, isNeighborhood=True, axLabels=[]):
    clusters = hierarchicalClustering(points, distFunc, k)
    print('got clusters for ' + title)
    plotHC(clusters, title, ptToName, axLabels=axLabels)

def fullBHC(k=3):
    points = makeAllBusinessPoints()
    #for func, title in zip(['s', 'l', 'm'], ['Single-Link', 'Complete-Link', 'Mean-Link']):
        #fullHC(points, func, k, 'Business ' + title, isNeighborhood=False)
    print(len(points))
    print('I don\'t think you actually want to run this on that many points...')
    #fullHC(points, 'm', k, 'Business Mean-Link', isNeighborhood=False)

def fullNHC(k=3, numBins=8, doBin=True):
    points, names, ptToName = makeAllNeighborhoodPoints()
    labels = [[], [], []]
    if doBin:
        labels[2] = binPoints(points, numBins, 2, updateMapping=True, mapping=ptToName)
    for func, title in zip(['s', 'l', 'm'], ['Single-Link', 'Complete-Link', 'Mean-Link']):
        fullHC(points, func, k, 'Neighborhood ' + title, ptToName, axLabels=labels)


### 2 Assignment-Based Clustering


## k-Means++

def kmeans(points, k=3):
    n = len(points) 
    c0 = points[0]
    c = [c0]
    xToC = [0] * n
    cphiInitial = [c0] * n
    cphi = [cphiInitial]
    for i in range(1, k):
        cost = 0
        cis = []
        cphi.append(cphiInitial)
        # choose ci from X with probability proportional do d(x, phi_ci-1(x))^2
        # okay so make ci a list and append if P_i == P_prev
        # find new center
        for xj, cphij in zip(points, cphi[i]):
            d = distance(xj, cphij)
            dsq = d * d
            if dsq > cost:
                cost = dsq
                cis.append(xj)
        ci = choice(cis)
        #ci = cis[0]
        c.append(ci)
        # assign points to new center
        for j, (xj, cphij) in enumerate(zip(points, cphi[i])):
            d1 = distance(xj, cphij)
            d2 = distance(xj, ci)
            if d1 > d2:
                xToC[j] = i
                cphi[i][j] = ci
    return c, xToC, points

def plotKM(xToC, points, title='k-Means++', ptToName={}, isNeighborhood=True, axLabels=[]):
    clusters = centerToCluster(xToC, points)
    plot3(clusters, title, ptToName, isNeighborhood=isNeighborhood, axLabels=axLabels)

def fullKM(points, k=3, title='k-Means++', doReturn=False, ptToName={}, isNeighborhood=True, axLabels=[]):
    c, xToC, points = kmeans(points, k)
    #print('Centers: {}'.format(c))
    plotKM(xToC, points, title, ptToName, isNeighborhood, axLabels=axLabels)
    if doReturn:
        return c, xToC

def fullBKM(k=3, title='Business k-Means++', doReturn=False, doBin=True, numBins=8):
    points = makeAllBusinessPoints()
    labels = [[], [], []]
    if doBin:
        labels[2] = binPoints(points, numBins, 2)
    return fullKM(points, k, title, doReturn, isNeighborhood=False, axLabels=labels)

def fullNKM(k=3, title='Neighborhood k-Means++', doReturn=False, doBin=True, numBins=8):
    points, names, ptToName = makeAllNeighborhoodPoints()
    labels = [[], [], []]
    if doBin:
        labels[2] = binPoints(points, numBins, 2, updateMapping=True, mapping=ptToName)
    return fullKM(points, k, title, doReturn, ptToName, axLabels=labels)


## Gonzalez

def gonzalez(points, k=3):
    # X is set of all points
    # C is set of all cluster centers
    # assign x to closest cluster c
    n = len(points)
    c0 = points[0]
    c = [c0]
    xmin = [0] * n
    cphiInitial = [c0] * n
    cphi = [cphiInitial]
    for i in range(1, k):
        m = 0
        ci = c0
        cphi.append(cphiInitial)
        # find new center
        for xj, cphij in zip(points, cphi[i]):
            d = distance(xj, cphij)
            if d > m:
                m = d
                ci = xj
        c.append(ci)
        # assign points to new center
        for j, (xj, cphij) in enumerate(zip(points, cphi[i])):
            if distance(xj, cphij) > distance(xj, ci):
                xmin[j] = i
                cphi[i][j] = ci
    return xmin, c, points

def gz(pointsFile, k=3):
    return gonzalez(pointsFile, k)

def plotGZ(xmin, points, title='Gonzalez Greedy Algorithm', ptToName={}, isNeighborhood=True, labels=[]):
    clusters = centerToCluster(xmin, points)
    #if len(clusters) > 4:
        #for i, pts in enumerate(clusters):
            #plot3([pts], '{} - Cluster {}'.format(title, i), ptToName, isNeighborhood=isNeighborhood, axLabels=labels)
    plot3(clusters, title, ptToName, isNeighborhood=isNeighborhood, axLabels=labels)

def fullGZ(points, k=3, title='Gonzalez Greedy Algorithm', doReturn=False, ptToName={}, isNeighborhood=True, labels=[]):
    xmin, c, points = gonzalez(points, k)
    #print('Centers: {}'.format(c))
    title = '{} (k={})'.format(title, k)
    plotGZ(xmin, points, title, ptToName, isNeighborhood, labels)
    if doReturn:
        return c, xmin

def fullBGZ(k=3, title='Business Gonzalez Greedy Algorithm', doBin=True, numBins=8, doNormalize=False, doReturn=False):
    points = makeAllBusinessPoints()
    labels = [[], [], []]
    if doBin:
        #labels = binPoints(points, numBins, 1)
        labels[2] = binPoints(points, numBins, 2)
    if doNormalize:
        normalize(points)
    return fullGZ(points, k, title, doReturn, isNeighborhood=False, labels=labels)

def fullNGZ(k=3, title='Neighborhood Gonzalez Greedy Algorithm', doBin=True, numBins=8, binNames=[], doReturn=False):
    points, names, ptToName = makeAllNeighborhoodPoints()
    labels = [[], [], []]
    if doBin:
        labels[2] = binPoints(points, numBins, 2, updateMapping=True, mapping=ptToName, binNames=binNames)
    return fullGZ(points, k, title, doReturn, ptToName, labels=labels)


## Lloyd's

def lloyds(c, points):
    # assume only 2d points <= I'm not sure I stuck to that
    n = len(points)
    k = len(c)
    while True:
        xToC = [c[0]] * n
        for i, x in enumerate(points):
            minD = maxint
            for j, cj in enumerate(c):
                d = distance(x, cj)
                if d < minD:
                    minD = d
                    xToC[i] = j
        clusters = []
        for i, x in zip(xToC, points):
            while len(clusters) <= i:
                clusters.append([])
            clusters[i].append(x)
        newc = []
        for ci, cluster in enumerate(clusters): # loop thru clusters 
            clen = float(len(cluster))
            avg = [0] * len(c[0])
            for pt in cluster: # loop thru points
                for dim, v in enumerate(pt): # loop thru dimensions
                    avg[dim] += v
            newc.append([i / clen for i in avg])
        if c == newc:
            break
        c = newc
    return c, xToC

def plotLL(xToC, points, title='Lloyd\'s Algorithm', ptToName={}, axLabels=[]):
    clusters = centerToCluster(xToC, points)
    plot3(clusters, title, ptToName, axLabels=axLabels)

def fullLL(points, cSource='g', k=3, title='Lloyd\'s Algorithm', doReturn=False, ptToName={}, axLabels=[]):
    if cSource == 'p':
        cinit = []
        for i in range(3):
            cinit.append(points[i])
        title += ' initialized with first three points'
    else:
        _, cinit, points = gonzalez(points, k)
        title += ' initialized with Gonzalez'
    c, xToC = lloyds(cinit, points)
    print('Centers: {}'.format(c))
    plotLL(xToC, points, title, ptToName, axLabels=axLabels)
    if doReturn:
        return c, xToC 

def fullNLL(cSource='g', k=3, title='Neighborhood Lloyd\'s Algorithm', doReturn=False, doBin=True, numBins=8):
    points, names, ptToName = makeAllNeighborhoodPoints()
    labels = [[], [], []]
    if doBin:
        labels[2] = binPoints(points, numBins, 2, updateMapping=True, mapping=ptToName)
    return fullLL(points, cSource, k, title, doReturn, ptToName, axLabels=labels)
