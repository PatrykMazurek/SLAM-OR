import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111)

import numpy as np
points = np.load("points.npy")
points_cluster = np.load("points_cluster.npy")

print(points.shape)
ccc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
from sklearn.decomposition import PCA
def draw_rect(xxx, ax, my_col):
    pca = PCA(n_components=2)
    pca.fit(xxx)
    print(pca.components_)
    v_my = []
    v_v = []
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 2.5 * np.sqrt(length)
        #print(pca.components_)
        #ax.plot((198, 210), (-49, -43))
        #ax.plot((pca.mean_[0] - v[0], pca.mean_[0] + v[0]),(pca.mean_[1] - v[1], pca.mean_[1] + v[1]))
        v_my.append((pca.mean_[0] - v[0], pca.mean_[0] + v[0],pca.mean_[1] - v[1], pca.mean_[1] + v[1]))
        v_v.append(v)
        #draw_vector(pca.mean_, pca.mean_ + v)

    #ax.plot((v_my[0][0], v_my[0][1]),(v_my[0][2], v_my[0][3]))
    #ax.plot((v_my[1][0], v_my[1][1]),(v_my[1][2], v_my[1][3]))

    #ax.plot((v_my[1][0] - v_v[1][0], v_my[1][1] - v_v[1][0]),(v_my[1][2], v_my[1][3]))
    ax.plot((pca.mean_[0] - v_v[0][0] - v_v[1][0], pca.mean_[0] + v_v[0][0] - v_v[1][0]),(pca.mean_[1] - v_v[0][1] - v_v[1][1], pca.mean_[1] + v_v[0][1]  - v_v[1][1]), color=my_col)
    ax.plot((pca.mean_[0] - v_v[0][0] + v_v[1][0], pca.mean_[0] + v_v[0][0] + v_v[1][0]),(pca.mean_[1] - v_v[0][1] + v_v[1][1], pca.mean_[1] + v_v[0][1]  + v_v[1][1]), color=my_col)


    #ax.plot((pca.mean_[0] - v_v[1][0] - v_v[0][0], pca.mean_[0] + v_v[1][0] + v_v[0][0]),(pca.mean_[1] - v_v[1][1] - v_v[0][0], pca.mean_[1] + v_v[1][1] - v_v[0][0]))
    ax.plot((pca.mean_[0] - v_v[1][0] - v_v[0][0], pca.mean_[0] + v_v[1][0] - v_v[0][0]),(pca.mean_[1] - v_v[1][1] - v_v[0][1], pca.mean_[1] + v_v[1][1] - v_v[0][1]), color=my_col)
    ax.plot((pca.mean_[0] - v_v[1][0] + v_v[0][0], pca.mean_[0] + v_v[1][0] + v_v[0][0]),(pca.mean_[1] - v_v[1][1] + v_v[0][1], pca.mean_[1] + v_v[1][1] + v_v[0][1]), color=my_col)


xxx = points[points_cluster == 1,]

pca = PCA(n_components=2)
pca.fit(xxx)
print(pca.components_)

for a in range(xxx.shape[0]):
    #print(vvc[a])
    #ax.scatter(x=listX[a], y=listY[a], s=25, c=colors[vvc[a]], marker='.')  # , alpha=0.5)
    ax.scatter(x=xxx[a,0], y=xxx[a,1], s=25, c='r', marker='.')  # , alpha=0.5)
    #ax.scatter(x=listX[a], y=listY[a], s=5, c="0", marker='.')  # , alpha=0.5)

draw_rect(xxx, ax, 'red')
#print(pca.mean_)



#ax.plot((pca.mean_[0] - v[0], pca.mean_[0] + v[0]),(pca.mean_[1] - v[1], pca.mean_[1] + v[1]))
#v00 =
plt.axis('equal');

plt.show()
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(points)
print(pca.components_)



def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    print(v1)
    print(v0)
    #ax.annotate('', v1, v0, arrowprops=arrowprops)



for a in range(points.shape[0]):
    #print(vvc[a])
    #ax.scatter(x=listX[a], y=listY[a], s=25, c=colors[vvc[a]], marker='.')  # , alpha=0.5)
    if points_cluster[a] == 1:
        ax.scatter(x=points[a,0], y=points[a,1], s=25, c=ccc[points_cluster[a] % 10], marker='.')  # , alpha=0.5)
        #ax.scatter(x=listX[a], y=listY[a], s=5, c="0", marker='.')  # , alpha=0.5)

print(pca.mean_)

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    #print(pca.components_)
    #ax.plot((198, 210), (-49, -43))
    ax.plot((pca.mean_[0], pca.mean_[0] + v[0]),(pca.mean_[1], pca.mean_[1] + v[1]))
    #draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');

plt.show()
"""