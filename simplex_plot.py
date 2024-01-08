"""
    Visualize points on the 3-simplex (eg, the parameters of a
    3-dimensional multinomial distributions) as a scatter plot
    contained within a 2D triangle.
    Adapted from David Andrzejewski (david.andrzej@gmail.com)
"""
import numpy as np
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C
import matplotlib.patches as PA


def plotSimplex(points, fig=None,
                vertexlabels=['1: initial flat PMFs', '2: intermediate unilateral PMFs', '3: final bilateral PMFs'], title='',
                save_title="./summary_figures/dur_simplex.png", show=False, vertexcolors=['k', 'k', 'k'], x_offset=0, y_offset=0, **kwargs):
    """
    Plot Nx3 points array on the 3-simplex
    (with optionally labeled vertices)

    kwargs will be passed along directly to matplotlib.pyplot.scatter
    """
    if fig is None:
        fig = P.figure(figsize=(9, 9))
    # Draw the triangle
    l1 = L.Line2D([0, 0.5, 1.0, 0], # xcoords
                  [0, np.sqrt(3) / 2, 0, 0], # ycoords
                  color='k')
    fig.gca().add_line(l1)
    fig.gca().xaxis.set_major_locator(MT.NullLocator())
    fig.gca().yaxis.set_major_locator(MT.NullLocator())
    # Draw vertex labels
    fig.gca().annotate(vertexlabels[0], (-0.23, 0.), size=24, color=vertexcolors[0], annotation_clip=False)
    fig.gca().annotate(vertexlabels[1], (1.015, 0.), size=24, color=vertexcolors[1], annotation_clip=False)
    fig.gca().annotate(vertexlabels[2], (0.395, np.sqrt(3) / 2 + 0.035), size=24, color=vertexcolors[2], annotation_clip=False)
    # Project and draw the actual points
    projected = projectSimplex(points / points.sum(1)[:, None])
    P.scatter(projected[:, 0] + x_offset, projected[:, 1] + y_offset, s=points.sum(1) * 3.5, **kwargs)#s=35

    # plot center with average size
    projected = projectSimplex(np.mean(points / points.sum(1)[:, None], axis=0).reshape(1, 3))
    P.scatter(projected[:, 0], projected[:, 1], marker='*', color='magenta', s=np.mean(points.sum(1)) * 3.5)

    # Leave some buffer around the triangle for vertex labels
    fig.gca().set_xlim(-0.08, 1.08)
    fig.gca().set_ylim(-0.08, 1.08)

    P.axis('off')
    if title != '':
        P.annotate(title, (0.395, np.sqrt(3) / 2 + 0.075), size=24)

    # P.tight_layout()
    P.savefig(save_title, dpi=300, transparent=True) #  bbox_inches='tight'
    if show:
        P.show()
    else:
        P.close()


def projectSimplex(points):
    """
    Project probabilities on the 3-simplex to a 2D triangle

    N points are given as N x 3 array
    """
    # Convert points one at a time
    tripts = np.zeros((points.shape[0], 2))
    for idx in range(points.shape[0]):
        # Init to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * np.sqrt(3))
        # Vector 1 - bisect out of lower left vertex
        p1 = points[idx, 0]
        x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
        # Vector 2 - bisect out of lower right vertex
        p2 = points[idx, 1]
        x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)
        # Vector 3 - bisect out of top vertex
        p3 = points[idx, 2]
        y = y + (1.0 / np.sqrt(3) * p3)

        tripts[idx, :] = (x, y)

    return tripts


if __name__ == '__main__':
    # Define a synthetic test dataset
    labels = ('[0.1  0.1  0.8]',
              '[0.8  0.1  0.1]',
              '[0.5  0.4  0.1]',
              '[0.17  0.33  0.5]',
              '[0.33  0.34  0.33]')
    testpoints = np.array([[0.1, 0.1, 0.8],
                           [0.8, 0.1, 0.1],
                           [0.5, 0.4, 0.1],
                           [0.17, 0.33, 0.5],
                           [0.33, 0.34, 0.33]])
    # Define different colors for each label
    c = range(len(labels))
    # Do scatter plot
    fig = plotSimplex(testpoints, c='k', show=1)
