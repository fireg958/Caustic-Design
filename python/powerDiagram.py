# MIT License

# Copyright (c) 2021 Devert Alexandre
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.

# original author: 
# https://gist.github.com/marmakoide/45d5389252683ae09c2df49d0548a627


import itertools
import numpy as np
from scipy.spatial import ConvexHull

from matplotlib.collections import LineCollection,PolyCollection
from matplotlib import pyplot as plot
from matplotlib import pyplot as animation
from matplotlib.animation import FuncAnimation

# --- Misc. geometry code -----------------------------------------------------

def norm2(X):
    return np.sqrt(np.sum(X ** 2))

def normalized(X):
    return X / norm2(X)


# --- Delaunay triangulation --------------------------------------------------

def get_triangle_normal(A, B, C):
    return normalized(np.cross(A, B) + np.cross(B, C) + np.cross(C, A))

def get_power_circumcenter(A, B, C):
    N = get_triangle_normal(A, B, C)
    return (-.5 / N[2]) * N[:2]

def is_ccw_triangle(A, B, C):
    M = np.concatenate([np.stack([A, B, C]), np.ones((3, 1))], axis = 1)
    return np.linalg.det(M) > 0

def get_power_triangulation(S, R):
    # Compute the lifted weighted points
    S_norm = np.sum(S ** 2, axis = 1) - R
    S_lifted = np.concatenate([S, S_norm[:,None]], axis = 1)

    # Special case for 3 points
    if S.shape[0] == 3:
        if is_ccw_triangle(S[0], S[1], S[2]):
            return [[0, 1, 2]], np.array([get_power_circumcenter(*S_lifted)])
        else:
            return [[0, 2, 1]], np.array([get_power_circumcenter(*S_lifted)])

    # Compute the convex hull of the lifted weighted points
    hull = ConvexHull(S_lifted)
        
    # Extract the Delaunay triangulation from the lower hull
    tri_list = tuple([a, b, c] if is_ccw_triangle(S[a], S[b], S[c]) else [a, c, b]  for (a, b, c), eq in zip(hull.simplices, hull.equations) if eq[2] <= 0)
        
    # Compute the Voronoi points
    V = np.array([get_power_circumcenter(*S_lifted[tri]) for tri in tri_list])

    # Job done
    return tri_list, V


# --- Compute Voronoi cells ---------------------------------------------------

def get_voronoi_cells(S, V, tri_list):
    # Keep track of which circles are included in the triangulation
    vertices_set = frozenset(itertools.chain(*tri_list))

    # Keep track of which edge separate which triangles
    edge_map = { }
    for i, tri in enumerate(tri_list):
        for edge in itertools.combinations(tri, 2):
            edge = tuple(sorted(edge))
            if edge in edge_map:
                edge_map[edge].append(i)
            else:
                edge_map[edge] = [i]

    # For each triangle
    voronoi_cell_map = { i : [] for i in vertices_set }

    for i, (a, b, c) in enumerate(tri_list):
        # For each edge of the triangle
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
        # Finite Voronoi edge
            edge = tuple(sorted((u, v)))
            if len(edge_map[edge]) == 2:
                j, k = edge_map[edge]
                if k == i:
                    j, k = k, j
                
                # Compute the segment parameters
                U = V[k] - V[j]
                U_norm = norm2(U)               

                # Add the segment
                voronoi_cell_map[u].append(((j, k), (V[j], U / U_norm, 0, U_norm)))
            else: 
            # Infinite Voronoi edge
                # Compute the segment parameters
                A, B, C, D = S[u], S[v], S[w], V[i]
                U = normalized(B - A)
                I = A + np.dot(D - A, U) * U
                W = normalized(I - D)
                if np.dot(W, I - C) < 0:
                    W = -W  
            
                # Add the segment
                voronoi_cell_map[u].append(((edge_map[edge][0], -1), (D,  W, 0, None)))             
                voronoi_cell_map[v].append(((-1, edge_map[edge][0]), (D, -W, None, 0)))             

    # Order the segments
    def order_segment_list(segment_list):
        # Pick the first element
        first = min((seg[0][0], i) for i, seg in enumerate(segment_list))[1]

        # In-place ordering
        segment_list[0], segment_list[first] = segment_list[first], segment_list[0]
        for i in range(len(segment_list) - 1):
            for j in range(i + 1, len(segment_list)):
                if segment_list[i][0][1] == segment_list[j][0][0]:
                    segment_list[i+1], segment_list[j] = segment_list[j], segment_list[i+1]
                    break

        # Job done
        return segment_list

    # Job done
    return { i : order_segment_list(segment_list) for i, segment_list in voronoi_cell_map.items() }


# --- Plot all the things -----------------------------------------------------
def get_weighted_centroid(A, U, tmin, tmax):
    tmid = (tmin + tmax) / 2.0
    return A + tmid * U

def is_infinite_triangle(tri):
    return any(idx == -1 for idx in tri)

# --- Main entry point --------------------------------------------------------

def main():
    # Load your points and weights
    points_file = "/home/dylan/blenders_full_1500_square.dat"
    weight_file = "/home/dylan/blenders_full_1500_square.weight"

    # Load points
    with open(points_file, "r") as f:
        points_data = f.read().splitlines()
    S = []
    for point in points_data:
        x, y = map(float, point.split(" "))
        S.append([x, y])
    S = np.array(S)

    # Load weights
    with open(weight_file, "r") as f:
        weight_data = f.read().splitlines()
    R = np.array(list(map(float, weight_data)))

    def animate_voronoi(i):
        # Compute the power triangulation of the circles
        tri_list, V = get_power_triangulation(S, R*(i/100))

        # Compute the Voronoi cells
        voronoi_cell_map = get_voronoi_cells(S, V, tri_list)

        plot.cla()  # Clear the previous frame

        # Setup
        plot.axis('equal')
        plot.axis('off')    

        # Set min/max display size, as Matplotlib does it wrong
        min_corner = np.amin(S, axis = 0) - np.max(R)
        max_corner = np.amax(S, axis = 0) + np.max(R)
        plot.xlim((min_corner[0], max_corner[0]))
        plot.ylim((min_corner[1], max_corner[1]))

        # Plot the power triangulation
        #edge_set = frozenset(tuple(sorted(edge)) for tri in tri_list for edge in itertools.combinations(tri, 2))
        #line_list = LineCollection([(S[i], S[j]) for i, j in edge_set], lw = 1., colors = '.9')
        #line_list.set_zorder(0)
        #ax.add_collection(line_list)

        # Plot the Voronoi cells
        edge_map = { }
        for segment_list in voronoi_cell_map.values():
            for edge, (A, U, tmin, tmax) in segment_list:
                edge = tuple(sorted(edge))
                if edge not in edge_map:
                    if tmax is None:
                        tmax = 10
                    if tmin is None:
                        tmin = -10

                    edge_map[edge] = (A + tmin * U, A + tmax * U)

        line_list = LineCollection(edge_map.values(), lw = 1., colors = 'k')
        line_list.set_zorder(0)
        ax.add_collection(line_list)

        #plot.scatter(V[:,0], V[:,1], s=1)

        # Job done
        #plot.show()
        print("frame " + str(i))


    # Create the figure and axis
    fig, ax = plot.subplots()

    fig.set_size_inches(10, 10, True)
    dpi = 50


    # Create the animation
    anim = FuncAnimation(fig, animate_voronoi,
                                frames=100, interval=4000/50, blit=True)

    anim.save('animation2.gif', writer='imagemagick', fps=25, dpi=dpi)


    # Display the animation
    plot.show()

    # Compute and export the weighted centroid of each Voronoi cell
    #output_file = "voronoi_centroids.dat"
    #with open(output_file, "w") as f:
    #    for point, segment_list in voronoi_cell_map.items():
    #        weighted_centroid = np.zeros(2)
    #        total_weight = 0.0
    #        for _, (A, U, tmin, tmax) in segment_list:
    #            if tmin is None:
    #                tmin = -10
    #            if tmax is None:
    #                tmax = 10
    #            segment_centroid = get_weighted_centroid(A, U, tmin, tmax)
    #            segment_weight = abs(tmax - tmin)
    #            weighted_centroid += segment_centroid * segment_weight
    #            total_weight += segment_weight
    #        weighted_centroid /= total_weight
    #        x, y = weighted_centroid
            
            # Exclude centroids from infinite triangles
            #if not any(is_infinite_triangle(tri) for tri in tri_list if point in tri):
    #        f.write(f"{x} {y}\n")

if __name__ == '__main__':
    main()