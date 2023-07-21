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

# Modified by Dylan Missuwe
# All modifications by Dylan Missuwe are released under the GPL 3.0 Licence

import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib import pyplot as plot

import cv2 as cv

import svgwrite

from shapely.geometry import Polygon
from shapely.ops import unary_union

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
    #return { i : order_segment_list(segment_list) for i, segment_list in voronoi_cell_map.items() }
    return { i : segment_list for i, segment_list in voronoi_cell_map.items() }


# --- Plot all the things -----------------------------------------------------

def display(S, R, tri_list, voronoi_cell_map, ax):
    # Setup
    plot.axis('equal')
    plot.axis('off')

    # Set min/max display size, as Matplotlib does it wrong
    min_corner = np.amin(S, axis=0) - np.max(R)
    max_corner = np.amax(S, axis=0) + np.max(R)
    plot.xlim((min_corner[0], max_corner[0]))
    plot.ylim((min_corner[1], max_corner[1]))

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

    # Job done
    plot.show()

def get_weighted_centroid(A, U, tmin, tmax):
    tmid = (tmin + tmax) / 2.0
    return A + tmid * U

def is_infinite_triangle(tri):
    return any(idx == -1 for idx in tri)

# thx ChatGPT
def sample_color(img, point, size):
    x_min, x_max, y_min, y_max = size

    if point[0] < x_min:
        return [0,0,0]
    
    if point[0] > x_max:
        return [0,0,0]
    
    if point[1] < y_min:
        return [0,0,0]

    if point[1] > y_max:
        return [0,0,0]

    # Convert the floating point values to the corresponding pixel coordinates in the image
    x_pixel = int((point[0] - x_min) * img.shape[1] / (x_max - x_min))
    y_pixel = int((point[1] - y_min) * img.shape[0] / (y_max - y_min))

    # Apply boundary checks to ensure the pixel coordinates stay within the image
    x_pixel = max(0, min(x_pixel, img.shape[1] - 1))
    y_pixel = max(0, min(y_pixel, img.shape[0] - 1))

    # Calculate the relative coordinates within the pixel
    x_rel = (point[0] - x_min) / (x_max - x_min)
    y_rel = (point[1] - y_min) / (y_max - y_min)

    # Determine the four surrounding pixels for bilinear interpolation
    x0 = int(x_rel * (img.shape[1] - 1))
    x1 = min(x0 + 1, img.shape[1] - 1)
    y0 = int(y_rel * (img.shape[0] - 1))
    y1 = min(y0 + 1, img.shape[0] - 1)

    # Bilinear interpolation weights
    wx = x_rel * (img.shape[1] - 1) - x0
    wy = y_rel * (img.shape[0] - 1) - y0

    # Sample each color channel separately and interpolate
    r_channel = (1 - wx) * (1 - wy) * img[y0, x0, 2] + wx * (1 - wy) * img[y0, x1, 2] + (1 - wx) * wy * img[y1, x0, 2] + wx * wy * img[y1, x1, 2]
    g_channel = (1 - wx) * (1 - wy) * img[y0, x0, 1] + wx * (1 - wy) * img[y0, x1, 1] + (1 - wx) * wy * img[y1, x0, 1] + wx * wy * img[y1, x1, 1]
    b_channel = (1 - wx) * (1 - wy) * img[y0, x0, 0] + wx * (1 - wy) * img[y0, x1, 0] + (1 - wx) * wy * img[y1, x0, 0] + wx * wy * img[y1, x1, 0]

    # Combine the color channels into an RGB tuple
    interpolated_rgb = (b_channel, g_channel, r_channel)

    return interpolated_rgb

<<<<<<< HEAD
def voronoi_finite_polygons_2d(vor, radius=None, clip_box=None):
=======
# https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

>>>>>>> 6ad8a27517bf95967f9e65350e812b6265ecb91b
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Clip the region to the bounding box
        if clip_box is not None:
            bounding_box = Polygon([(clip_box[0], clip_box[2]), (clip_box[0], clip_box[3]),
                                    (clip_box[1], clip_box[3]), (clip_box[1], clip_box[2])])
            polygon_points = [new_vertices[v] for v in new_region]
            polygon = Polygon(polygon_points)
            clipped_polygon = polygon.intersection(bounding_box)
            if not clipped_polygon.is_empty:
                if clipped_polygon.geom_type == "Polygon":
                    new_region = [v for v in new_region if clipped_polygon.contains(Polygon(new_vertices[v]))]
                elif clipped_polygon.geom_type == "MultiPolygon":
                    new_region = [v for v in new_region if any(clipped_polygon.contains(Polygon(new_vertices[v])))
                                  for geom in clipped_polygon.geoms]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def save_voronoi_as_svg(regions, vertices, colors, file_name):
    # Create a new SVG drawing
    dwg = svgwrite.Drawing(file_name, profile='tiny')

    # Define the bounding box
    bbox = [-0.5, 0.5, -0.5, 0.5]

    # Iterate over the regions and draw the cropped polygons
    for i, region in enumerate(regions):
        polygon = vertices[region]

        cropped_polygon = []
        for x, y in polygon:
            # Ensure x and y coordinates are within the bounding box
            x = max(min(x, bbox[1]), bbox[0])
            y = max(min(y, bbox[3]), bbox[2])
            cropped_polygon.append((x, y))

        # Check if any point of the polygon is outside the bounding box
        outside_bbox = any(x < bbox[0] or x > bbox[1] or y < bbox[2] or y > bbox[3] for x, y in polygon)

        if not outside_bbox:
            dwg.add(dwg.polygon(points=cropped_polygon, fill=f"rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})"))

    # Save the cropped SVG drawing
    dwg.save()

def sample_colors_from_power_diagram(voronoi_cell_map, img, bounding_box):
    C_x = []
    C_y = []
    colors = []
    for point, segment_list in voronoi_cell_map.items():
        weighted_centroid = np.zeros(2)
        total_weight = 0.0
        for edge, (A, U, tmin, tmax) in segment_list:
            if tmin is None:
                tmin = -10
            if tmax is None:
                tmax = 10
            segment_centroid = get_weighted_centroid(A, U, tmin, tmax)
            segment_weight = abs(tmax - tmin)
            weighted_centroid += segment_centroid * segment_weight
            total_weight += segment_weight
        weighted_centroid /= total_weight
        C_x.append(weighted_centroid[0])
        C_y.append(weighted_centroid[1])

        # TODO: sample average pixel color within cell instead of only sampeling the centroid of the cell
        color = sample_color(img, [weighted_centroid[0], 0.00001-weighted_centroid[1]], bounding_box)
        colors.append(color)
    
    return colors


# --- Main entry point --------------------------------------------------------

def main():
    # Load your points, weights, and color image
    points_file = "/home/dylan/Olympic_Rings.dat"
    weight_file = "/home/dylan/Olympic_Rings.weight"
    img = cv.imread('/home/dylan/Olympic_Rings_Color.png') 

    size = [-0.5, 0.5, -0.5, 0.5]

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #fig, ax = plot.subplots()
    #ax.imshow(img, extent=size)

    print("Extracting points and weights from files..")

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

    print("Performing regular triangulation..")

    # Compute the power triangulation of the circles
    tri_list, V = get_power_triangulation(S, R)

    print("Generating power diagram..")

    # Compute the Voronoi cells
    voronoi_cell_map = get_voronoi_cells(S, V, tri_list)

    print("sample colors of image according to centroids of power diagram..")

    colors = sample_colors_from_power_diagram(voronoi_cell_map, img, size)

    print("drawing voronoi diagram with sampled colors..")
    
    colors = np.array(colors, dtype=float)
    colors /= 256.0

    # Generate Voronoi diagram
    vor = Voronoi(S)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    # colorize
    for i, region in enumerate(regions):
        polygon = vertices[region]
        plot.fill(*zip(*polygon), alpha=1, color=colors[i])

    save_voronoi_as_svg(regions, vertices, colors, '../cropped_polygons.svg')

    # Display the result
    plot.axis('equal')
    plot.xlim(-0.5, 0.5)
    plot.ylim(-0.5, 0.5)

    plot.show()

if __name__ == '__main__':
    main()
