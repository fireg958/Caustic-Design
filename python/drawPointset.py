import matplotlib.pyplot as plt

# Read the pointset data from the file
filename = 'voronoi_centroids.dat'
with open(filename, 'r') as file:
    lines = file.readlines()

# Extract the x and y coordinates from each line
x_coords = []
y_coords = []
for line in lines:
    x, y = map(float, line.strip().split())
    x_coords.append(x)
    y_coords.append(y)

# Plot the points
plt.scatter(x_coords, y_coords, s=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pointset')
plt.grid(True)
plt.show()