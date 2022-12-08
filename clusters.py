import csv
import math
import matplotlib.pyplot as plt
import sys
import numpy
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


# CSV 1 - Solar/Wind Dataset
csv_filename = sys.argv[1]

csv_fields = []
csv_rows = []

with open(csv_filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	csv_fields = next(csvreader)
	csv_rows = [row for row in csvreader]

# Data is grouped into 4 rows, one for each corner of the cell
#  Only the center coordinate is needed, so only one row per cell is kept
csv_rows_cells = csv_rows[::4]

# Remove cells with no data
csv_rows_cells = [cell for cell in csv_rows_cells if cell[5]]


# CSV 2 - Population Data
pop_csv_filename = sys.argv[2]

pop_csv_fields = []
pop_csv_rows = []

with open(pop_csv_filename, 'r') as pop_csvfile:
	pop_csvreader = csv.reader(pop_csvfile)
	pop_csv_fields = next(pop_csvreader)
	pop_csv_rows = [row for row in pop_csvreader]


# Function to get the nearest county population of a cell
def get_cell_population(lat, long):
	nearst_population = -9999
	nearest_distance = 9999999
	for county in pop_csv_rows:
		distance = math.sqrt(pow(lat - float(county[5]), 2) + pow(long - float(county[6]), 2))
		if distance < nearest_distance:
			nearest_distance = distance
			nearst_population = float(county[4])
	return nearst_population


# Setup lists for clustering
#  x_coords: the corresponding lat and long of the data point in x
x_coords = [(float(pt[1]), float(pt[2])) for pt in csv_rows_cells]
#  x: the data point for each cell
x = [float(pt[5]) for pt in csv_rows_cells]
#  y: the nearest population value for the location of point x
y = [get_cell_population(pt[0], pt[1]) for pt in x_coords]


# Elbow method
data = list(zip(x, y))
inertias = []

for i in range(1, 11):
	kmeans = KMeans(n_clusters=i)
	kmeans.fit(data)
	inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_inertia.png')
plt.show()


# K-means clustering
kmeans = KMeans(n_clusters=4)
data = list(zip(x, y))
kmeans.fit(data)


# Output cluster scatter plot
colors = ['b', 'g', 'y', 'c', 'm', 'r', 'k']
k_colors = [colors[k] for k in kmeans.labels_]
plt.scatter(x, y, c=k_colors)
plt.title('Clusters')
plt.xlabel('Net Downward Shortwave Radiation (W/m^2)')
# plt.xlabel('Net Surface Wind Speed (m/s)')
plt.ylabel('Population')
plt.savefig('clusters.png')
plt.show()

# Output maps
for k in range(kmeans.n_clusters):

	# Add each cluster to the scatter plot, color coded
	pts_in_cluster = [x_coords[i] for i in range(len(x_coords)) if kmeans.labels_[i] == k]
	plt.scatter([x[1] for x in pts_in_cluster], [x[0] for x in pts_in_cluster], c=colors[k], marker='p', s=3)

	# Output map for each cluster
	osm = Image.open('osm.png', 'r')
	# Get pixel dimensions of cell
	cell_w = 0.1 / abs(-79.8926 - -66.9287) * osm.size[0]
	cell_h = 0.1 / abs(38.9766 - 47.6777) * osm.size[1]
	draw = ImageDraw.Draw(osm)
	for pt in pts_in_cluster:
		x_img = numpy.interp(pt[1], [-79.8926, -66.9287], [0, osm.size[0]])
		y_img = osm.size[1] - numpy.interp(pt[0], [38.9766, 47.6777], [0, osm.size[1]-50])
		draw.rectangle((x_img-(cell_w/2), y_img-(cell_h/2), x_img+(cell_w/2), y_img+(cell_h/2)), fill="red")
	osm.save('cluster_overlay_map-' + str(k) + '.png')

# Show the color coded scatter plot
plt.title('Cluster Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('cluster_map.png')
plt.show()
