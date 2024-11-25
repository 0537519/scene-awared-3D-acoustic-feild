from options import Options
import os
import pickle



with open("D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/minmax/frl_apartment_4_minmax", "rb") as min_max_loader:
    min_maxes = pickle.load(min_max_loader)
    min_pos = min_maxes[0]  # Only x and y coordinates
    max_pos = min_maxes[1]  # Only x and y coordinates

# Calculate x and y lengths for the room as defined by min_max_loader
x_length_minmax = max_pos[0] - min_pos[0]
y_length_minmax = max_pos[2] - min_pos[2]
print(f"Room length in x-direction (min_max_loader): {x_length_minmax}")
print(f"Room length in y-direction (min_max_loader): {y_length_minmax}")

# Path to points file
points_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/replica/frl_apartment_4/points.txt"

# Initialize min and max x, y values for the points in points_path
min_x = float("inf")
max_x = float("-inf")
min_y = float("inf")
max_y = float("-inf")

# Read points from points_path and update min and max values for x, y
with open(points_path, "r") as f:
    for line in f:
        data = line.strip().split()
        if len(data) == 4:
            x, y, z = float(data[1]), float(data[2]),float(data[3])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

# Calculate x and y lengths for the points in points_path
x_length_points = max_x - min_x
y_length_points = max_y - min_y
print(f"Room length in x-direction (points_path): {x_length_points}")
print(f"Room length in y-direction (points_path): {y_length_points}")

# Check if lengths from points_path exceed min_max_loader; if so, raise an error
if x_length_points > x_length_minmax or y_length_points > y_length_minmax:
    raise ValueError("Error: Room dimensions in min_max_loader are smaller than in points_path.")

# Calculate midpoints for x and y in points_path
x_midpoint_points = (max_x + min_x) / 2
y_midpoint_points = (max_y + min_y) / 2

# Adjust min_max_loader bounds based on the calculated midpoints while keeping lengths unchanged
new_min_x = x_midpoint_points - x_length_minmax / 2
new_max_x = x_midpoint_points + x_length_minmax / 2
new_min_y = y_midpoint_points - y_length_minmax / 2
new_max_y = y_midpoint_points + y_length_minmax / 2

# Create new min and max positions maintaining the original format
new_min_pos = min_pos.copy()
new_max_pos = max_pos.copy()
new_min_pos[0] = new_min_x
new_min_pos[2] = new_min_y
new_max_pos[0] = new_max_x
new_max_pos[2] = new_max_y

print(new_min_pos.shape)

# Create new minmax array with the same format as the original
new_min_maxes = [new_min_pos, new_max_pos]

# Save the adjusted min_max_loader to the specified file path
minmax_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/minmax_2/frl_apartment_4_minmax"
with open(minmax_path, "wb") as f:
    pickle.dump(new_min_maxes, f)

print(f"Adjusted min_max_loader saved to {minmax_path}")

