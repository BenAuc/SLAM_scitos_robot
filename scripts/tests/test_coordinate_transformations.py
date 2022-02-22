# script for testing

# this works in VS Code
import sys
sys.path.append("scripts")
import coordinate_transformations as ct
# this should work anywhere else 
# from .. import coordinate_transformations as ct

width = 4
height = 6
resolution = 2 #size of a grid cell
origin_x = 0
origin_y = 0

print(ct.world_to_grid(4, 6, origin_x, origin_y, width, height, resolution))
print(ct.grid_to_world(1, 2, origin_x, origin_y, width, height, resolution))