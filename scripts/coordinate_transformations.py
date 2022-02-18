"""Functions to transform world coordinates to grid coordinates and back."""

def world_to_grid(x,y,origin_x,origin_y,width,height,resolution):
    """Returns grid cell from given world coordinates.

    Args:
        x (float): positon in world coordinates
        y (float): positon in world coordinates
        origin_x: defining the bottom left corner of the grid in world coordinates
        origin_y: defining the bottom left corner of the grid in world coordinates
        width: width of map in world units
        height: height of map in world units
        resolution: the size of each grid cell in world units

    Returns:
        tuple: (i, j) index positions in the grid
        None if input is out of bounds.
    """

    if x < origin_x or y < origin_y or x > origin_x+width or y > origin_y+height:
        return None
    else:
        i = int((x-origin_x)/resolution)
        j = int((y-origin_y)/resolution)
        return (i, j)
    
    
def grid_to_world(gx: int,gy: int,origin_x,origin_y,width,height,resolution):
    """Is given position in the grid and returns position in world coordinates if in bounds of the map, else returns None.

    Args:
        gx: x position in grid
        gy: y position in grid
        origin_x: defining the bottom left corner of the grid in world coordinates
        origin_y: defining the bottom left corner of the grid in world coordinates
        width: width of map in world units
        height: height of map in world units
        resolution: the size (under the assumption of square sized grid cells) of each grid cell in world units

    Returns:
        tuple: (x, y) - centre position in the given grid cell in world coordinates.
        None if input is out of grid.
    """
    if gx < 0 or gy < 0 or gx > width/resolution or gy > height/resolution:
        return None
    else:
        x = origin_x + (gx + 0.5)*resolution
        y = origin_y + (gy + 0.5)*resolution
        return (x, y)