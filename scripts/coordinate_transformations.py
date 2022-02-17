

def world_to_grid(x,y,origin_x,origin_y,width,height,resolution):
    if x < origin_x or y < origin_y or x > origin_x+width or y > origin_y+height:
        return None
    else:
        i = int((x-origin_x)/resolution)
        j = int((y-origin_y)/resolution)
        return (i, j)
    
    
def grid_to_world(gx: int,gy: int,origin_x,origin_y,width,height,resolution):
    if gx < 0 or gy < 0 or gx > width/resolution or gy > height/resolution:
        return None
    else:
        x = origin_x + (gx + 0.5)*resolution
        y = origin_y + (gy + 0.5)*resolution
        return (x, y)  