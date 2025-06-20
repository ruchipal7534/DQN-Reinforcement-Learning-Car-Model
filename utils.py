def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
    
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    if abs(denominator) < 1e-10:
        return None
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
    
    if ua < 0 or ua > 1 or ub < 0 or ub > 1:
        return None
        
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return (x, y)

def smooth_track_points(points, smoothing_factor=0.2):
    if len(points) < 3:
        return points
        
    smoothed = []
    for i in range(len(points)):
        prev_idx = (i - 1) % len(points)
        next_idx = (i + 1) % len(points)
        
        x = (points[prev_idx][0] * smoothing_factor + 
             points[i][0] * (1 - 2 * smoothing_factor) + 
             points[next_idx][0] * smoothing_factor)
        y = (points[prev_idx][1] * smoothing_factor + 
             points[i][1] * (1 - 2 * smoothing_factor) + 
             points[next_idx][1] * smoothing_factor)
        
        smoothed.append((x, y))
    
    return smoothed