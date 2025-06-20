import pygame
import math
from constants import *
from utils import line_intersection, smooth_track_points

class Track:
    def __init__(self, track_type=TrackType.OVAL, track_width=140):
        self.track_width = max(track_width, 120) 
        self.inner_points = []
        self.outer_points = []
        self.centerline = []
        self.track_type = track_type
        self.start_position = None
        self.start_angle = 0
        self.track_length = 0
        
        self.generate_track()
        
    def generate_track(self):
        if self.track_type == TrackType.OVAL:
            self.create_oval_track()
        elif self.track_type == TrackType.RECTANGLE:
            self.create_rectangle_track()
        elif self.track_type == TrackType.L_TRACK:
            self.create_l_track()
        elif self.track_type == TrackType.U_TRACK:
            self.create_u_track()
        elif self.track_type == TrackType.SIMPLE_CURVE:
            self.create_simple_curve_track()
        elif self.track_type == TrackType.DOUBLE_LOOP:
            self.create_double_loop_track()
        elif self.track_type == TrackType.TEST_TRACK:
            self.create_test_track()
            
        self.track_length = self.calculate_track_length()
            
    def create_oval_track(self):
        cx, cy = WIDTH/2, HEIGHT/2
        rx, ry = 200, 120
        
        num_points = 80
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
            self.centerline.append((x, y))
            
        self.centerline = smooth_track_points(self.centerline)
        self.generate_boundaries()
        self.start_position = (cx + rx, cy)
        self.start_angle = 90
        
    def create_rectangle_track(self):
        margin = 150
        corners = [
            (margin, margin),
            (WIDTH - margin, margin),
            (WIDTH - margin, HEIGHT - margin),
            (margin, HEIGHT - margin)
        ]
        
        corner_radius = 50
        points = []
        
        for i in range(len(corners)):
            curr = corners[i]
            next_corner = corners[(i + 1) % len(corners)]
            prev_corner = corners[(i - 1) % len(corners)]
            
            to_next = (next_corner[0] - curr[0], next_corner[1] - curr[1])
            from_prev = (curr[0] - prev_corner[0], curr[1] - prev_corner[1])
            
            len_next = math.sqrt(to_next[0]**2 + to_next[1]**2)
            len_prev = math.sqrt(from_prev[0]**2 + from_prev[1]**2)
            
            to_next = (to_next[0]/len_next, to_next[1]/len_next)
            from_prev = (from_prev[0]/len_prev, from_prev[1]/len_prev)
            
            for j in range(20):
                t = j / 20.0
                angle = math.pi/2 * t
                
                x = curr[0] - from_prev[0] * corner_radius * (1 - t) + to_next[0] * corner_radius * t
                y = curr[1] - from_prev[1] * corner_radius * (1 - t) + to_next[1] * corner_radius * t
                
                points.append((x, y))
                
            for j in range(10):
                t = j / 10.0
                x = curr[0] + to_next[0] * corner_radius + to_next[0] * (len_next - 2*corner_radius) * t
                y = curr[1] + to_next[1] * corner_radius + to_next[1] * (len_next - 2*corner_radius) * t
                points.append((x, y))
                
        self.centerline = smooth_track_points(points)
        self.generate_boundaries()
        self.start_position = ((corners[0][0] + corners[1][0])/2, corners[0][1])
        self.start_angle = 0
        
    def create_l_track(self):
        waypoints = [
            (150, 400),
            (150, 200),
            (150, 150), 
            (200, 150),
            (400, 150),
            (600, 150),
            (650, 150),
            (650, 200),
            (650, 400),
            (600, 400), 
            (400, 400),
            (200, 400)
        ]
        
        self.centerline = []
        for i in range(len(waypoints)):
            p1 = waypoints[i]
            p2 = waypoints[(i + 1) % len(waypoints)]
            
            for t in range(10):
                ratio = t / 10.0
                x = p1[0] * (1 - ratio) + p2[0] * ratio
                y = p1[1] * (1 - ratio) + p2[1] * ratio
                self.centerline.append((x, y))
                
        self.centerline = smooth_track_points(self.centerline, 0.2)
        self.generate_boundaries()
        self.start_position = waypoints[0]
        self.start_angle = -90
        
    def create_u_track(self):
        points = []
        
        left_x = 200
        right_x = 600
        top_y = 150
        bottom_y = 450
        
        num_vertical = 30
        for i in range(num_vertical):
            x = left_x
            y = bottom_y - i * ((bottom_y - top_y) / num_vertical)
            points.append((x, y))
        
        curve_points = 30
        for i in range(curve_points):
            t = i / (curve_points - 1)
            angle = math.pi - t * math.pi 
            x = (left_x + right_x) / 2 + ((right_x - left_x) / 2) * math.cos(angle)
            y = top_y
            points.append((x, y))
        
        for i in range(num_vertical):
            x = right_x
            y = top_y + i * ((bottom_y - top_y) / num_vertical)
            points.append((x, y))
        
        points.append((right_x, bottom_y + 50))
        points.append((left_x, bottom_y + 50))
        
        self.centerline = points
        self.generate_boundaries()
        self.start_position = (left_x, bottom_y - 50)
        self.start_angle = -90
        
    def create_simple_curve_track(self):
        points = []
        
        num_points = 80
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            
            r = 180 + 40 * math.sin(3 * t)
            x = WIDTH/2 + r * math.cos(t)
            y = HEIGHT/2 + 0.8 * r * math.sin(t)
            
            points.append((x, y))
            
        self.centerline = smooth_track_points(points, 0.2)
        self.generate_boundaries()
        self.start_position = (WIDTH/2 + 220, HEIGHT/2)
        self.start_angle = 90
        
    def create_double_loop_track(self):
        points = []

        center_x = WIDTH // 2
        center_y = HEIGHT // 2
        radius = 90
        separation = 2 * radius + 40  

        left_center_x = center_x - separation // 2
        right_center_x = center_x + separation // 2

        for i in range(45, 405): 
            angle = math.radians(i)
            x = left_center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        for i in range(15):
            t = i / 14
            x = (1 - t) * (left_center_x + radius) + t * (right_center_x - radius)
            y = center_y
            points.append((x, y))

        for i in range(45, 405): 
            angle = math.radians(-i)
            x = right_center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        for i in range(15):
            t = i / 14
            x = (1 - t) * (right_center_x - radius) + t * (left_center_x + radius)
            y = center_y
            points.append((x, y))

        self.centerline = smooth_track_points(points, 0.1)
        self.generate_boundaries()

        self.start_position = (left_center_x, center_y + radius)
        self.start_angle = -math.pi / 2
        
    def create_test_track(self):
        cx, cy = WIDTH/2, HEIGHT/2
        radius = 150
        
        num_points = 60
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            self.centerline.append((x, y))
            
        self.centerline = smooth_track_points(self.centerline)
        self.generate_boundaries()
        self.start_position = (cx + radius, cy)
        self.start_angle = 90
        
    def generate_boundaries(self):
        self.inner_points = []
        self.outer_points = []
        
        if len(self.centerline) < 3:
            return
            
        min_width = 120 
        actual_width = max(self.track_width, min_width)
            
        for i in range(len(self.centerline)):
            p1 = self.centerline[i]
            p2 = self.centerline[(i + 1) % len(self.centerline)]
            p0 = self.centerline[(i - 1) % len(self.centerline)]
            
            dx1 = p2[0] - p1[0]
            dy1 = p2[1] - p1[1]
            dx2 = p1[0] - p0[0]
            dy2 = p1[1] - p0[1]
            
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2
            
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0.001:  
                dx /= length
                dy /= length
                
                nx, ny = -dy, dx
                
                inner_x = p1[0] - nx * actual_width/2
                inner_y = p1[1] - ny * actual_width/2
                outer_x = p1[0] + nx * actual_width/2
                outer_y = p1[1] + ny * actual_width/2
                
                margin = 30
                inner_x = max(margin, min(WIDTH - margin, inner_x))
                inner_y = max(margin, min(HEIGHT - margin, inner_y))
                outer_x = max(margin, min(WIDTH - margin, outer_x))
                outer_y = max(margin, min(HEIGHT - margin, outer_y))
                
                self.inner_points.append((inner_x, inner_y))
                self.outer_points.append((outer_x, outer_y))
                
        self.inner_points = smooth_track_points(self.inner_points, 0.2)
        self.outer_points = smooth_track_points(self.outer_points, 0.2)
        
        self.verify_track_width(min_width * 0.9) 
        
    def verify_track_width(self, min_width):
        if len(self.inner_points) != len(self.outer_points):
            return
            
        for i in range(len(self.inner_points)):
            inner = self.inner_points[i]
            outer = self.outer_points[i]
            
            width = math.sqrt((outer[0] - inner[0])**2 + (outer[1] - inner[1])**2)
            
            if width < min_width:
                dx = outer[0] - inner[0]
                dy = outer[1] - inner[1]
                
                if width > 0:
                    dx /= width
                    dy /= width
                    
                    adjustment = (min_width - width) / 2
                    self.inner_points[i] = (inner[0] - dx * adjustment, inner[1] - dy * adjustment)
                    self.outer_points[i] = (outer[0] + dx * adjustment, outer[1] + dy * adjustment)
                
    def calculate_track_length(self):
        total_length = 0
        for i in range(len(self.centerline)):
            p1 = self.centerline[i]
            p2 = self.centerline[(i + 1) % len(self.centerline)]
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += dist
        return total_length
                                        
    def check_collision(self, car_corners):
        for corner in car_corners:
            if corner[0] < 0 or corner[0] > WIDTH or corner[1] < 0 or corner[1] > HEIGHT:
                return True
                
        car_edges = []
        for i in range(len(car_corners)):
            car_edges.append([car_corners[i], car_corners[(i + 1) % len(car_corners)]])
            
        for car_edge in car_edges:
            for i in range(len(self.inner_points)):
                inner_segment = [
                    self.inner_points[i],
                    self.inner_points[(i + 1) % len(self.inner_points)]
                ]
                outer_segment = [
                    self.outer_points[i],
                    self.outer_points[(i + 1) % len(self.outer_points)]
                ]
                
                if line_intersection(car_edge, inner_segment) or \
                   line_intersection(car_edge, outer_segment):
                    return True
                    
        return False
        
    def draw(self, surface):
        if len(self.outer_points) > 2 and len(self.inner_points) > 2:
            track_polygon = self.outer_points + self.inner_points[::-1]
            pygame.draw.polygon(surface, GRAY, track_polygon)
            
            if len(self.centerline) > 1:
                for i in range(0, len(self.centerline), 5):
                    if i + 1 < len(self.centerline):
                        pygame.draw.line(surface, DARK_GRAY, 
                                       (int(self.centerline[i][0]), int(self.centerline[i][1])),
                                       (int(self.centerline[i+1][0]), int(self.centerline[i+1][1])), 1)
            
        if len(self.inner_points) > 1:
            pygame.draw.lines(surface, WHITE, True, 
                            [(int(x), int(y)) for x, y in self.inner_points], 3)
        if len(self.outer_points) > 1:
            pygame.draw.lines(surface, WHITE, True, 
                            [(int(x), int(y)) for x, y in self.outer_points], 3)
            
        if self.start_position:
            start_x, start_y = int(self.start_position[0]), int(self.start_position[1])
            pattern_size = 10
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if (i + j) % 2 == 0:
                        rect = pygame.Rect(start_x + i * pattern_size - pattern_size//2,
                                         start_y + j * pattern_size - pattern_size//2,
                                         pattern_size, pattern_size)
                        pygame.draw.rect(surface, GREEN if j < 0 else WHITE, rect)