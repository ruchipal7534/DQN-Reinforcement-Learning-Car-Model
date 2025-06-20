import pygame
import math
from collections import deque
from constants import *
from utils import line_intersection

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.max_speed = 7.0
        self.min_speed = -2.0
        self.acceleration = 0.3
        self.brake_deceleration = 0.5
        self.rotation_speed = 4.0
        self.friction = 0.1
        self.direction = pygame.Vector2(0, -1)
        self.velocity = pygame.Vector2(0, 0)
        self.orig_image = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
        pygame.draw.rect(self.orig_image, RED, (0, 0, CAR_SIZE[0], CAR_SIZE[1]))
        pygame.draw.rect(self.orig_image, BLACK, (0, 0, CAR_SIZE[0], CAR_SIZE[1]), 2)
        pygame.draw.rect(self.orig_image, WHITE, (5, 0, 10, 8))
        self.image = self.orig_image
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.collided = False
        self.sensor_angles = [-90, -75, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 75, 90]
        self.sensor_length = 120
        self.sensor_readings = [0] * len(self.sensor_angles)
        self.distance_traveled = 0
        self.time_alive = 0
        self.last_position = (x, y)
        self.stuck_counter = 0
        self.avg_speed = 0
        self.speed_samples = deque(maxlen=30)
        
        self.lap_times = []
        self.best_lap_time = float('inf')
        self.current_lap_time = 0
        self.lap_start_time = 0
        self.max_distance_this_session = 0
        
        self.prev_velocity = pygame.Vector2(0, 0)
        self.acceleration_vector = pygame.Vector2(0, 0)
        self.angular_velocity = 0
        self.prev_angle = angle
        self.g_force = 0
        self.turning_radius = float('inf')
        self.distance_from_center = 0
        
    def get_corners(self):
        cos_val = math.cos(math.radians(self.angle))
        sin_val = math.sin(math.radians(self.angle))
        half_width = CAR_SIZE[0] / 2
        half_height = CAR_SIZE[1] / 2
        corners_local = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        corners_world = []
        for corner_x, corner_y in corners_local:
            rotated_x = corner_x * cos_val - corner_y * sin_val
            rotated_y = corner_x * sin_val + corner_y * cos_val
            world_x = rotated_x + self.x
            world_y = rotated_y + self.y
            corners_world.append((world_x, world_y))
            
        return corners_world
        
    def update(self, action=None, keys_pressed=None):
        self.direction = pygame.Vector2(
            math.sin(math.radians(self.angle)),
            -math.cos(math.radians(self.angle))
        )
        
        if action is not None:
            steer = action.get('steer', 0)
            accelerate = action.get('accelerate', 0)
            
            if accelerate > 0:
                self.speed += self.acceleration * accelerate
            else:
                self.speed += self.brake_deceleration * accelerate
            
            if abs(self.speed) > 0.5:
                steer_effectiveness = 1.0 - min(0.5, abs(self.speed) / 10.0)
                self.angle += self.rotation_speed * steer * steer_effectiveness
        
        elif keys_pressed is not None:
            if keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
                self.speed += self.acceleration
            if keys_pressed[pygame.K_DOWN] or keys_pressed[pygame.K_s]:
                self.speed -= self.brake_deceleration
                
            if abs(self.speed) > 0.5:
                if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
                    self.angle -= self.rotation_speed
                if keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
                    self.angle += self.rotation_speed
        
        if self.speed > 0:
            self.speed = max(0, self.speed - self.friction)
        elif self.speed < 0:
            self.speed = min(0, self.speed + self.friction)
            
        self.speed = max(self.min_speed, min(self.max_speed, self.speed))
        
        movement = self.direction * self.speed
        self.velocity = movement
        
        prev_pos = (self.x, self.y)
        self.x += self.velocity.x
        self.y += self.velocity.y
        
        dist = math.sqrt((self.x - prev_pos[0])**2 + (self.y - prev_pos[1])**2)
        self.distance_traveled += dist
        self.max_distance_this_session = max(self.max_distance_this_session, self.distance_traveled)
        
        self.acceleration_vector = self.velocity - self.prev_velocity
        self.prev_velocity = self.velocity.copy()
        
        angle_diff = self.angle - self.prev_angle
        
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        self.angular_velocity = angle_diff
        self.prev_angle = self.angle
        
        self.g_force = self.acceleration_vector.magnitude()
        
        if abs(self.angular_velocity) > 0.1 and abs(self.speed) > 0.1:
            angular_vel_rad = math.radians(abs(self.angular_velocity))
            self.turning_radius = abs(self.speed) / angular_vel_rad
        else:
            self.turning_radius = float('inf')
        
        self.speed_samples.append(abs(self.speed))
        self.avg_speed = sum(self.speed_samples) / len(self.speed_samples) if self.speed_samples else 0
        
        if dist < 0.1 and abs(self.speed) < 0.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)
            
        self.last_position = (self.x, self.y)
        self.time_alive += 1
        self.current_lap_time += 1
        
        self.image = pygame.transform.rotate(self.orig_image, -self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))
        
    def cast_sensors(self, track):
        sensor_lines = []
        self.sensor_readings = []
        
        for angle_offset in self.sensor_angles:
            sensor_angle = self.angle + angle_offset
            rad_angle = math.radians(sensor_angle)
            
            end_x = self.x + self.sensor_length * math.sin(rad_angle)
            end_y = self.y - self.sensor_length * math.cos(rad_angle)
            
            sensor_line = [(self.x, self.y), (end_x, end_y)]
            sensor_lines.append(sensor_line)
            
            min_distance = self.sensor_length
            
            for i in range(len(track.inner_points)):
                inner_segment = [
                    track.inner_points[i],
                    track.inner_points[(i + 1) % len(track.inner_points)]
                ]
                outer_segment = [
                    track.outer_points[i],
                    track.outer_points[(i + 1) % len(track.outer_points)]
                ]
                
                for segment in [inner_segment, outer_segment]:
                    intersection = line_intersection(sensor_line, segment)
                    if intersection:
                        distance = math.sqrt((intersection[0] - self.x)**2 + 
                                           (intersection[1] - self.y)**2)
                        min_distance = min(min_distance, distance)
            
            normalized_reading = min_distance / self.sensor_length
            self.sensor_readings.append(normalized_reading)
        
        if track.centerline:
            min_center_dist = float('inf')
            for center_point in track.centerline:
                dist_to_center = math.sqrt((self.x - center_point[0])**2 + (self.y - center_point[1])**2)
                min_center_dist = min(min_center_dist, dist_to_center)
            self.distance_from_center = min_center_dist
            
        return sensor_lines
        
    def draw(self, surface, sensor_lines=None, show_sensors=True):
        surface.blit(self.image, self.rect)
        
        if sensor_lines and show_sensors:
            for i, line in enumerate(sensor_lines):
                reading = self.sensor_readings[i]
                if reading < 0.3:
                    color = RED
                elif reading < 0.6:
                    color = YELLOW
                else:
                    color = GREEN
                    
                if i % 3 == 0 or show_sensors: 
                    pygame.draw.line(surface, color, line[0], line[1], 1)
                
    def reset(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.velocity = pygame.Vector2(0, 0)
        self.collided = False
        self.distance_traveled = 0
        self.time_alive = 0
        self.stuck_counter = 0
        self.last_position = (x, y)
        self.avg_speed = 0
        self.speed_samples.clear()
        self.current_lap_time = 0
        self.lap_start_time = pygame.time.get_ticks()
        
        self.prev_velocity = pygame.Vector2(0, 0)
        self.acceleration_vector = pygame.Vector2(0, 0)
        self.angular_velocity = 0
        self.prev_angle = angle
        self.g_force = 0
        self.turning_radius = float('inf')
        self.distance_from_center = 0