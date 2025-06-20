import pygame
import random
import numpy as np
import math
from constants import *
from car import Car
from track import Track

class GameEnvironment:
    def __init__(self, track_types=None):
        if track_types is None:
            track_types = [TrackType.OVAL, TrackType.RECTANGLE, 
                          TrackType.L_TRACK, TrackType.U_TRACK]
        
        self.track_types = track_types
        self.current_track_idx = 0
        self.tracks = [Track(track_type, track_width=140) for track_type in track_types]
        self.track = self.tracks[0]
        
        self.car = Car(self.track.start_position[0], 
                      self.track.start_position[1],
                      self.track.start_angle)
        
        self.episode_steps = 0
        self.max_steps = 2000
        self.done = False
        self.render_mode = "human"
        
        self.best_lap_distance = {}
        
        self.total_episodes = 0
        
    def reset(self, random_track=True):
        self.total_episodes += 1
        
        if random_track and len(self.tracks) > 1:
            self.current_track_idx = random.randint(0, len(self.tracks) - 1)
        
        self.track = self.tracks[self.current_track_idx]
        
        start_x = self.track.start_position[0] + random.uniform(-10, 10)
        start_y = self.track.start_position[1] + random.uniform(-10, 10)
        start_angle = self.track.start_angle + random.uniform(-10, 10)
        
        self.car.reset(start_x, start_y, start_angle)
        self.episode_steps = 0
        self.done = False
        
        self.car.cast_sensors(self.track)
        
        safety_attempts = 0
        while self.track.check_collision(self.car.get_corners()) and safety_attempts < 10:
            self.car.x = self.track.start_position[0]
            self.car.y = self.track.start_position[1]
            self.car.angle = self.track.start_angle
            self.car.cast_sensors(self.track)
            safety_attempts += 1
            
        return self.get_state()
        
    def step(self, action):
        self.episode_steps += 1
        
        prev_distance = self.car.distance_traveled
        prev_avg_speed = self.car.avg_speed
        
        self.car.update(action=action)
        
        sensor_lines = self.car.cast_sensors(self.track)
        
        self.car.collided = self.track.check_collision(self.car.get_corners())
        
        reward = self.calculate_reward(prev_distance, prev_avg_speed)
        
        if self.car.collided:
            self.done = True
            
        if self.car.stuck_counter > 50:
            self.done = True
            reward -= 5
            
        if self.episode_steps >= self.max_steps:
            self.done = True
            reward += 10
            
        return self.get_state(), reward, self.done
        
    def calculate_reward(self, prev_distance, prev_avg_speed):
        reward = 0
        
        min_sensor = min(self.car.sensor_readings)
        if min_sensor > 0.3:  
            distance_delta = self.car.distance_traveled - prev_distance
            reward += distance_delta * 0.5
        
        if min_sensor < 0.2:
            reward -= 5.0
        elif min_sensor < 0.4:
            reward -= 1.0
        else:
            reward += 0.5  
        
        if min_sensor < 0.5 and self.car.speed > 3.0:
            reward -= 2.0
        
        if self.car.collided:
            reward -= 50  
            
        return reward
        
    def get_state(self):
        state = []
        
        state.extend(self.car.sensor_readings)
        
        state.append(self.car.speed / self.car.max_speed)
        
        state.append(self.car.avg_speed / self.car.max_speed)
        
        angle_rad = math.radians(self.car.angle)
        state.append(math.sin(angle_rad))
        state.append(math.cos(angle_rad))
        
        front_sensors = self.car.sensor_readings[6:9]
        state.append(sum(front_sensors) / len(front_sensors))
        
        left_avg = sum(self.car.sensor_readings[:5]) / 5
        right_avg = sum(self.car.sensor_readings[-5:]) / 5
        state.append(left_avg - right_avg)  
        
        state.append(1.0 if min(self.car.sensor_readings) < 0.2 else 0.0)
        state.append(1.0 if self.car.sensor_readings[7] < 0.3 else 0.0)  
        
        state.append(min(1.0, self.car.stuck_counter / 20.0))
        
        return np.array(state, dtype=np.float32)
        
    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode
            
        if self.render_mode == "headless":
            return True
            
        screen.fill(BLACK)
        
        self.track.draw(screen)
        
        sensor_lines = self.car.cast_sensors(self.track)
        show_all_sensors = self.episode_steps < 60  
        self.car.draw(screen, sensor_lines, show_all_sensors)
        
        font = pygame.font.SysFont(None, 24)
        
        track_text = font.render(f"Track: {self.track.track_type.value}", True, WHITE)
        screen.blit(track_text, (10, 10))
        
        speed_text = font.render(f"Speed: {self.car.speed:.1f} / Avg: {self.car.avg_speed:.1f}", True, WHITE)
        screen.blit(speed_text, (10, 40))
        
        dist_text = font.render(f"Distance: {self.car.distance_traveled:.0f}", True, WHITE)
        screen.blit(dist_text, (10, 70))
        
        steps_text = font.render(f"Steps: {self.episode_steps}", True, WHITE)
        screen.blit(steps_text, (10, 100))
        
        min_sensor = min(self.car.sensor_readings)
        if min_sensor < 0.3:
            warning_text = font.render(f"WARNING: Wall proximity {min_sensor:.2f}", True, ORANGE)
            screen.blit(warning_text, (10, 130))
        
        if self.car.collided:
            collision_text = font.render("COLLISION!", True, RED)
            screen.blit(collision_text, (WIDTH//2 - 50, HEIGHT//2))
            
        if self.car.stuck_counter > 20:
            stuck_text = font.render(f"STUCK: {self.car.stuck_counter}", True, YELLOW)
            screen.blit(stuck_text, (10, 160))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
                elif event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                paused = False
                            elif event.type == pygame.QUIT:
                                pygame.quit()
                                return False
                        clock.tick(30)
                    
        pygame.display.flip()
        clock.tick(FPS)
        
        return True