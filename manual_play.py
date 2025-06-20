import pygame
import math
from constants import *
from car import Car
from track import Track

class ManualPlaySession:
    def __init__(self):
        self.reset_stats()
        
    def reset_stats(self):
        self.sessions_played = 0
        self.total_distance = 0
        self.best_distance = 0
        self.crashes = 0
        self.best_lap_time = float('inf')
        self.session_start_time = pygame.time.get_ticks()
        self.current_track = None
        
    def update_stats(self, car, crashed=False):
        if crashed:
            self.crashes += 1
        
        if car.distance_traveled > self.best_distance:
            self.best_distance = car.distance_traveled
            
        if car.lap_times and min(car.lap_times) < self.best_lap_time:
            self.best_lap_time = min(car.lap_times)

def draw_track_selection_menu():
    screen.fill(BLACK)
    
    title_font = pygame.font.SysFont(None, 48)
    title_text = title_font.render("Select Track", True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH//2, 80))
    screen.blit(title_text, title_rect)
    
    font = pygame.font.SysFont(None, 32)
    tracks = [
        (TrackType.OVAL, "1. Oval Track (Beginner)"),
        (TrackType.RECTANGLE, "2. Rectangle Track (Easy)"),
        (TrackType.L_TRACK, "3. L-Shape Track (Medium)"),
        (TrackType.U_TRACK, "4. U-Shape Track (Medium)"),
        (TrackType.SIMPLE_CURVE, "5. Curved Track (Hard)"),
        (TrackType.DOUBLE_LOOP, "6. Double Loop (Expert)"),
        (TrackType.TEST_TRACK, "7. Test Track (Simple)")
    ]
    
    y_start = 150
    for i, (track_type, description) in enumerate(tracks):
        text = font.render(description, True, WHITE)
        text_rect = text.get_rect(center=(WIDTH//2, y_start + i * 40))
        screen.blit(text, text_rect)
    
    instruction_font = pygame.font.SysFont(None, 24)
    instructions = [
        "Press number key to select track",
        "ESC to return to main menu"
    ]
    
    for i, instruction in enumerate(instructions):
        text = instruction_font.render(instruction, True, LIGHT_BLUE)
        text_rect = text.get_rect(center=(WIDTH//2, HEIGHT - 80 + i * 25))
        screen.blit(text, text_rect)
    
    pygame.display.flip()

def manual_play_mode():
    print("Entering Manual Play Mode...")
    
    session = ManualPlaySession()
    current_track_type = None
    track = None
    car = None
    
    while True:
        draw_track_selection_menu()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                elif event.key == pygame.K_1:
                    current_track_type = TrackType.OVAL
                elif event.key == pygame.K_2:
                    current_track_type = TrackType.RECTANGLE
                elif event.key == pygame.K_3:
                    current_track_type = TrackType.L_TRACK
                elif event.key == pygame.K_4:
                    current_track_type = TrackType.U_TRACK
                elif event.key == pygame.K_5:
                    current_track_type = TrackType.SIMPLE_CURVE
                elif event.key == pygame.K_6:
                    current_track_type = TrackType.DOUBLE_LOOP
                elif event.key == pygame.K_7:
                    current_track_type = TrackType.TEST_TRACK
                
                if current_track_type:
                    track = Track(current_track_type, track_width=140)
                    car = Car(track.start_position[0], track.start_position[1], track.start_angle)
                    session.current_track = current_track_type
                    break
        
        if current_track_type:
            break
        
        clock.tick(30)
    
    print(f"Selected track: {current_track_type.value}")
    print("Controls: Arrow keys or WASD to drive")
    print("R to reset, T to change track, ESC to exit")
    
    running = True
    paused = False
    show_sensors = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset(track.start_position[0], track.start_position[1], track.start_angle)
                    print("Car reset!")
                elif event.key == pygame.K_t:
                    return manual_play_mode()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_h:
                    show_sensors = not show_sensors
                    print(f"Sensors {'shown' if show_sensors else 'hidden'}")
        
        if paused:
            font = pygame.font.SysFont(None, 48)
            pause_text = font.render("PAUSED - Press SPACE to continue", True, YELLOW)
            pause_rect = pause_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(pause_text, pause_rect)
            pygame.display.flip()
            clock.tick(30)
            continue
        
        keys = pygame.key.get_pressed()
        
        car.update(keys_pressed=keys)
        
        sensor_lines = car.cast_sensors(track)
        
        car.collided = track.check_collision(car.get_corners())
        
        if car.collided:
            session.update_stats(car, crashed=True)
            print(f"Crashed! Distance: {car.distance_traveled:.0f}")
            
            car.reset(track.start_position[0], track.start_position[1], track.start_angle)
        
        session.update_stats(car)
        
        screen.fill(BLACK)
        
        track.draw(screen)
        
        car.draw(screen, sensor_lines, show_sensors)
        
        font = pygame.font.SysFont(None, 24)
        
        stats = [
            f"Track: {track.track_type.value.title()}",
            f"Speed: {car.speed:.1f} / Max: {car.max_speed}",
            f"Distance: {car.distance_traveled:.0f}",
            f"Best Distance: {session.best_distance:.0f}",
            f"Session Crashes: {session.crashes}",
            f"Time: {car.current_lap_time // 60:.0f}s"
        ]
        
        for i, stat in enumerate(stats):
            text = font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 25))
        
        tech_data = [
            "Technical Data:",
            f"Position: ({car.x:.1f}, {car.y:.1f})",
            f"Velocity: ({car.velocity.x:.2f}, {car.velocity.y:.2f})",
            f"Speed: {abs(car.speed):.3f} u/frame",
            f"Angle: {car.angle:.1f}° ({math.radians(car.angle):.3f} rad)",
            f"Angular Vel: {car.angular_velocity:.2f}°/frame",
            "",
            "Physics Constants:",
            f"Friction: {car.friction:.3f}",
            f"Acceleration: {car.acceleration:.3f}",
            f"Max Speed: {car.max_speed:.1f}",
            f"Rotation Speed: {car.rotation_speed:.1f}°/frame",
            "",
            "Calculated Physics:",
            f"G-Force: {car.g_force:.3f}",
            f"Turn Radius: {car.turning_radius:.1f}" if car.turning_radius != float('inf') else "Turn Radius: ∞",
            f"Center Dist: {car.distance_from_center:.1f}",
            f"Accel Vector: ({car.acceleration_vector.x:.3f}, {car.acceleration_vector.y:.3f})",
            "",
            "Sensor Data:",
            f"Front: {car.sensor_readings[7]:.3f}" if car.sensor_readings else "Front: N/A",
            f"Left: {car.sensor_readings[2]:.3f}" if car.sensor_readings else "Left: N/A", 
            f"Right: {car.sensor_readings[12]:.3f}" if car.sensor_readings else "Right: N/A",
            f"Min Distance: {min(car.sensor_readings):.3f}" if car.sensor_readings else "Min: N/A",
            f"Sensor Count: {len(car.sensor_readings)}",
            "",
            "State Info:",
            f"Stuck Counter: {car.stuck_counter}",
            f"Time Alive: {car.time_alive}",
            f"Speed Samples: {len(car.speed_samples)}/30"
        ]
        
        line_height = 16
        for i, data in enumerate(tech_data):
            if data == "":  
                continue
            color = LIGHT_BLUE if data.endswith(":") else WHITE
            font_size = 20 if data.endswith(":") else 16
            data_font = pygame.font.SysFont(None, font_size)
            text = data_font.render(data, True, color)
            screen.blit(text, (WIDTH - 280, 10 + i * line_height))
        
        if car.sensor_readings:
            min_sensor = min(car.sensor_readings)
            if min_sensor < 0.3:
                warning_text = font.render(f"WARNING: Wall proximity {min_sensor:.2f}", True, ORANGE)
                screen.blit(warning_text, (10, 170))
        
        if car.speed > 5.0:
            speed_warning = font.render("HIGH SPEED!", True, RED)
            screen.blit(speed_warning, (10, 195))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    print(f"\nSession Summary:")
    print(f"Track: {session.current_track.value}")
    print(f"Best Distance: {session.best_distance:.0f}")
    print(f"Total Crashes: {session.crashes}")