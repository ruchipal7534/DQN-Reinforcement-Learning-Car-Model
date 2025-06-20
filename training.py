import os
import numpy as np
import pygame
from constants import *
from environment import GameEnvironment
from dqn_agent import DQNAgent
from track import Track

def train_multi_track(num_episodes=1000, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    very_easy_tracks = [TrackType.OVAL]
    easy_tracks = [TrackType.RECTANGLE]
    medium_tracks = [TrackType.L_TRACK, TrackType.SIMPLE_CURVE]
    hard_tracks = [TrackType.U_TRACK, TrackType.DOUBLE_LOOP]
    
    env = GameEnvironment(very_easy_tracks)
    
    state_size = 24  
    agent = DQNAgent(state_size, lr=0.00003)  
    
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    best_avg_reward = -float('inf')
    curriculum_stage = 0
    
    warmup_episodes = 10
    
    for episode in range(num_episodes):
        if episode == 50 and curriculum_stage == 0:
            print("Adding rectangle track...")
            env.track_types.extend(easy_tracks)
            env.tracks.extend([Track(t, track_width=140) for t in easy_tracks])
            curriculum_stage = 1
            
        elif episode == 150 and curriculum_stage == 1:
            print("Adding medium difficulty tracks...")
            env.track_types.extend(medium_tracks)
            env.tracks.extend([Track(t, track_width=140) for t in medium_tracks])
            curriculum_stage = 2
            
        elif episode == 300 and curriculum_stage == 2:
            print("Adding hard difficulty tracks...")
            env.track_types.extend(hard_tracks)
            env.tracks.extend([Track(t, track_width=140) for t in hard_tracks])
            curriculum_stage = 3
            
        state = env.reset(random_track=True)
        total_reward = 0
        
        render_mode = "human" if episode % 50 == 0 else "headless"
        env.render(mode=render_mode)
        
        for step in range(env.max_steps):
            action, action_idx = agent.act(state)
            
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action_idx, reward, next_state, done)
            
            if episode >= warmup_episodes and step % 4 == 0 and len(agent.memory) > agent.batch_size * 2:
                loss = agent.replay()
                
            state = next_state
            total_reward += reward
            
            if render_mode == "human":
                if not env.render():
                    return agent
                    
            if done:
                break
                
        episode_rewards.append(total_reward)
        episode_lengths.append(env.episode_steps)
        episode_distances.append(env.car.distance_traveled)
        
        if len(episode_rewards) >= 20:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_distance = np.mean(episode_distances[-20:])
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(f"{save_dir}/best_model.pt")
                print(f"New best model saved! Avg reward: {avg_reward:.2f}")
                
        print(f"Episode {episode+1}/{num_episodes} - "
              f"Track: {env.track.track_type.value} - "
              f"Reward: {total_reward:.2f} - "
              f"Distance: {env.car.distance_traveled:.0f} - "
              f"Steps: {env.episode_steps} - "
              f"Epsilon: {agent.epsilon:.3f}")
              
        if (episode + 1) % 100 == 0:
            agent.save(f"{save_dir}/checkpoint_{episode+1}.pt")
            
    agent.save(f"{save_dir}/final_model.pt")
    
    print("\nTraining Summary:")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Total experiences collected: {len(agent.memory)}")
    
    return agent

def test_on_new_track(model_path='models/best_model.pt', num_tests=5):
    env = GameEnvironment([TrackType.U_TRACK])
    
    state_size = 24
    agent = DQNAgent(state_size)
    
    if not agent.load(model_path):
        print("Failed to load model!")
        return
        
    agent.epsilon = 0  
    
    print(f"\nTesting on {num_tests} runs of the test track...")
    
    test_results = []
    
    for test in range(num_tests):
        state = env.reset(random_track=False)
        total_reward = 0
        
        print(f"\nTest {test+1}/{num_tests} on test track...")
        
        while not env.done:
            action, _ = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            
            if not env.render():
                break
                
        test_results.append({
            'reward': total_reward,
            'distance': env.car.distance_traveled,
            'steps': env.episode_steps,
            'completed': not env.car.collided
        })
        
        print(f"Result - Reward: {total_reward:.2f}, "
              f"Distance: {env.car.distance_traveled:.0f}, "
              f"Steps: {env.episode_steps}, "
              f"Status: {'Completed' if not env.car.collided else 'Crashed'}")
              
    print("\nTest Summary:")
    print(f"Average reward: {np.mean([r['reward'] for r in test_results]):.2f}")
    print(f"Average distance: {np.mean([r['distance'] for r in test_results]):.0f}")
    print(f"Success rate: {sum(r['completed'] for r in test_results) / num_tests * 100:.1f}%")

def visualize_all_tracks():
    print("Visualizing all track designs...")
    
    track_types = [
        TrackType.OVAL,
        TrackType.RECTANGLE,
        TrackType.L_TRACK,
        TrackType.U_TRACK,
        TrackType.SIMPLE_CURVE,
        TrackType.DOUBLE_LOOP,
        TrackType.TEST_TRACK
    ]
    
    for track_type in track_types:
        track = Track(track_type, track_width=140)
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Track Visualization: {track_type.value}")
        
        screen.fill(BLACK)
        
        track.draw(screen)
        
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Track: {track_type.value}", True, WHITE)
        screen.blit(text, (10, 10))
        
        info_font = pygame.font.SysFont(None, 24)
        info_text = info_font.render(f"Width: {track.track_width}px, Press SPACE for next", True, WHITE)
        screen.blit(info_text, (10, 50))
        
        if track.start_position:
            car_rect = pygame.Rect(track.start_position[0] - 10, 
                                 track.start_position[1] - 20, 20, 40)
            rotated_car = pygame.transform.rotate(pygame.Surface((20, 40)), -track.start_angle)
            rotated_car.fill(RED)
            screen.blit(rotated_car, car_rect)
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                        
        pygame.time.wait(100)
    
    print("Track visualization complete!")

def test_single_track(track_type=TrackType.DOUBLE_LOOP):
    print(f"Testing track: {track_type.value}")
    
    env = GameEnvironment([track_type])
    
    state = env.reset(random_track=False)
    
    print("Use arrow keys or WASD to drive. Press ESC to exit.")
    
    running = True
    while running:
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            accel = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            accel = -1.0
        else:
            accel = 0.0
            
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer = 1.0
        else:
            steer = 0.0
            
        action = {'steer': steer, 'accelerate': accel}
        
        state, reward, done = env.step(action)
        
        if not env.render():
            break
            
        if done:
            print(f"Episode ended - Distance: {env.car.distance_traveled:.0f}")
            state = env.reset(random_track=False)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = env.reset(random_track=False)
                    
    pygame.quit()