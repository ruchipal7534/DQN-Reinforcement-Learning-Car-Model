import pygame
from constants import *
from manual_play import manual_play_mode
from training import train_multi_track, test_on_new_track, visualize_all_tracks

def draw_main_menu():
    screen.fill(BLACK)
    
    title_font = pygame.font.SysFont(None, 64)
    title_text = title_font.render("Car Racing", True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH//2, 100))
    screen.blit(title_text, title_rect)
    
    subtitle_font = pygame.font.SysFont(None, 32)
    subtitle_text = subtitle_font.render("Deep Q-Learning & Manual Play", True, LIGHT_BLUE)
    subtitle_rect = subtitle_text.get_rect(center=(WIDTH//2, 140))
    screen.blit(subtitle_text, subtitle_rect)
    
    font = pygame.font.SysFont(None, 36)
    options = [
        "1. Manual Play Mode",
        "2. Watch AI Training",
        "3. Test Trained AI",
        "4. View All Tracks",
        "ESC. Exit"
    ]
    
    y_start = 220
    for i, option in enumerate(options):
        color = WHITE if not option.startswith("ESC") else RED
        text = font.render(option, True, color)
        text_rect = text.get_rect(center=(WIDTH//2, y_start + i * 50))
        screen.blit(text, text_rect)
    
    instruction_font = pygame.font.SysFont(None, 24)
    
    pygame.display.flip()

def main_menu():
    while True:
        draw_main_menu()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                elif event.key == pygame.K_1:
                    manual_play_mode()
                    return
                elif event.key == pygame.K_2:
                    print("Starting AI Training...")
                    train_multi_track(num_episodes=500)
                elif event.key == pygame.K_3:
                    print("Testing trained AI...")
                    test_on_new_track('models/best_model.pt', num_tests=5)
                elif event.key == pygame.K_4:
                    visualize_all_tracks()
        
        clock.tick(30)

def main():
    print("Enhanced Multi-track Car Racing RL Training")
    print(f"Device: {device}")
    
    main_menu()

if __name__ == "__main__":
    print("Multi-Track Car Racing with Manual Play Mode")
    print("=" * 50)
    
    main_menu()