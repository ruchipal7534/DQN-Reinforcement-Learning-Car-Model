import pygame
import math
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from datetime import datetime
from enum import Enum

pygame.init()

WIDTH, HEIGHT = 800, 600
CAR_SIZE = (20, 40)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_BLUE = (173, 216, 230)
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Track Car Racing")
clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrackType(Enum):
    OVAL = "oval"
    RECTANGLE = "rectangle"
    L_TRACK = "l_track"
    U_TRACK = "u_track"
    SIMPLE_CURVE = "simple_curve"
    DOUBLE_LOOP = "double_loop"
    TEST_TRACK = "test_track"