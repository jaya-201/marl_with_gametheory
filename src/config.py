import os

# --- Agents ---
AGENTS = ['AA', 'DL', 'UA', 'B6']  # Replace with actual airlines

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'results', 'logs')

# --- Training Hyperparameters ---
TOTAL_TIMESTEPS = 5000
N_STEPS = 2048
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
ENTROPY_COEF = 0.01
CHECKPOINT_FREQ = 50_000

# --- Environment Parameters ---
MAX_DAYS = 30
PRICE_POINTS = list(range(150, 801, 35))  # discrete price options