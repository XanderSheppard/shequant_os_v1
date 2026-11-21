import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ctypes
import os
import sys
import importlib.util
import random

def load_engine():
    loader = importlib.util.find_spec("atmosphere_kernel")
    if loader and loader.origin:
        return ctypes.CDLL(loader.origin)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "build", "atmosphere_kernel.so"),
        os.path.join(script_dir, "build", "Release", "atmosphere_kernel.pyd"),
        os.path.join(script_dir, "atmosphere_kernel.so"),
        os.path.join(script_dir, "atmosphere_kernel.pyd"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return ctypes.CDLL(p)
    print("!! ERROR: Physics engine not found. Run 'cmake -S . -B build && cmake --build build' first.")
    sys.exit(1)

lib = load_engine()

# Signatures
lib.Sheyo_Init.argtypes = [ctypes.c_int]
lib.Sheyo_SetGravity.argtypes = [ctypes.c_double]
lib.Sheyo_UpdatePhysics.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.c_double
]
lib.Sheyo_GetReward.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.Sheyo_GetReward.restype = ctypes.c_double
lib.Sheyo_Randomize.argtypes = [ctypes.c_double]

class SheyoSurfaceEnv(gym.Env):
    def __init__(self, gravity=-9.81):
        super(SheyoSurfaceEnv, self).__init__()
        
        # Action: Thrust (X, Y, Z)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: [Vel_X, Vel_Y, Vel_Z, Distance_To_Target]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.max_steps = 400
        self.current_step = 0
        
        # FIXED GOAL: 50m Altitude
        self.target_height = 50.0 
        self.gravity_val = gravity 
        
        self.state = np.zeros(3, dtype=np.float64) 
        self.velocity = np.zeros(3, dtype=np.float64)
        
        lib.Sheyo_Init(0)
        lib.Sheyo_SetGravity(ctypes.c_double(self.gravity_val))

    # Helper to change difficulty
    def set_difficulty(self, g_force):
        self.gravity_val = g_force

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Enforce current gravity
        lib.Sheyo_SetGravity(ctypes.c_double(self.gravity_val))

        # Start low (5m) so it has to CLIMB to 50m
        self.state = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        
        # Randomize Drone Mass slightly
        lib.Sheyo_Randomize(0.1) 

        self.velocity = np.zeros(3, dtype=np.float64)
        self.current_step = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Distance to 50m Target
        dist_y = self.target_height - self.state[1]
        
        obs = np.array([
            self.velocity[0], 
            self.velocity[1], 
            self.velocity[2], 
            dist_y 
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        dt = 0.05
        
        v_ptr = self.velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        act = np.clip(action, -1.0, 1.0).astype(np.float64)
        c_ptr = act.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        lib.Sheyo_UpdatePhysics(v_ptr, c_ptr, ctypes.c_double(dt))
        self.state += self.velocity * dt
        
        reward = lib.Sheyo_GetReward(
            ctypes.c_double(self.state[1]), 
            ctypes.c_double(self.target_height), 
            ctypes.c_double(self.velocity[1])
        )

        terminated = False
        truncated = False

        # Crash Logic
        if self.state[1] <= 0.0:
            self.state[1] = 0.0
            reward = -10.0
            terminated = True
        # Raised Ceiling to 100m (since target is 50m)
        elif self.state[1] >= 100.0:
            self.state[1] = 100.0
            reward = -5.0
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode=None):
        # Fallback to the class render_mode if no argument is passed
        current_mode = mode if mode is not None else getattr(self, 'render_mode', 'human')
        
        # --- Your existing Pygame drawing code is here ---
        # (Keep all your existing code that draws the drone, target, and background)
        # pygame.display.flip()
        
        # --- Add this block at the very end of the function ---
        if current_mode == "rgb_array":
            import pygame
            import numpy as np
            # Extract the raw pixel buffer from the Pygame surface
            raw_pixels = pygame.surfarray.array3d(pygame.display.get_surface())
            # Pygame outputs (Width, Height, Channels). We transpose to (Height, Width, Channels)
            return np.transpose(raw_pixels, (1, 0, 2))