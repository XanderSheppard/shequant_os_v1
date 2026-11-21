import sys
import importlib.util
import ctypes
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# --- 1. CONNECT TO THE ENGINE ---
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

# Define Signatures
lib.Sheyo_Init.argtypes = [ctypes.c_int]
lib.Sheyo_UpdatePhysics.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.c_double
]

# --- 2. CONFIGURATION ---
SIM_STEPS = 1000
DT = 0.01  
TARGET_HEIGHT = 10.0
WIND_INTENSITY = 0.5 

# Pre-calculate Wind (The "Attacks")
wind_gusts = np.zeros(SIM_STEPS)
for i in range(0, SIM_STEPS, 50): # Hit it every 0.5 seconds
    wind_gusts[i] = np.random.normal(0, WIND_INTENSITY)

def run_visual_test():
    print(">> SHEYO ATMOSPHERE: GENERATING TELEMETRY LOG...")
    
    lib.Sheyo_Init(0)
    
    velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    position = np.array([0.0, 10.0, 0.0], dtype=np.float64)
    control  = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    v_ptr = velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_ptr = control.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dt_c = ctypes.c_double(DT)
    
    # BLACKBOX LOGGING
    history_alt = []
    history_wind = []
    history_thrust = []
    
    max_drift = 0.0
    start_time = time.time()
    
    for t in range(SIM_STEPS):
        # 1. Controller (The Brain)
        error_y = TARGET_HEIGHT - position[1]
        thrust_cmd = np.clip(error_y * 2.5, -1.0, 1.0)
        control[1] = thrust_cmd
        
        # 2. Physics (The World)
        velocity[0] += wind_gusts[t] * DT # Inject Wind
        lib.Sheyo_UpdatePhysics(v_ptr, c_ptr, dt_c)
        position += velocity * DT
        
        # 3. Log Data
        drift = abs(position[1] - TARGET_HEIGHT)
        if drift > max_drift: max_drift = drift
        
        history_alt.append(position[1])
        history_wind.append(wind_gusts[t])
        history_thrust.append(thrust_cmd)

    end_time = time.time()
    compute_time = (end_time - start_time) / SIM_STEPS * 1000 

    # --- 4. RENDER THE EVIDENCE ---
    print(f">> Simulation Complete. Rendering Graph...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Altitude (The Struggle)
    plt.subplot(3, 1, 1)
    plt.plot(history_alt, color='green', label='Drone Altitude')
    plt.axhline(y=TARGET_HEIGHT, color='red', linestyle='--', label='Target (10m)')
    plt.title(f"Sheyo Atmosphere: Stabilization Test (Max Drift: {max_drift:.4f}m)")
    plt.ylabel("Height (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Wind (The Enemy)
    plt.subplot(3, 1, 2)
    plt.plot(history_wind, color='blue', label='Wind Gusts (X-Axis)')
    plt.ylabel("Wind Velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Thrust (The Reaction)
    plt.subplot(3, 1, 3)
    plt.plot(history_thrust, color='orange', label='Motor Output')
    plt.xlabel("Time Steps (10ms)")
    plt.ylabel("Thrust Command")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save Proof
    plt.tight_layout()
    plt.savefig("Sheyo_Telemetry_Log.png")
    print(f">> LOG SAVED: Sheyo_Telemetry_Log.png")
    
    # Verdict
    print("-" * 30)
    print(f"MAX DRIFT:    {max_drift:.4f} m")
    print(f"COMPUTE TIME: {compute_time:.4f} ms")
    print("-" * 30)

if __name__ == "__main__":
    run_visual_test()