#include "atmosphere_dynamics.h"
#include <vector>
#include <cmath>
#include <algorithm>

#if defined(_WIN32)
    #define SHEYO_EXPORT __declspec(dllexport)
#else
    #define SHEYO_EXPORT __attribute__((visibility("default")))
#endif

Sheyo::Physics::SurfaceBody* global_drone = nullptr;

extern "C" {
    SHEYO_EXPORT void Sheyo_Init(int mode) {
        if (global_drone) delete global_drone;
        global_drone = new Sheyo::Physics::SurfaceBody();
    }

    SHEYO_EXPORT void Sheyo_SetGravity(double g) {
        if (global_drone) {
            global_drone->set_gravity(g);
        }
    }

    SHEYO_EXPORT void Sheyo_UpdatePhysics(double* velocity, double* control, double dt) {
        if (!global_drone) return;
        std::vector<double> vel_vec = { velocity[0], velocity[1], velocity[2] };
        std::vector<double> ctrl_vec = { control[0], control[1], control[2] };
        std::vector<double> next_vel = global_drone->step(vel_vec, ctrl_vec, dt, 0.0, 0.0);
        velocity[0] = next_vel[0];
        velocity[1] = next_vel[1];
        velocity[2] = next_vel[2];
    }

    SHEYO_EXPORT double Sheyo_GetReward(double current_z, double target_z, double vel_z) {
        double z_error = std::abs(target_z - current_z);
        double vel_mag = std::abs(vel_z);

        double r_dist = std::exp(-z_error);
        double r_loiter = 0.0;
        if (z_error < 1.0) {
            r_loiter = 0.5 * std::max(0.0, (1.0 - vel_mag));
        }
        double r_step_penalty = -0.01;
        double r_penalty = 0.0;
        if (z_error < 2.0 && vel_mag > 2.0) {
            r_penalty = -0.1 * vel_mag;
        }

        return r_dist + r_loiter + r_penalty + r_step_penalty;
    }

    SHEYO_EXPORT void Sheyo_Randomize(double variability) {
        if (global_drone) global_drone->randomize_properties(variability);
    }
}
