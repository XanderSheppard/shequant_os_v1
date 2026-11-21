#pragma once
#include <vector>
#include <cmath>
#include <random>

namespace Sheyo {
    namespace Physics {

        class SurfaceBody {
        private:
            double mass;
            double drag_coeff;
            double gravity; // NOW A VARIABLE

            std::mt19937 rng;
            std::normal_distribution<double> wind_dist;

        public:
            SurfaceBody() : mass(1.0), drag_coeff(0.1), gravity(-9.81), wind_dist(0.0, 0.5) {
                std::random_device rd;
                rng.seed(rd());
            }

            // NEW: Function to change Gravity on the fly
            void set_gravity(double g) {
                this->gravity = -std::abs(g); // Always negative (downward)
            }

            void randomize_properties(double variability) {
                std::uniform_real_distribution<double> m_dist(1.0 - variability, 1.0 + variability);
                std::uniform_real_distribution<double> d_dist(0.1, 0.1 + (variability * 0.5));

                this->mass = m_dist(rng);
                this->drag_coeff = d_dist(rng);
            }

            std::vector<double> step(const std::vector<double>& velocity,
                const std::vector<double>& control,
                double dt,
                double target_z,
                double current_z) {

                double v_x = velocity[0];
                double v_y = velocity[1];
                double v_z = velocity[2];

                // 1. WIND (Stochastic)
                double wind_gust = wind_dist(rng);

                // 2. DRAG (Aerodynamics)
                double drag_force = -drag_coeff * v_y * std::abs(v_y);

                // 3. THRUST (20N Max)
                double thrust = control[1] * 40.0;

                // 4. GRAVITY (Variable)
                double weight = mass * gravity;

                // 5. ACCELERATION (F = ma)
                double accel_y = (thrust + weight + drag_force + wind_gust) / mass;

                // 6. INTEGRATION
                double next_v_y = v_y + (accel_y * dt);

                return { v_x, next_v_y, v_z };
            }
        };
    }
}