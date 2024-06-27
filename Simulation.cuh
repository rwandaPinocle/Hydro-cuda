#ifndef SIMULATION_H
#define SIMULATION_H
#include <cstdint>
#include "GPUField.cuh"


class Simulation {
    public:
        Simulation(unsigned int width, unsigned int height, float dt);
        ~Simulation();

        void add_obstacles();
        void step();
        void render_texture(uint8_t *pixels);
        void to_device();
        void from_device();

    private:
        unsigned int m_width, m_height;
        GPUField<float> *m_u, *m_v, *m_uNext, *m_vNext;
        GPUField<float> *m_smoke, *m_smokeNext;
        GPUField<uint8_t> *m_obstacles;
        GPUField<uint8_t> *m_pixels;
};

#endif