#include "Simulation.cuh"
#include <iostream>
#include <iomanip>

__global__ void d_advect_smoke(
    float *smoke,
    float *smokeNext,
    float *uField,
    float *vField,
    uint8_t *obstacles,
    unsigned int w,
    unsigned int h,
    float deltaT,
    float metersPerCell)
{
    int stride = gridDim.x * blockDim.x;
    int max_index = w * h;

    for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < max_index; i+=stride) {
        int x = i % w;
        int y = i / w;


        // Advect smoke
        // Find the velocities at x, y
        float u = (uField[(y * (w + 1)) + (x    )] + uField[((y    ) * (w + 1)) + (x + 1)]) / 2.0;
        float v = (vField[(y * (w    )) + (x    )] + vField[((y + 1) * (w    )) + (x    )]) / 2.0;

        // Add smoke to left side of the screen
        if ((x < 1) && (y % 50 < 50)) {
            smoke[i] = 1.0;
        }


        //if ((u == 0.0) && (v == 0.0)) {
        //    smokeNext[i] = smoke[i];
        //    continue;
        //}

        // Calculate the coordinates of the sample location
        float newX = max(min(x - ((u / metersPerCell) * deltaT), (float) w), 0.0);
        float newY = max(min(y - ((v / metersPerCell) * deltaT), (float) h), 0.0);
        //printf("X: %i, Y: %i, NewX: %f, NewY: %f, u: %f, v: %f\n", x, y, newX, newY, uField[i], vField[i]);

        //float uIdxLow = std::floor(newX - 0.5);
        //float uIdxHigh = std::ceil(newX - 0.5);

        //float vIdxLow = std::floor();
        float xFrac = x - (long)x;
        float yFrac = y - (long)y;
        
        // Sample the smoke at location - delta_t * velocity
        float w11 = (1 - xFrac) * (1 - yFrac);
        float w12 = (1 - xFrac) * (    yFrac);
        float w21 = (    xFrac) * (1 - yFrac);
        float w22 = (    xFrac) * (    yFrac);


        smokeNext[i] = (
            w11 * smoke[(unsigned int)(((newY    ) * w) + (newX    ))] +
            w12 * smoke[(unsigned int)(((newY + 1) * w) + (newX    ))] +
            w21 * smoke[(unsigned int)(((newY    ) * w) + (newX + 1))] +
            w22 * smoke[(unsigned int)(((newY + 1) * w) + (newX + 1))] 
        );
        //printf("Smoke: %f\n", smokeNext[i]);
    }
}

__global__ void d_render_texture(uint8_t *pixels, float *smoke, float *uField, float *vField, unsigned int width, unsigned int height) {
    int stride = gridDim.x * blockDim.x;
    int max_index = width * height;

    for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < max_index; i+=stride) {
        // Clamp the smoke value to 0-255
        //int pixel_value = static_cast<int>(uField[i * width/(width + 1)] * 25500 * 2);
        int pixel_value = static_cast<int>(smoke[i] * 255);
        if (pixel_value > 255) {
            pixel_value = 255;
        } else if (pixel_value < 0) {
            pixel_value = 0;
        }
        pixels[(4*i) + 1] = pixel_value;
    }
}

Simulation::Simulation(unsigned int width, unsigned int height, float dt) {
    m_width = width;
    m_height = height;
    m_u =         new GPUField<float>(  (width + 1) *  height     , -0.001);
    m_v =         new GPUField<float>(   width      * (height + 1), 0.001);
    m_uNext =     new GPUField<float>(  (width + 1) *  height     );
    m_vNext =     new GPUField<float>(   width      * (height + 1));
    m_smoke =     new GPUField<float>(   width      *  height     );
    m_smokeNext = new GPUField<float>(   width      *  height     );
    m_obstacles = new GPUField<uint8_t>( width      *  height     );
    m_pixels =    new GPUField<uint8_t>(4 * width   * height);
}

Simulation::~Simulation() {
    delete m_u;
    delete m_v;
    delete m_uNext;
    delete m_vNext;
    delete m_smoke;
    delete m_obstacles;
    delete m_pixels;
}

void Simulation::to_device(){
    m_u->to_device();
    m_v->to_device();
    m_uNext->to_device();
    m_vNext->to_device();
    m_smoke->to_device();
    m_smokeNext->to_device();
    m_obstacles->to_device();

    // d_pixels only goes from device to host
    //d_pixels.to_device();
}

void Simulation::from_device(){
    m_u->from_device();
    m_v->from_device();
    m_uNext->from_device();
    m_vNext->from_device();
    m_smoke->from_device();
    m_smokeNext->from_device();
    m_obstacles->from_device();
}

void Simulation::step() {
    //project();
    //advect_velocity();

    this->to_device();   
    printf("u: %f\n", m_u->m_hostData[1]);
    
    d_advect_smoke<<<1000, 256>>>(
        m_smoke->m_deviceData,
        m_smokeNext->m_deviceData,
        m_u->m_deviceData,
        m_v->m_deviceData,
        m_obstacles->m_deviceData,
        m_width,
        m_height,
        0.0001,
        0.0001);
    
    this->from_device();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    //printf("Smoke hex: %x\n", *reinterpret_cast<unsigned int*>(&m_smoke->m_hostData[m_smoke->m_size - 1]));
    printf("Smoke: %f\n", m_smoke->m_hostData[1]);
    printf("SmokeNext: %f\n", m_smokeNext->m_hostData[1]);
    //printf("Smoke cell size: %i\n", (int) sizeof(float));
    //printf("Smoke size: %i\n", m_smoke->get_byte_size());
    auto temp = m_smoke;
    m_smoke = m_smokeNext;
    m_smokeNext = temp;
}

void Simulation::render_texture(uint8_t *pixels) {
    d_render_texture<<<1000, 256>>>(m_pixels->m_deviceData, m_smoke->m_deviceData, m_u->m_deviceData, m_v->m_deviceData, m_width, m_height);
    m_pixels->from_device(pixels);
}
