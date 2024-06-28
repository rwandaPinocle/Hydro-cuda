#include "Simulation.cuh"
#include <iostream>
#include <iomanip>

__global__ void d_advect_vel(
    float *uField,
    float *vField,
    float *uNext,
    float *vNext,
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

        // Find the velocities at x, y
        float u = (uField[(y * (w + 1)) + (x    )] + uField[((y    ) * (w + 1)) + (x + 1)]) / 2.0;
        float v = (vField[(y * (w    )) + (x    )] + vField[((y + 1) * (w    )) + (x    )]) / 2.0;

        // Add velocity to left side of the screen
        if ((x > 0) && (x < 2) && (y < h) && (y > 0)) {
            uField[i] = 0.02;
        }

        // Calculate the coordinates of the sample location
        float newX = max(min(x - (u * deltaT / metersPerCell) - 0.5, (float) w), 0.0f);
        float newY = max(min(y - (v * deltaT / metersPerCell), (float) h), 0.0f);

        float xFrac = newX - (long)newX;
        float yFrac = newY - (long)newY;
        
        // Sample uField
        float w11 = (1 - xFrac) * (1 - yFrac);
        float w12 = (1 - xFrac) * (    yFrac);
        float w21 = (    xFrac) * (1 - yFrac);
        float w22 = (    xFrac) * (    yFrac);

        uNext[i] = (
            w11 * uField[(unsigned int)(((newY    ) * w) + (newX    ))] +
            w12 * uField[(unsigned int)(((newY + 1) * w) + (newX    ))] +
            w21 * uField[(unsigned int)(((newY    ) * w) + (newX + 1))] +
            w22 * uField[(unsigned int)(((newY + 1) * w) + (newX + 1))] 
        );

        // Sample vField
        newX = max(min(x - (u * deltaT / metersPerCell), (float) w), 0.0f);
        newY = max(min(y - (v * deltaT / metersPerCell) - 0.5, (float) h), 0.0f);

        xFrac = newX - (long)newX;
        yFrac = newY - (long)newY;

        w11 = (1 - xFrac) * (1 - yFrac);
        w12 = (1 - xFrac) * (    yFrac);
        w21 = (    xFrac) * (1 - yFrac);
        w22 = (    xFrac) * (    yFrac);

        vNext[i] = (
            w11 * vField[(unsigned int)(((newY    ) * w) + (newX    ))] +
            w12 * vField[(unsigned int)(((newY + 1) * w) + (newX    ))] +
            w21 * vField[(unsigned int)(((newY    ) * w) + (newX + 1))] +
            w22 * vField[(unsigned int)(((newY + 1) * w) + (newX + 1))] 
        );
    }
}

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
        if ((x > 0) && (x < 2) && (y < h) && (y > 0)) {
            smoke[i] = 1.0;
        }

        // Calculate the coordinates of the sample location
        float newX = max(min(x - (u * deltaT / metersPerCell), (float) w), 0.0f);
        float newY = max(min(y - (v * deltaT / metersPerCell), (float) h), 0.0f);

        float xFrac = newX - (long)newX;
        float yFrac = newY - (long)newY;
        
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
    m_u =         new GPUField<float>(  (width + 1) *  height     , 0.001);
    m_v =         new GPUField<float>(   width      * (height + 1), 0.00);
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

    //m_u->m_hostData[0] = 0.11;
    //m_u->m_hostData[1] = 0.22;
    this->to_device();   
    //printf("u: %f\n", m_u->m_hostData[1]);

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

    d_advect_vel<<<1000, 256>>>(
        m_u->m_deviceData,
        m_v->m_deviceData,
        m_uNext->m_deviceData,
        m_vNext->m_deviceData,
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

    GPUField<float> *temp;

    temp = m_u;
    m_u = m_uNext;
    m_uNext = temp;

    temp = m_v;
    m_v = m_vNext;
    m_vNext = temp;
    
    temp = m_smoke;
    m_smoke = m_smokeNext;
    m_smokeNext = temp;
}

void Simulation::render_texture(uint8_t *pixels) {
    d_render_texture<<<1000, 256>>>(m_pixels->m_deviceData, m_smoke->m_deviceData, m_u->m_deviceData, m_v->m_deviceData, m_width, m_height);
    m_pixels->from_device(pixels);
}
