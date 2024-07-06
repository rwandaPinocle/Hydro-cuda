#include "Simulation.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>

void swapFields(GPUField<float> **a, GPUField<float> **b) {
    GPUField<float> *temp = *a;
    *a = *b;
    *b = temp;
}

__device__ float sampleField(
    float *f,
    float x,
    float y,
    unsigned int w,
    unsigned int h,
    float xOffset,
    float yOffset
) {
    // Calculate the coordinates of the sample location
    float newX = max(min(x, (float) w-1), 0.0f);
    float newY = max(min(y, (float) h-1), 0.0f);
    newX = x;
    newY = y;

    float xFrac = newX - (long)newX;
    float yFrac = newY - (long)newY;
    
    // Sample the field at the new location with bilinear interpolation
    float w11 = (1 - xFrac) * (1 - yFrac);
    float w12 = (1 - xFrac) * (    yFrac);
    float w21 = (    xFrac) * (1 - yFrac);
    float w22 = (    xFrac) * (    yFrac);

    unsigned int x1 = newX;
    unsigned int y1 = newY;
    unsigned int x2 = min((unsigned int) (newX + 1), w-1);
    unsigned int y2 = min((unsigned int) (newY + 1), h-1);

    return (
        w11 * f[x1 + y1*w] +
        w12 * f[x1 + y2*w] +
        w21 * f[x2 + y1*w] +
        w22 * f[x2 + y2*w] 
    );
}

__global__ void d_advect_vel(
    float *uField,
    float *vField,
    float *uNext,
    float *vNext,
    float *obstacles,
    unsigned int w,
    unsigned int h,
    float deltaT,
    float metersPerCell)
{
    int stride = gridDim.x * blockDim.x;
    int max_index = (w+1) * (h+1);

    for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < max_index; i+=stride) {
        int x = (i % (w + 1));
        int y = (i / (w + 1));

        float u, v, newX, newY;

        if (x != w) {
            // Advect u 
            u = uField[(x    ) + ((y    ) * (w + 1))];
            v = (
                vField[(x    ) + ((y    ) * (w    ))] +
                vField[(x    ) + ((y + 1) * (w    ))] + 
                vField[(x + 1) + ((y    ) * (w    ))] +
                vField[(x + 1) + ((y + 1) * (w    ))]
            ) / 4.0f;


            // Calculate the coordinates of the sample location
            newX = max(min(x - (u * deltaT / metersPerCell), (float) w), 0.0f);
            newY = max(min(y - (v * deltaT / metersPerCell), (float) h), 0.0f);

            uNext[x + (y * (w + 1))] = sampleField(uField, newX, newY, w + 1, h + 1, 0.0, 0.0);
        }

        if (y != h) {
            // Advect v 
            u = (
                uField[(x    ) + ((y    ) * (w + 1))] +
                uField[(x + 1) + ((y    ) * (w + 1))] +
                uField[(x    ) + ((y + 1) * (w + 1))] +
                uField[(x + 1) + ((y + 1) * (w + 1))] 
            ) / 4.0f;
            v = vField[(x    ) + ((y    ) * (w    ))];

            // Calculate the coordinates of the sample location
            newX = max(min(x - (u * deltaT / metersPerCell), (float) w), 0.0f);
            newY = max(min(y - (v * deltaT / metersPerCell), (float) h), 0.0f);

            vNext[x + (y * w)] = sampleField(vField, newX, newY, w, h + 1, 0.0, 0.0);
        }
    }
}

__global__ void d_advect_smoke(
    float *smoke,
    float *smokeNext,
    float *uField,
    float *vField,
    float *obstacles,
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

        // Add smoke to the screen
        if (x == 1 && (y % 10 < 2)) {
            smoke[i] = 1.0;
            smoke[i+1] = 1.0;
            smoke[i+2] = 1.0;
            smoke[i+3] = 1.0;
            smoke[i+4] = 1.0;
        }

        // Find the velocities at x, y
        float u = (uField[(x    ) + ((y    ) * (w + 1))] + uField[(x + 1) + ((y    ) * (w + 1))]) / 2.0;
        float v = (vField[(x    ) + ((y    ) * (w    ))] + vField[(x    ) + ((y + 1) * (w    ))]) / 2.0;

        // Calculate the coordinates of the sample location
        float newX = max(min(x - (u * deltaT / metersPerCell), (float) w), 0.0f);
        float newY = max(min(y - (v * deltaT / metersPerCell), (float) h), 0.0f);

        smokeNext[i] = sampleField(smoke, newX, newY, w, h, 0.0f, 0.0f);
    }
}

__host__ __device__ void project_cell(
    float *u,
    float *v,
    float *obs,
    unsigned int w,
    unsigned int h,
    unsigned int x,
    unsigned int y,
    float overrelaxation
) {
    unsigned int obsIdxX0 = (x - 1) + ((y    ) * w);
    unsigned int obsIdxX1 = (x + 1) + ((y    ) * w);
    unsigned int obsIdxY0 = (x    ) + ((y - 1) * w);
    unsigned int obsIdxY1 = (x    ) + ((y + 1) * w);

    unsigned int uIdx0 =  x      + ( y      * (w + 1));
    unsigned int uIdx1 = (x + 1) + ( y      * (w + 1));

    unsigned int vIdx0 =  x      + ( y      *  w     );
    unsigned int vIdx1 =  x      + ((y + 1) *  w     );

    float obsCount = 4.0f - (
        obs[obsIdxX0] +
        obs[obsIdxX1] +
        obs[obsIdxY0] +
        obs[obsIdxY1]
    );
    if (obsCount == 4.0) {
        return;
    }

    float divergence = (
        - u[uIdx0] + u[uIdx1]
        - v[vIdx0] + v[vIdx1]
    ) / (4 - obsCount);

    float p = -divergence/(4 - obsCount);
    p *= overrelaxation;

    u[uIdx0] -= (obs[obsIdxX0] * p);
    u[uIdx1] += (obs[obsIdxX1] * p);
    v[vIdx0] -= (obs[obsIdxY0] * p);
    v[vIdx1] += (obs[obsIdxY1] * p);
}

__global__ void d_project(
    float *u,
    float *v,
    float *obs,
    unsigned int w,
    unsigned int h
) {
    int stride = gridDim.x * blockDim.x;
    int max_index = (w-2) * (h-2);
    float overrelaxation = 1.9;

    for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < max_index; i+=stride) {
        int x = (i % (w-2)) + 1;
        int y = (i / (w-2)) + 1;

        project_cell(u, v, obs, w, h, x, y, overrelaxation);
    }
}

void h_project(float *u,
    float *v,
    float *obs,
    unsigned int w,
    unsigned int h
) {
    int max_index = (w-2) * (h-2);
    float overrelaxation = 1.9;

    for (int i=0; i<max_index; i++) {
        int x = (i % (w-2)) + 1;
        int y = (i / (w-2)) + 1;
    
        project_cell(u, v, obs, w, h, x, y, overrelaxation);
    }
}

__global__ void d_render_texture(
    uint8_t *pixels,
    float *smoke,
    float *uField,
    float *vField,
    float *obs,
    unsigned int width,
    unsigned int height
) {
    int stride = gridDim.x * blockDim.x;
    int max_index = width * height;

    for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < max_index; i+=stride) {
        // Clamp the smoke value to 0-255
        int green = static_cast<int>(smoke[i] * 255);
        if (green > 255) {
            green = 255;
        } else if (green < 0) {
            green = 0;
        }
        pixels[(4*i) + 1] = green;

        int red = static_cast<int>((1.0 - obs[i]) * 255);
        pixels[(4*i) + 2] = red;
    }
}

Simulation::Simulation(unsigned int width, unsigned int height, float dt) {
    m_width = width;
    m_height = height;
    m_u =         new GPUField<float>(  (width + 1) *  height     , 0.0);
    m_v =         new GPUField<float>(   width      * (height + 1), 0.0);
    m_uNext =     new GPUField<float>(  (width + 1) *  height     );
    m_vNext =     new GPUField<float>(   width      * (height + 1));
    m_smoke =     new GPUField<float>(   width      *  height     );
    m_smokeNext = new GPUField<float>(   width      *  height     );
    m_obstacles = new GPUField<float>(   width      *  height     , 1.0);
    m_pixels =    new GPUField<uint8_t>(4 * width   * height);

    // Add obstacles
    float radius = 7;
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            if (i == 0 || j == 0 || j == height - 1) {
                m_obstacles->m_hostData[i + j * width] = 0.0;
            }  
            if (pow((float)i-((float)width/5), 2) + pow((float)j-((float)height/2), 2) < pow(radius, 2)) {
                m_obstacles->m_hostData[i + j * width] = 0.0;
            }
        }
    }
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

    // d_pixels is excluded because it only goes from device to host
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
    // Add velocity to left side of the screen
    for (int y=0; y<m_height; y++) {
        m_u->m_hostData[(y * (m_width + 1)) + 1] = 15.0;
    }

    this->to_device();   

    const bool useGPU = true;
    int iterations = 100;
    for (int i=0; i<iterations; i++) {
        if (useGPU) {
            d_project<<<1, 256>>>(
                m_u->m_deviceData,
                m_v->m_deviceData,
                m_obstacles->m_deviceData,
                m_width,
                m_height
            );
        } else {
            h_project(
                m_u->m_hostData,
                m_v->m_hostData,
                m_obstacles->m_hostData,
                m_width,
                m_height
            );
        }
    }
    if (!useGPU) {
        this->to_device();
    }

    d_advect_vel<<<1000, 256>>>(
        m_u->m_deviceData,
        m_v->m_deviceData,
        m_uNext->m_deviceData,
        m_vNext->m_deviceData,
        m_obstacles->m_deviceData,
        m_width,
        m_height,
        0.0001,
        0.001);

    d_advect_smoke<<<1000, 256>>>(
        m_smoke->m_deviceData,
        m_smokeNext->m_deviceData,
        m_u->m_deviceData,
        m_v->m_deviceData,
        m_obstacles->m_deviceData,
        m_width,
        m_height,
        0.0001,
        0.001);

    this->from_device();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    swapFields(&m_u, &m_uNext);
    swapFields(&m_v, &m_vNext);
    swapFields(&m_smoke, &m_smokeNext);
}

void Simulation::render_texture(uint8_t *pixels) {
    d_render_texture<<<1000, 256>>>(
        m_pixels->m_deviceData,
        m_smoke->m_deviceData,
        m_u->m_deviceData,
        m_v->m_deviceData,
        m_obstacles->m_deviceData,
        m_width,
        m_height
    );
    m_pixels->from_device(pixels);
}
