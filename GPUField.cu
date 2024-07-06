#include "GPUField.cuh"
#include <cstdint>

template<typename T>
GPUField<T>::GPUField(unsigned int size, T initialValue) {
    m_size = size;
    m_hostData = new T[size];
    for (int i=0; i<size; i++) {
        m_hostData[i] = initialValue;
    }
    cudaMalloc(&m_deviceData, size * sizeof(T));
}

template<typename T>
GPUField<T>::~GPUField() {
    delete[] m_hostData;
    cudaFree(m_deviceData);
}

template<typename T>
void GPUField<T>::to_device() {
    cudaMemcpy(m_deviceData, m_hostData, m_size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPUField<T>::from_device() {
    cudaMemcpy(m_hostData, m_deviceData, m_size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void GPUField<T>::from_device(T *hostData) {
    cudaMemcpy(hostData, m_deviceData, m_size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
unsigned int GPUField<T>::get_byte_size() {
    return m_size * sizeof(T);
}

template<typename T>
void GPUField<T>::set_host_data(T val) {
    for (int i=0; i<m_size; i++) {
        m_hostData[i] = val;
    }
}

template class GPUField<float>;
template class GPUField<uint8_t>;
template class GPUField<bool>;