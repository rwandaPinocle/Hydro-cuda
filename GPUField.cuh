#ifndef GPUFIELD_H
#define GPUFIELD_H

template<typename T>
class GPUField {
    public:
        GPUField(unsigned int size, T initialValue = 0);
        ~GPUField();

        void to_device();
        void from_device();
        void from_device(T *hostData);

        unsigned int get_byte_size();

        void set_host_data(T val);

        T *m_hostData;
        T *m_deviceData;
        unsigned int m_size;
};

#endif