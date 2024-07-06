build:
	nvcc -std=c++17 -I/usr/include/SDL2 main.cpp SimulationWindow.cpp Simulation.cu GPUField.cu -lSDL2 -o hydro

build-debug:
	nvcc -g -std=c++17 -I/usr/include/SDL2 main.cpp SimulationWindow.cpp Simulation.cu GPUField.cu -lSDL2 -o hydro