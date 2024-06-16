build:
	nvcc -I/usr/include/SDL2 main.cpp SimulationWindow.cpp -lSDL2 -o main