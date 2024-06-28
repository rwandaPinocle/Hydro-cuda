#include <SDL.h>
#include "Simulation.cuh"

#ifndef SIMULATIONWINDOW_H
#define SIMULATIONWINDOW_H

class SimulationWindow
{
    public:
        SimulationWindow(const char *title, int width, int height, bool stepMode);
        ~SimulationWindow();

        int init();
        void render();
        void update();
        void handleEvents();
        void clean();
        void stop();
        void addSteps(int steps) { m_numSteps += steps; };
        void setStepMode(bool stepMode) { m_stepMode = stepMode; }

        bool running() { return m_bRunning; }
    
    private:
        SDL_Window *m_pWindow;
        SDL_Renderer *m_pRenderer;
        bool m_bRunning;
        int m_width;
        int m_height;
        uint8_t *m_pixels;
        int m_pitch;
        bool m_stepMode;
        int m_numSteps;
        Simulation *m_sim;
};
#endif