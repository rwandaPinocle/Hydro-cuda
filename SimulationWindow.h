#include <SDL.h>
#ifndef SIMULATIONWINDOW_H
#define SIMULATIONWINDOW_H

class SimulationWindow
{
    public:
        SimulationWindow(const char* title, int width, int height);
        ~SimulationWindow();

        int init();
        void render();
        void update();
        void handleEvents();
        void clean();

        bool running() { return m_bRunning; }
    
    private:
        SDL_Window* m_pWindow;
        SDL_Renderer* m_pRenderer;
        bool m_bRunning;
        int m_width;
        int m_height;
        uint8_t* m_pixels;
        int m_pitch;
};
#endif