#include <SDL.h>
#include "SimulationWindow.h"

SimulationWindow::SimulationWindow(const char* title, int width, int height) {
    m_width = width;
    m_height = height;
    m_bRunning = false;
    SDL_Renderer* m_pRenderer;
}

SimulationWindow::~SimulationWindow()
{
}

int SimulationWindow::init() {
    // initialize SDL
    if (SDL_Init(SDL_INIT_EVERYTHING) >= 0)
    {
        // if succeeded create our window
        m_pWindow = SDL_CreateWindow("Hydro",
                                     SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                     m_width, m_height,
                                     SDL_WINDOW_SHOWN);
        // if the window creation succeeded create our renderer
        if (m_pWindow != 0)
        {
            m_pRenderer = SDL_CreateRenderer(m_pWindow, -1, 0);
        }
    }
    else
    {
        return 1; // sdl could not initialize
    }
    m_bRunning = true; // everything inited successfully, start the main loop
    return 0;
}

void SimulationWindow::render() {
    SDL_SetRenderDrawColor(m_pRenderer, 0, 0, 0, 255);


    SDL_Texture *texture;
    m_pitch = m_width * m_height * 4;
    texture = SDL_CreateTexture(m_pRenderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, m_width, m_height);
    SDL_LockTexture(texture, NULL, (void**)&m_pixels, &m_pitch);
    for (int i = 0; i < m_width * 100; i++)
    {
        m_pixels[i * 4] = 255;
        m_pixels[i * 4 + 1] = 0;
        m_pixels[i * 4 + 2] = 0;
        m_pixels[i * 4 + 3] = 0;
    }
    SDL_UnlockTexture(texture);

    // clear the window to black
    SDL_RenderClear(m_pRenderer);

    SDL_RenderCopy(m_pRenderer, texture, NULL, NULL);

    // show the window
    SDL_RenderPresent(m_pRenderer);
}

void SimulationWindow::update() {
    SDL_Event event;
    if(SDL_PollEvent(&event))
    {
        switch(event.type)
        {
            case SDL_QUIT:
                m_bRunning = false;
            break;

            default:
            break;
        }
    }
}

void SimulationWindow::handleEvents() {}

void SimulationWindow::clean() {
    SDL_DestroyWindow(m_pWindow);
    SDL_DestroyRenderer(m_pRenderer);
    SDL_Quit();
}