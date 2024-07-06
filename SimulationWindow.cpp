#include <SDL.h>
#include <algorithm>
#include "SimulationWindow.h"

#define ZOOM 5

SimulationWindow::SimulationWindow(const char* title, int width, int height, bool stepMode) {
    m_width = width;
    m_height = height;
    m_bRunning = false;
    m_sim = new Simulation(width, height, 0.00001);
    m_stepMode = stepMode;
    m_numSteps = 0;
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
                                     m_width * ZOOM, m_height * ZOOM,
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
    m_sim->render_texture(m_pixels);
    SDL_UnlockTexture(texture);

    // clear the window to black
    SDL_RenderClear(m_pRenderer);

    SDL_Rect destRect;
    destRect.x = 0;
    destRect.y = 0;
    destRect.w = ZOOM * m_width;
    destRect.h = ZOOM * m_height;

    SDL_RenderCopy(m_pRenderer, texture, NULL, &destRect);

    // show the window
    SDL_RenderPresent(m_pRenderer);

    SDL_DestroyTexture(texture);
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

            // if click, then add a step
            case SDL_MOUSEBUTTONDOWN:
                addSteps(1);
            break;

            // if spacebar, then toggle step mode
            case SDL_KEYDOWN:
                if (event.key.keysym.sym == SDLK_SPACE)
                {
                    m_stepMode = !m_stepMode;
                }
            break;

            default:
            break;
        }
    }
    if (m_stepMode && m_numSteps > 0)
    {
        m_sim->step();
        m_numSteps--;
    }
    else if (!m_stepMode) {
        m_sim->step();
    }
}


void SimulationWindow::handleEvents() {}

void SimulationWindow::clean() {
    SDL_DestroyWindow(m_pWindow);
    SDL_DestroyRenderer(m_pRenderer);
    SDL_Quit();
    delete m_sim;
}

void SimulationWindow::stop() {
    m_bRunning = false;
}