#include <stdlib.h>
#include <iostream>
#include <SDL.h>
#include "SimulationWindow.h"

#define SCREEN_WIDTH 2000
#define SCREEN_HEIGHT 1000

int main(int argc, char* args[])
{
    SimulationWindow* window = new SimulationWindow("Hydro", SCREEN_WIDTH, SCREEN_HEIGHT);   
    window->init();
    while(window->running())
    {
        window->handleEvents();
        window->update();
        window->render();
    }

    window->clean();
    return 0;
}