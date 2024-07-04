#include <stdlib.h>
#include <iostream>
#include <SDL.h>
#include "SimulationWindow.h"

#define SCREEN_WIDTH 600
#define SCREEN_HEIGHT 300

int main(int argc, char* args[])
{
    SimulationWindow* window = new SimulationWindow("Hydro", SCREEN_WIDTH, SCREEN_HEIGHT, true);   
    window->init();
    while(window->running())
    {
        window->handleEvents();
        window->update();
        window->render();
        //window->stop();
    }

    window->clean();
    return 0;
}