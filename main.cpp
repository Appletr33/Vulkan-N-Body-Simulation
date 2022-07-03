#include "maze_app.hpp"

int main(int argc, char *argv[])
{
    HelloTriangleApplication app{argv};
    
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
