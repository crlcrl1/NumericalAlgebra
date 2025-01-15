#include "cli.h"

int main(const int argc, char **argv) {
    const Config config(argc, argv);
    config.apply();
    return 0;
}
