#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "particle.h"

int main(void) {
    struct Particle * X = malloc(sizeof(particle));
    X->x = 1;
    printf("%f\n", X->x);
    return EXIT_SUCCESS;
}