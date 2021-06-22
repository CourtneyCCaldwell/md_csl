#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "particle/particle.h"


int main(int argc, char ** argv) {
    struct Particle * X = malloc(sizeof(struct Particle));
    X->x = 1;
    printf("%f\n", X->x);
    free(X);
    return EXIT_SUCCESS;
}