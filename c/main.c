#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "particle.h"

// MACROS
// params
#define N 20000
#define rho 0.8
#define E -3.00
// time params
#define nblk  5
#define nstep  10
#define dt 0.001


int main(void) {
    struct Particle * X = malloc(sizeof(particle));
    X->x = 1;
    printf("%f\n", X->x);
    free(X);
    return EXIT_SUCCESS;
}