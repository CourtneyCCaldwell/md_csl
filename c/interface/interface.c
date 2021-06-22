// Copyright 2021 Courtney Caldwell
// and Patrick T. F. Kelly
#include <stdio.h>
#include <stdlib.h>

#include "shell/shell.h"

int main(void) {
    // shell loop
    mdsh_loop();
    // exit process
    return EXIT_SUCCESS;
}