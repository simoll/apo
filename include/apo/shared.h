#ifndef APO_SHARED_H
#define APO_SHARED_H

#include <random>

void InitRandom();

// thread safe random number generator
std::mt19937 & randGen();
float drawUnitRand(); // draw a random number from the interval [0,1] // thread safe

#endif // APO_SHARED_H
