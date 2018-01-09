#ifndef APO_SHARED_H
#define APO_SHARED_H

#include <random>

void InitRandom();

// thread safe random number generator
std::mt19937 & randGen();

#endif // APO_SHARED_H
