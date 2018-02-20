#include "apo/shared.h"
#ifdef _OPENMP
#include "omp.h"
#endif

std::vector<std::mt19937> threadGens;
std::vector<std::uniform_real_distribution<float>> unitDists;

static bool initialized = false;

void
InitRandom() {
  if (initialized) return;
  initialized = true;

#ifdef _OPENMP
  std::mt19937 randGen(42);

  size_t maxThreads = omp_get_max_threads();
  std::uniform_int_distribution<size_t> seeder;

  for (size_t i = 0; i < maxThreads; ++i) {
    threadGens.emplace_back(seeder(randGen));
    unitDists.emplace_back(0, 1);
  }
#else
  size_t seed = 42;
  threadGens.emplace_back(seed);
  unitDists.emplace_back(0, 1);
#endif
}

inline int getThreadId() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

std::mt19937 & randGen()  { return threadGens[getThreadId()]; }
float drawUnitRand() { return unitDists[getThreadId()](randGen()); }
