#include "apo/shared.h"
#ifdef _OPENMP
#include "omp.h"
#endif

std::vector<std::mt19937> threadGens;

void
InitRandom() {
#ifdef _OPENMP
  std::mt19937 randGen(42);

  size_t maxThreads = omp_get_max_threads();
  std::uniform_int_distribution<size_t> seeder;

  for (size_t i = 0; i < maxThreads; ++i) {
    threadGens.push_back(std::mt19937(seeder(randGen)));
  }
#else
  size_t seed = 42;
  threadGens.emplace_back(seed);
#endif
}

std::mt19937 & randGen()  {
#ifdef _OPENMP
  return threadGens[omp_get_thread_num()];
#else
  return threadGens[0];
#endif
}
