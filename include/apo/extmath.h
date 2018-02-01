#ifndef APO_EXTMATh_H
#define APO_EXTMATh_H

#include <ostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <iostream>
#include <functional>

#include "apo/ADT/SmallVector.h"

#define EPS 0.00001

using CatDist = std::vector<float>;

void DumpDist(const CatDist & dist);
void PrintDist(const CatDist & dist, std::ostream &);

static void
Normalize(CatDist & dist) {
  double a = 0;
  for (auto v : dist) { a += v; }
  if (a == 0.0) return;
  for (auto & v : dist) { v /= a; }
}

static bool
IsValidDistribution(const CatDist & catDist) {
  double sum = 0.0;
  for (float v : catDist) {
    if (v < 0.0) return false;
    sum += v;
  }
  return sum + EPS >= 1.0;
}

static void
VisitDescending(const CatDist & catDist, std::function<bool(float, int)> handlerFunc) {
// collect events with highest probability mass
  float topMass = 2.0; // entries can not be larger than 1.0

  bool keepGoing = true;
  do {
    float maxMass = 0.0;

    llvm::SmallVector<int, 2> likelyEventVec;
    for (int i = 0; i < catDist.size(); ++i) {
      float thisMass = catDist[i];
      if (thisMass >= topMass) continue; // skip

      // new incumbent mass
      if (thisMass > maxMass) {
        likelyEventVec.clear();
        maxMass = thisMass;
      }

      // collect all events with this mass
      if (thisMass == maxMass) {
        likelyEventVec.push_back(i);
      }
    }

    // no event seen
    if (likelyEventVec.empty()) return;

    // visit all events with highest admissable mass
    for (int event : likelyEventVec) {
      keepGoing = handlerFunc(maxMass, event);
      if (!keepGoing) return;
    }

    topMass = maxMass; // lower the cap
  } while (keepGoing);
}

static int
SampleMostLikely(const CatDist & catDist, double p) {
// collect events with highest probability mass
  float maxMass = 0.0;
  llvm::SmallVector<int, 2> likelyEventVec;

  for (int i = 0; i < catDist.size(); ++i) {
    if (catDist[i] > maxMass) {
      likelyEventVec.clear();
      maxMass = catDist[i];
    }
    if (catDist[i] == maxMass) {
      likelyEventVec.push_back(i);
    }
  }

  // invalid distribution
  if (maxMass <= 0.0) return -1;

  // otw, sample among most likely
  assert(likelyEventVec.size() > 0);

  if (likelyEventVec.size() == 1) { return likelyEventVec[0]; }
  int idx = floor(likelyEventVec.size() * p);
  assert(0 <= idx && idx < likelyEventVec.size());
  return idx;
}

static int
SampleCategoryDistribution(const CatDist & catDist, double p) {
  assert(((0.0 <= p) && (p <= 1.0)) && "not a valid sample value [0,1]");

  double catBase = 0.0;
  for (int e = 0; e < catDist.size(); ++e) {
    double lastCatBase = catBase;
    double catInterval = catDist[e];
    catBase = lastCatBase + catInterval;
    if (p > catBase) continue; // skip category. Otw, sample pos is inside target range for this rule
    if (lastCatBase >= catBase) continue; // skip empty intervals
    assert(p >= lastCatBase);
    return e;
  }

  assert(catBase + EPS >= 1.0); // check that @catDist is a valid distribution
  // Categories may not add up to 1.0 due to fp imprecision.
  // So if p does not lie in a checked category interval, p points to the interval between the accumulated catogories and 1.0
  return catDist.size() - 1;
}

static double
Clamp(double v, double low, double high) { return std::min<double>(high, std::max<double>(low, v)); }

#endif // APO_EXTMATH_H
