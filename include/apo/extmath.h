#ifndef APO_EXTMATh_H
#define APO_EXTMATh_H

#include <ostream>
#include <algorithm>
#include <vector>
#include <cassert>

#define EPS 0.00001

using CatDist = std::vector<float>;

void PrintDist(const CatDist & dist, std::ostream &);

static void
Normalize(std::vector<float> & dist) {
  double a = 0;
  for (auto v : dist) { a += v; }
  if (a == 0.0) return;
  for (auto & v : dist) { v /= a; }
}

static int
SampleCategoryDistribution(CatDist & catDist, double p) {
  assert(((0.0 <= p) && (p <= 1.0)) && "not a valid sample value [0,1]");

  double catBase = 0.0;
  for (int e = 0; e < catDist.size(); ++e) {
    double lastCatBase = catBase;
    double catInterval = catDist[e];
    catBase = lastCatBase + catInterval;
    if (p > catBase) continue; // skip rule. Otw, sample pos is inside target range for this rule
    if (lastCatBase + EPS >= catBase) continue; // skip empty intervals
    assert(p >= lastCatBase);
    return e;
  }

  assert(catBase + EPS >= 1.0);
  abort();
}

static double
Clamp(double v, double low, double high) { return std::min<double>(high, std::max<double>(low, v)); }

#endif // APO_EXTMATH_H
