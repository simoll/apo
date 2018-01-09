#include "apo/extmath.h"

#include <set>

void
PrintDist(const CatDist & dist, std::ostream & out) {
#if 0
  for (int i = 0; i < dist.size(); ++i) {
    out << i << " : " << dist[i] << "\n";
  }
  // return; // DEBUG HACK
#endif

  if (dist.empty()) return;

  float lastMaxElem = std::numeric_limits<float>::max();
  const int topElems = 3;
  std::set<int> emittedElems;
  for (int i = 0; i < std::min<int>(dist.size(), topElems); ++i) {
    int topId = -1;
    float maxElem = dist[0];
    for (int j = 0; j < dist.size(); ++j) {
      if (dist[j] > lastMaxElem) continue;
      if (topId >= 0 && (dist[j] < maxElem)) continue;
      if (emittedElems.count(j)) continue; // printed that one before

      topId = j;
      maxElem = dist[j];
    }

    assert(topId >= 0);
    emittedElems.insert(topId);
    lastMaxElem = maxElem;

    if (maxElem == 0.0) return;
    out << topId << ":" << maxElem;
    if (i + 1 < topElems) out << " ";
  }
}
