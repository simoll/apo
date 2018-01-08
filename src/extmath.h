#ifndef APO_EXTMATh_H
#define APO_EXTMATh_H

static void
Normalize(std::vector<float> & dist) {
  double a = 0;
  for (auto v : dist) { a += v; }
  if (a == 0.0) return;
  for (auto & v : dist) { v /= a; }
}

#endif // APO_EXTMATh_H
