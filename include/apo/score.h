#ifndef APO_SCORE_H
#define APO_SCORE_H

namespace apo {

static int
GetProgramScore(const Program &P) {
  int score = 0;
  for (auto & stat : P.code) {
    if ((stat.oc == OpCode::Constant) ||
        (stat.oc == OpCode::Nop)) {
      continue;
    }
    score++;
  }
  return score;
}


}


#endif // APO_SCORE_H
