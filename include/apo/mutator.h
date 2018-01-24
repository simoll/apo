#ifndef APO_MUTATOR_H
#define APO_MUTATOR_H

#include "apo/shared.h"
#include "apo/rewriting.h"

namespace apo {

struct RewriteAction {
  int pc;
  int pairId;
  bool leftMatch;

  int
  getEnumId() const {
    return (pairId * 2) + (leftMatch ? 1 : 0);
  }

  std::ostream& print(std::ostream& out) const {
    out << pc << " " << pairId << " " << leftMatch; return out;
  }

  bool operator==(const RewriteAction & O) const {
    return pc == O.pc && pairId == O.pairId && leftMatch == O.leftMatch;
  }

  RewriteAction(int _pc, int _pairId, bool _leftMatch)
  : pc(_pc), pairId(_pairId), leftMatch(_leftMatch)
  {}
  RewriteAction() {}
};

struct Mutator {
  const RewritePairVec & rewritePairs;
  const float pExpand;

  Mutator(const RewritePairVec & _rewritePairs, float _pExpand=0.5)
  : rewritePairs(_rewritePairs)
  , pExpand(_pExpand)
  {}

  RewriteAction
  mutate(Program & P, int steps) const {
    RewriteAction mut;
    auto handler=[&mut](int pc, int pairId, bool leftMatch, const Program & P) {
      mut.pc = pc, mut.pairId = pairId; mut.leftMatch = leftMatch;
    };

    mutate(P, steps, handler);

    return mut;
  }

  bool
  tryApply(Program & P, const RewriteAction act) const {
    return tryApply(P, act.pc, act.pairId, act.leftMatch);
  }

  // try to apply this rule based on the rule id
  bool
  tryApply(Program & P, int pc, int pairId, bool leftMatch) const {
    NodeVec holes;
    NodeSet matchedNodes;
    if (!rewritePairs[pairId].match_ext(leftMatch, P, pc, holes, matchedNodes)) { return false; }
    apply(P, pc, pairId, leftMatch, holes, matchedNodes);
    return true;
  }

  // single shot rule application by index (with random fillers for pattern holes)_
  void apply(Program & P, int pc, int pairIdx, bool leftMatch, NodeVec & holes, NodeSet & matchedNodes) const {
    IF_VERBOSE { std::cerr << "Rewrite at " << pc << " with rule: "; rewritePairs[pairIdx].dump(leftMatch); }

    // supplement holes
    if (!rewritePairs[pairIdx].removesHoles(leftMatch)) {
      const auto & lhs = rewritePairs[pairIdx].getMatchProg(leftMatch);
      const auto & rhs = rewritePairs[pairIdx].getRewriteProg(leftMatch);

      int lowestVal = -(P.num_Params());
      int highestVal = std::max(pc - rhs.size(), 0) - 1;
      assert(lowestVal <= highestVal);
      std::uniform_int_distribution<int> opRand(lowestVal, highestVal);

      holes.resize(rhs.num_Params(), 0);
      for (int h = lhs.num_Params(); h < rhs.num_Params(); ++h) {
        int opIdx;
        // draw operands that do not occur in the pattern
        do {
          opIdx = opRand(randGen());
        } while (matchedNodes.count(opIdx));

        holes[h] = opIdx;
      }
    }

    // apply rewrite
    rewritePairs[pairIdx].rewrite(leftMatch, P, pc, holes);

    IF_DEBUG if (!P.verify()) {
      P.dump();
      abort();
    }
  }

  // apply a random mutating rewrite
  void mutate(Program & P, int steps, std::function<void(int pc, int pairId, bool leftMatch, const Program & P)> handler) const {
    if (steps <= 0) return;

    for (int i = 0; i < steps; ) {
      // pick a random pc
      std::uniform_int_distribution<int> pcRand(0, P.size() - 2); // don't allow return rewrites
      int pc = pcRand(randGen());

      // pick a random rule
      std::uniform_real_distribution<float> shrinkRand(0, 1);
      bool expandingMatch = shrinkRand(randGen()) < pExpand;

      std::uniform_int_distribution<int> flipRand(0, 1);
      bool leftMatch = flipRand(randGen())  == 1;

      NodeVec holes;

      // std::cerr << "(" << pc << ", " << leftMatch << ", " << numSkips << ")\n";
      // check if any rule matches
      bool hasMatch = false;
      int pairIdx = 0;
      for (int t = 0; t < rewritePairs.size(); ++t) {
        if (rewritePairs[t].isExpanding(leftMatch) != expandingMatch) continue;

        if (rewritePairs[t].match(leftMatch, P, pc, holes)) {
          hasMatch = true;
          pairIdx = t;
          break;
        }
      }

      // no rule matches -> pick different rule
      if (!hasMatch) continue;

      // number of applicable rewritePairs to skip
      std::uniform_int_distribution<int> ruleRand(0, rewritePairs.size() - 1);
      int numSkips = ruleRand(randGen());

      NodeSet matchedNodes;
      for (int skip = 1; skip < numSkips; ) {
        pairIdx = (pairIdx + 1) % rewritePairs.size();
        if (rewritePairs[pairIdx].isExpanding(leftMatch) != expandingMatch) continue;

        matchedNodes.clear();
        if (rewritePairs[pairIdx].match_ext(leftMatch, P, pc, holes, matchedNodes)) {
          ++skip;
        }
      }

      // apply this rule
      apply(P, pc, pairIdx, leftMatch, holes, matchedNodes);

      ++i;
    }
  }
};


} // namespace apo

#endif // APO_MUTATOR_H
