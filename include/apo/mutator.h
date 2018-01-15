#ifndef APO_MUTATOR_H
#define APO_MUTATOR_H

#include "apo/shared.h"
#include "apo/rules.h"

namespace apo {

struct Rewrite {
  int pc;
  int ruleId;
  bool leftMatch;

  static Rewrite
  fromModel(int pc, int r) {
    return Rewrite{pc, r / 2, (bool) (r % 2)};
  }

  int
  getEnumId() const {
    return (ruleId * 2) + (int) leftMatch;
  }

  std::ostream& print(std::ostream& out) const {
    out << pc << " " << ruleId << " " << leftMatch; return out;
  }

  bool operator==(const Rewrite & O) const {
    return pc == O.pc && ruleId == O.ruleId && leftMatch == O.leftMatch;
  }
};

struct Mutator {
  const RuleVec & rules;
  const float pExpand;

  Mutator(const RuleVec & _rules, float _pExpand=0.5)
  : rules(_rules)
  , pExpand(_pExpand)
  {}

  Rewrite
  mutate(Program & P, int steps) const {
    Rewrite mut;
    auto handler=[&mut](int pc, int ruleId, bool leftMatch, const Program & P) {
      mut.pc = pc, mut.ruleId = ruleId; mut.leftMatch = leftMatch;
    };

    mutate(P, steps, handler);

    return mut;
  }

  // try to apply this rule based on the rule id
  bool
  tryApply(Program & P, int pc, int ruleId, bool leftMatch) const {
    NodeVec holes;
    NodeSet matchedNodes;
    if (!rules[ruleId].match_ext(leftMatch, P, pc, holes, matchedNodes)) { return false; }
    apply(P, pc, ruleId, leftMatch, holes, matchedNodes);
    return true;
  }

  // single shot rule application by index (with random fillers for pattern holes)_
  void apply(Program & P, int pc, int ruleIdx, bool leftMatch, NodeVec & holes, NodeSet & matchedNodes) const {
    IF_VERBOSE { std::cerr << "Rewrite at " << pc << " with rule: "; rules[ruleIdx].dump(leftMatch); }

    // supplement holes
    if (!rules[ruleIdx].removesHoles(leftMatch)) {
      const auto & lhs = rules[ruleIdx].getMatchProg(leftMatch);
      const auto & rhs = rules[ruleIdx].getRewriteProg(leftMatch);

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
    rules[ruleIdx].rewrite(leftMatch, P, pc, holes);

    IF_DEBUG if (!P.verify()) {
      P.dump();
      abort();
    }
  }

  // apply a random mutating rewrite
  void mutate(Program & P, int steps, std::function<void(int pc, int ruleId, bool leftMatch, const Program & P)> handler) const {
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
      int ruleIdx = 0;
      for (int t = 0; t < rules.size(); ++t) {
        if (rules[t].isExpanding(leftMatch) != expandingMatch) continue;

        if (rules[t].match(leftMatch, P, pc, holes)) {
          hasMatch = true;
          ruleIdx = t;
          break;
        }
      }

      // no rule matches -> pick different rule
      if (!hasMatch) continue;

      // number of applicable rules to skip
      std::uniform_int_distribution<int> ruleRand(0, rules.size() - 1);
      int numSkips = ruleRand(randGen());

      NodeSet matchedNodes;
      for (int skip = 1; skip < numSkips; ) {
        ruleIdx = (ruleIdx + 1) % rules.size();
        if (rules[ruleIdx].isExpanding(leftMatch) != expandingMatch) continue;

        matchedNodes.clear();
        if (rules[ruleIdx].match_ext(leftMatch, P, pc, holes, matchedNodes)) {
          ++skip;
        }
      }

      // apply this rule
      apply(P, pc, ruleIdx, leftMatch, holes, matchedNodes);

      ++i;
    }
  }
};


} // namespace apo

#endif // APO_MUTATOR_H
