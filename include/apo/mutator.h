#ifndef APO_MUTATOR_H
#define APO_MUTATOR_H

#include "apo/shared.h"
#include "apo/rewriting.h"
#include "apo/ruleBook.h"

namespace apo {

struct Mutator {
  const RuleBook & ruleBook;

  Mutator(const RuleBook & _ruleBook)
  : ruleBook(_ruleBook)
  {}

  Action
  mutate(Program & P, int steps, float pExpand) const {
    Action mut;
    auto handler=[&mut](int pc, int ruleId, const Program & P) {
      mut.pc = pc, mut.ruleId = ruleId;
    };

    mutate(P, steps, pExpand, handler);

    return mut;
  }

  bool
  tryApply(Program & P, const Action act) const {
    return tryApply(P, act.pc, act.ruleId);
  }

  // try to apply this rule based on the rule id
  bool
  tryApply(Program & P, int pc, int ruleId) const {
    NodeVec holes;
    NodeSet matchedNodes;
    const auto & rewRule = ruleBook.getRewriteRule(ruleId);
    if (!ruleBook.matchRule_ext(rewRule, P, pc, holes, matchedNodes)) { return false; }
    applyRule(P, pc, rewRule, holes, matchedNodes);
    return true;
  }

  // single shot rule application by index (with random fillers for pattern holes)_
  void applyRule(Program & P, int pc, const RewriteRule & rewRule, NodeVec & holes, NodeSet & matchedNodes) const {
    // IF_VERBOSE { std::cerr << "Rewrite at " << pc << " with rule: "; rewritePairs[pairIdx].dump(leftMatch); }

    // supplement holes
    int numLeftHoles = ruleBook.getLeftHandHoles(rewRule);
    int numRightHoles = ruleBook.getRightHandHoles(rewRule);

    if (numRightHoles > numLeftHoles) {
      // const auto & lhs = rewritePairs[pairIdx].getMatchProg(leftMatch);
      // const auto & rhs = rewritePairs[pairIdx].getRewriteProg(leftMatch);

      int lowestVal = -(P.num_Params());
      int highestVal = std::max(pc - numRightHoles, 0) - 1;
      assert(lowestVal <= highestVal);
      std::uniform_int_distribution<int> opRand(lowestVal, highestVal);

      holes.resize(numRightHoles, 0);
      for (int h = numLeftHoles;  h < numRightHoles; ++h) {
        int opIdx;
        // draw operands that do not occur in the pattern
        do {
          opIdx = opRand(randGen());
        } while (matchedNodes.count(opIdx));

        holes[h] = opIdx;
      }
    }

    // apply rewrite
    ruleBook.transform(rewRule, P, pc, holes);//rewritePairs[pairIdx].rewrite(leftMatch, P, pc, holes);

    IF_DEBUG if (!P.verify()) {
      P.dump();
      abort();
    }
  }

  // apply a random mutating rewrite
  void mutate(Program & P, int steps, float pExpand, std::function<void(int pc, int ruleId, const Program & P)> handler) const {
    if (steps <= 0) return;

    std::uniform_real_distribution<float> shrinkRand(0, 1);
    std::uniform_int_distribution<int> ruleRand(0, ruleBook.num_Rules() - 1);

    for (int i = 0; i < steps; ) {
      // pick a random pc
      std::uniform_int_distribution<int> pcRand(0, P.size() - 2); // don't allow return rewrites
      int pc = pcRand(randGen());

      // pick a random rule
      bool expandingMatch = shrinkRand(randGen()) < pExpand;

      NodeVec holes;

      // std::cerr << "(" << pc << ", " << leftMatch << ", " << numSkips << ")\n";
      // check if any rule matches
      bool hasMatch = false;
      int ruleId = 0;

      const RewriteRule * rewRule = nullptr;
      for (int t = 0; t < ruleBook.num_Rules(); ++t) {
        rewRule = &ruleBook.getRewriteRule(t);
        if (ruleBook.isExpanding(*rewRule) != expandingMatch) continue;

        if (ruleBook.matchRule(*rewRule, P, pc, holes)) {
          hasMatch = true;
          ruleId = t;
          break;
        }
      }

      // no rule matches -> pick different rule
      if (!hasMatch) continue;

      // number of applicable rewritePairs to skip
      int numSkips = ruleRand(randGen());

      NodeSet matchedNodes;
      for (int skip = 1; skip < numSkips; ) {
        ruleId = (ruleId + 1) % ruleBook.num_Rules();
        rewRule = &ruleBook.getRewriteRule(ruleId);
        if (ruleBook.isExpanding(*rewRule) != expandingMatch) continue;

        matchedNodes.clear();
        if (ruleBook.matchRule_ext(*rewRule, P, pc, holes, matchedNodes)) {
          ++skip;
        }
      }

      // call back
      handler(pc, ruleId, P);

      // apply this rule
      applyRule(P, pc, *rewRule, holes, matchedNodes);

      ++i;
    }
  }
};


} // namespace apo

#endif // APO_MUTATOR_H
