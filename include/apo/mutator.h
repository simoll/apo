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
    if (!ruleBook.matchRule_ext(ruleId, P, pc, holes, matchedNodes)) { return false; }
    applyRule(P, pc, ruleId, holes, matchedNodes);
    return true;
  }

  // shuffle up the instructions in the program
  void shuffle(Program & P, int numShuffles) const {
    if (P.size() <= 2) return; // dont' shuffle ret+inst programs
    std::uniform_int_distribution<int> pcRand(1, P.size() - 2); // don't allow return rewrites
    // std::cerr << "SHUFFLE!!\n";
    // P.dump();
    for (int i = 0; i < numShuffles; ++i) {
      int pc = pcRand(randGen());
      auto & stat = P.code[pc];

      // check independence of pc and (pc-1)
      bool legalSwap = true;
      if (stat.isOperator()) {
        for (int o = 0; o < stat.num_Operands(); ++o) {
          if (stat.getOperand(o) == pc - 1) {
            legalSwap = false; break;
          }
        }
      }

      if (!legalSwap) continue;

      // swap operands of down stream instructions (pc <-> pc - 1)
      for (int j = pc + 1; j < P.size(); ++j) {
        if (!P.code[j].isOperator()) continue;
        for (int o = 0; o < P.code[j].num_Operands(); ++o) {
          if (P.code[j].getOperand(o) == pc) {
            P.code[j].setOperand(o, pc - 1);
          } else if (P.code[j].getOperand(o) == pc - 1) {
            P.code[j].setOperand(o, pc);
          }
        }
      }

      // perform the swap
      std::swap(P.code[pc], P.code[pc - 1]);

      // std::cerr << "after " << i << " : "; P.dump();
    }

    assert(P.verify());
  }

  // TODO rewrite using match ledger
  // apply a random mutating rewrite
  void mutate(Program & P, int steps, float pExpand, std::function<void(int pc, int ruleId, const Program & P)> handler) const {
    if (steps <= 0) return;

    std::uniform_real_distribution<float> shrinkRand(0, 1);
    std::uniform_int_distribution<int> ruleRand(0, std::numeric_limits<int>::max());

    for (int i = 0; i < steps; ) {
      // pick a random pc
      std::uniform_int_distribution<int> pcRand(0, std::max(0, P.size() - 2)); // don't allow return rewrites
      int pc = pcRand(randGen());

      // pick a random rule
      bool expandingMatch = shrinkRand(randGen()) < pExpand;

      NodeVec holes;

      // std::cerr << "(" << pc << ", " << leftMatch << ", " << numSkips << ")\n";
      // check if any rule matches
      bool hasMatch = false;

      const auto & ruleVec = ruleBook.getRulesForOpCode(P(pc).oc, expandingMatch);
      if (ruleVec.empty()) continue; // no applicable rule

      // drawn random sample base
      int ruleVecId = ruleRand(randGen()) % ruleVec.size();

      // check for match
      NodeSet matchedNodes;
      int ruleId = ruleVec[ruleVecId];

      if (!ruleBook.matchRule_ext(ruleId, P, pc, holes, matchedNodes)) {
        continue; // re-draw
      }

      // call back
      handler(pc, ruleId, P);

      // apply this rule
      applyRule(P, pc, ruleId, holes, matchedNodes);

      ++i;
    }
  }

private:
  // single shot rule application by index (with random fillers for pattern holes)_
  void applyRule(Program & P, int pc, int ruleId, NodeVec & holes, NodeSet & matchedNodes) const {
    // IF_VERBOSE { std::cerr << "Rewrite at " << pc << " with rule: "; rewritePairs[pairIdx].dump(leftMatch); }

    // supplement holes
    int numLeftHoles = ruleBook.getLeftHandHoles(ruleId);
    int numRightHoles = ruleBook.getRightHandHoles(ruleId);

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
    Program* copP = nullptr;
    IF_DEBUG {
      copP = new Program(P);
    }

    ruleBook.transform(ruleId, P, pc, holes);//rewritePairs[pairIdx].rewrite(leftMatch, P, pc, holes);

    IF_DEBUG {
      if (!P.verify()) {
        std::cerr << "Offending ruleId " << ruleId << " at pc = " << pc << "\n";
        std::cerr << "before:\n";
        copP->dump();
        std::cerr << "after:\n";
        P.dump();
        abort();
      }
      delete copP;
    }
  }

};


} // namespace apo

#endif // APO_MUTATOR_H
