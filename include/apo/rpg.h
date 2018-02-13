#ifndef APO_RPG_H
#define APO_RPG_H

#include <random>
#include "apo/program.h"
#include <queue>
#include "apo/ruleBook.h"

namespace apo {


struct RPG {
  int numParams;

  struct Elem {
    int numUses;
    int valIdx; // instruction or arg

    /// Elem()
    /// : numUses(0)
    /// , valIdx(0)
    /// {}

    Elem(int _numUses, int _valIdx) : numUses(_numUses), valIdx(_valIdx) {}
    bool operator< (const Elem & right) const {
      // tuple order
      return (numUses < right.numUses) ||
             (numUses == right.numUses && (valIdx < right.valIdx)); // prioritize unused elements
    }
  };

  struct Sampler {
    std::vector<int> unused;
    std::set<Elem> opQueue;

    const int peekBias = 1;

    int num_Unused() const { return unused.size(); }

    // value may be used but does not have to (arguments)
    void addOptionalUseable(int valIdx) {
      // std::cerr << "OPT " << valIdx << "\n";
      opQueue.emplace(0, valIdx);
    }

    // value has to be used (instruction)
    void addUseable(int valIdx) {
      // std::cerr << "MUST USE: " << valIdx << "\n";
      unused.push_back(valIdx);
    }

    // fetch a random operand and increase its use count
    int acquireOperand(int distToLimit) {

      // check whether its safe to peek to already used operands at this point
      int peekDepth = opQueue.size() - 1;
      if (num_Unused() > 0) {
        bool allowPeek = !opQueue.empty() && (distToLimit > unused.size());

        const float pPeek = 0.8;
        if (!allowPeek || (drawUnitRand() > pPeek) ) {
          // pick a random unused element
          std::uniform_int_distribution<int> opRand(0, unused.size() - 1);

          // index into unused vector
          int idx = opRand(randGen());

          // std::cerr << "UNUED ID " << idx << "\n";
          int valIdx = unused[idx];
          unused.erase(unused.begin() + idx);
          opQueue.emplace(1, valIdx); // add to used-tracked queue for potential re-use
          return valIdx;
        }
      }

      assert(!opQueue.empty());

    // skip some queue elements (controlled by peekDepth)
#if 1
      const float pSkip = 0.5;
      auto itElem = opQueue.begin();
      peekDepth = std::min<int>(peekDepth, opQueue.size() - 1);
      int j = 0;
      for (j = 0; j < peekDepth ; ++itElem, ++j) {
        if (drawUnitRand() > pSkip) {
          break;
        }
      }

      // std::cerr << "PICK " << j << " of " << opQueue.size() << "\n";

#endif

      // Otw, take element from queue
      Elem elem = *itElem;
      opQueue.erase(itElem);

      int valIdx = elem.valIdx;

      // re-insert used element
      elem.numUses++;
      opQueue.insert(elem);

      return valIdx;
    }

    bool empty() const { return unused.empty() && opQueue.empty(); }
  };

  std::vector<data_t> constVec; // recognized constants in the match rules
  const double pConstant = 0.20;

  void collectConstants(const Program & prog, std::set<data_t> & seen) {
    for (auto & stat : prog.code) {
      if (stat.isConstant()) {
        if (seen.insert(stat.getValue()).second) {
          constVec.push_back(stat.getValue());
        }
      }
    }
  }

  RPG(const RuleBook & ruleBook, int _numParams)
  : numParams(_numParams)
  {
    std::set<data_t> seen;
    // FIXME factor this out
    for (auto & rule : ruleBook.rewritePairs) {
      collectConstants(rule.lhs, seen);
      collectConstants(rule.rhs, seen);
    }
    std::cerr << "found " << constVec.size() << " different constants in rule set!\n";
  }

  data_t
  drawRandomConstant() const {
    // random constant
    std::uniform_int_distribution<int> constIdxRand(0, constVec.size() * 2 - 1);
    int idx = constIdxRand(randGen());
    if (idx< constVec.size()) {
      return constVec[idx]; // known random constant from pool
    } else {
      std::uniform_int_distribution<data_t> constValRand(std::numeric_limits<data_t>::min(), std::numeric_limits<data_t>::max());
      return constValRand(randGen()); // proper random constant
    }
  }

  Program*
  generate(int length) {
    Program & P = *(new Program(numParams, {}));
    P.code.reserve(length);

    Sampler S;
    // wrap all arguments in pipes
    for (int a = 0; a < numParams; ++a) {
      P.push(build_pipe(-a - 1));
      S.addOptionalUseable(a);
    }

    int s = numParams;
    for (int i = 0; i < length - 1; ++i, ++s) {
      bool forceOperand = length - i < S.num_Unused() + 1;

      if (!forceOperand && // hard criterion to avoid dead code
          (S.empty() || (drawUnitRand() <= pConstant))) { // soft preference criterion
        data_t constVal = drawRandomConstant();
        P.push(build_const(constVal));

      } else {
        // pick random opCode and operands
        int beginBin = (int) OpCode::Begin_Binary;
        int endBin = (int) OpCode::End_Binary;

        std::uniform_int_distribution<int> ocRand(beginBin, endBin);
        OpCode oc = (OpCode) ocRand(randGen());
        int distToLimit = length - 1 - i;
        int firstOp = S.acquireOperand(distToLimit);
        int sndOp = S.acquireOperand(distToLimit);
        P.push(Statement(oc, firstOp, sndOp));
      }

      // publish the i-th instruction as useable in an operand position
      S.addUseable(s);
    }

    P.push(build_ret(P.size() - 1));

    assert(P.verify());

    // DEBUG
    if (getenv("DUMP_PROGS")) P.dump();

    return &P;
  }
};


} // namespace apo

#endif // APO_RPG_H
