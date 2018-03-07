#ifndef APO_RPG_H
#define APO_RPG_H

#include <random>
#include "apo/program.h"
#include <queue>
#include "apo/ruleBook.h"
#include "apo/shared.h"
#include "apo/extmath.h"

namespace apo {

static CatDist
CreateRandomOpCodeDist() {
  const double smoothing = 0.001; // minimal mass *before* scaling
  const double exponent = 6; // increase to get closer to hard on/off behavior

  CatDist dist((int) OpCode::End_OpCode + 1, 0.0);

#if 0
  for (int i = 0; i < dist.size(); ++i) {
    dist[i] = smoothing + pow(drawUnitRand(), exponent);
  }

  // zero out utilility opCodes
  dist[(int) OpCode::Nop] = 0.0;
  dist[(int) OpCode::Pipe] = 0.0;
  dist[(int) OpCode::Return] = 0.0;
  dist[(int) OpCode::Constant] = 0.0;
#else
  dist[(int) OpCode::Add] = 1.0;
#endif

  Normalize(dist);
  return dist;
}

struct RPG {
  const RuleBook & ruleBook;
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
        bool allowPeek = !opQueue.empty() && (distToLimit + 1 > unused.size());

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

  RPG(const RuleBook & _ruleBook, int _numParams)
  : ruleBook(_ruleBook)
  , numParams(_numParams)
  {}

  data_t
  drawRandomConstant() const {
    // random constant
    std::uniform_int_distribution<int> constIdxRand(0, ruleBook.constVec.size() * 2 - 1);
    int idx = constIdxRand(randGen());
    if (idx< ruleBook.constVec.size()) {
      return ruleBook.constVec[idx]; // known random constant from pool
    } else {
      std::uniform_int_distribution<data_t> constValRand(std::numeric_limits<data_t>::min(), std::numeric_limits<data_t>::max());
      return constValRand(randGen()); // proper random constant
    }
  }

  Program*
  generate_ext(int length) {
    CatDist ocDist = CreateRandomOpCodeDist();

    Program & P = *(new Program(numParams, {}));
    P.code.reserve(length);

    Sampler S;
    // push (optinonal) arguments
    for (int a = 0; a < numParams; ++a) {
      // P.push(build_pipe(-a - 1));
      S.addOptionalUseable(-a - 1);
    }

    const double pConstant = 0.2 + 0.4 * drawUnitRand();
    for (int i = 0; i < length - 1; ++i) {
      bool forceOperand = length - i < S.num_Unused() + 2;

      if (!forceOperand && // hard criterion to avoid dead code
          (S.empty() || (drawUnitRand() <= pConstant))) { // soft preference criterion
        data_t constVal = drawRandomConstant();
        P.push(build_const(constVal));

      } else {
        // pick random opCode and operands
        // int beginBin = (int) OpCode::Begin_Binary;
        // int endBin = (int) OpCode::End_Binary;

        OpCode oc = (OpCode) SampleCategoryDistribution(ocDist, drawUnitRand());
        int distToLimit = length - 1 - i;
        int firstOp = S.acquireOperand(distToLimit);
        int sndOp = S.acquireOperand(distToLimit);
        P.push(Statement(oc, firstOp, sndOp));
      }

      // publish the i-th instruction as useable in an operand position
      S.addUseable(i);
    }

    P.push(build_ret(P.size() - 1));

    IF_DEBUG {
      if (!P.verify()) {
        P.dump();
        abort();
      }
    }

    // DEBUG
    if (getenv("DUMP_PROGS")) P.dump();

    return &P;
  }

  Program*
  generate(int length) {
    Program & P = *(new Program(numParams, {}));
    P.code.reserve(length);

    Sampler S;
    // push (optinonal) arguments
    for (int a = 0; a < numParams; ++a) {
      // P.push(build_pipe(-a - 1));
      S.addOptionalUseable(-a - 1);
    }

    const double pConstant = 0.20;
    for (int i = 0; i < length - 1; ++i) {
      bool forceOperand = length - i < S.num_Unused() + 2;

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
      S.addUseable(i);
    }

    P.push(build_ret(P.size() - 1));

    IF_DEBUG {
      if (!P.verify()) {
        P.dump();
        abort();
      }
    }

    // DEBUG
    if (getenv("DUMP_PROGS")) P.dump();

    return &P;
  }
};


} // namespace apo

#endif // APO_RPG_H
