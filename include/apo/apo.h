#ifndef APO_APO_H
#define APO_APO_H

#include <stdint.h>
#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <set>
#include <initializer_list>
#include <functional>
#include <queue>

#include "apo/program.h"
#include "apo/shared.h"

// #include "tensorflow/cc/client/client_session.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/cc/framework/tensor.h"

namespace apo {

using NodeSet = std::set<int32_t>;

static bool
rec_MatchPattern(const Program & prog, int pc, const Program & pattern, int patternPc, NodeVec & holes, std::vector<bool> & defined, NodeSet & nodes) {
  // matching a pattern hole
  if (IsArgument(patternPc)) {
    int holeIdx = GetHoleIndex(patternPc);
    if (defined[holeIdx]) {
      return holes[holeIdx] == pc; // matched a defined hole
    } else {
      // add a new definition
      holes[holeIdx] = pc;
      defined[holeIdx] = true;
      return true;
    }
  }

  assert (IsStatement(patternPc));
  if (pc < 0) {
    // argument matching
    return false; // can only match parameters with placeholders

  } else {
    nodes.insert(pc); // keep track of matched nodes

    OpCode oc = prog.code[pc].oc;
    if (oc != pattern.code[patternPc].oc) return false;
    if (oc == OpCode::Constant) {
      // constant matching
      return pattern.code[patternPc].getValue() == prog.code[pc].getValue(); // matching constant

    } else {
      // generic statement matching
      for (int i = 0; i < 2; ++i) {
        if (!rec_MatchPattern(prog, prog.code[pc].getOperand(i), pattern, pattern.code[patternPc].getOperand(i), holes, defined, nodes)) return false;
      }
      return true;
    }
  }
}


// matches program @pattern aligning the two at the statement at @pc of @prog (stores the operand index at "holes" (parameters) in @holes
static bool
MatchPattern(const Program & prog, int pc, const Program & pattern, NodeVec & holes, NodeSet & matchedNodes) {
  assert (pattern.size() > 0);
  holes.resize(pattern.num_Params(), -1);

  std::vector<bool> defined(pattern.numParams, false);
  // match the pattern
  int retIndex = pattern.code[pattern.size() - 1].getOperand(0);
  bool ok = rec_MatchPattern(prog, pc, pattern, retIndex, holes, defined, matchedNodes);
  if (Verbose) {
    std::cerr << "op match: " << ok << "\n";
  }

  // make sure that holes do not refer to nodes that are part of the match
  for (int i = 0; i < pattern.num_Params(); ++i) {
    if (IsStatement(holes[i]) && matchedNodes.count(holes[i])) { ok = false; break; }
  }

  // verify that there is no user of a matched node (except for the root)
  for (int i = 0; ok && (i < prog.size()); ++i) {
    if (!prog.code[i].isOperator()) continue;

    for (int o = 0; ok && (o < prog.code[i].num_Operands()); ++o) {
      int opIdx = prog.code[i].getOperand(o);
      if (opIdx < 0 || opIdx == pc) continue; // operand is parameter or the match root (don't care)
      if (matchedNodes.count(opIdx)) { // uses part of the match
        if (!matchedNodes.count(i)) { // .. only possible if this instruction is also part of th match
          ok = false;
          break;
        }
      }
    }
  }

  if (Verbose) {
    if (ok) {
      std::cerr << "Match. Holes:";
      for (int i = 0; i < pattern.numParams; ++i) {
        std::cerr << " "; PrintIndex(holes[i], std::cerr);
      }
      std::cerr << "\n";
    }
    else {
      std::cerr << "Mismatch.\n";
    }
  }
  return ok;
}

static void
rec_ErasePattern(Program & prog, int pc, const Program & pattern, int patternPc, NodeSet & erased) {
  if (!IsStatement(pc)) return; // do not erase by argument indices
  if (!erased.insert(pc).second) return;

  auto & stat = prog.code[pc];
  auto & patStat = pattern.code[patternPc];

  if (patStat.isOperator()) {
    for (int i = 0; i < stat.num_Operands(); ++i) {
      int nextPatIdx = patStat.getOperand(i);
      int nextProgIdx = stat.getOperand(i);
      if (IsStatement(nextPatIdx)) {
        rec_ErasePattern(prog, nextProgIdx, pattern, nextPatIdx, erased);
      }
    }
  }

  stat.oc = OpCode::Nop;
}

static void
ErasePattern(Program & prog, int pc, const Program & pattern) {
  if (pattern.size() <= 1) return; // nothing to erase
  NodeSet erased;
  rec_ErasePattern(prog, pc, pattern, pattern.getReturnIndex(), erased);
}


// rewrite rule lhs -> rhs
struct Rule {
  Program lhs;
  Program rhs;

  bool isExpanding(bool leftMatch) const {
    return getMatchProg(leftMatch).size() < getRewriteProg(leftMatch).size();
  }

  bool removesHoles(bool leftMatch) const {
    return getMatchProg(leftMatch).num_Params() > getRewriteProg(leftMatch).num_Params();
  }

  Program & getMatchProg(bool matchLeft) {
    if (matchLeft) return lhs; else return rhs;
  }
  const Program & getMatchProg(bool matchLeft) const {
    if (matchLeft) return lhs; else return rhs;
  }

  Program & getRewriteProg(bool matchLeft) {
    if (matchLeft) return rhs; else return lhs;
  }
  const Program & getRewriteProg(bool matchLeft) const {
    if (matchLeft) return rhs; else return lhs;
  }

  Rule(Program _lhs, Program _rhs)
  : lhs(_lhs)
  , rhs(_rhs)
  {}

  bool match_ext(bool matchLeft, const Program & prog, int pc, NodeVec & holes, NodeSet & matchedNodes) const {
    return MatchPattern(prog, pc, getMatchProg(matchLeft), holes, matchedNodes);
  }

  inline bool match(bool matchLeft, const Program & prog, int pc, NodeVec & holes) const {
    NodeSet matchedNodes;
    return match_ext(matchLeft, prog, pc, holes, matchedNodes);
  }

  void print(std::ostream & out) const {
    out << "Rule [[ lhs = "; lhs.print(out);
    out << "rhs = ";
    rhs.print(out);
    out << "]]\n";
  }

  void print(std::ostream & out, bool leftMatch) const {
    out << "Rule [[ from = "; getMatchProg(leftMatch).print(out);
    out << "to = ";
    getRewriteProg(leftMatch).print(out);
    out << "]]\n";
  }

  void dump() const { print(std::cerr); }
  void dump(bool leftMatch) const { print(std::cerr, leftMatch); }

  // applies this rule (after a match)
  void rewrite(bool matchLeft, Program & prog, int rootPc, NodeVec & holes) const {
    IF_VERBOSE { std::cerr << "-- rewrite at " << rootPc << " --\n"; prog.dump(); }

    // erase @lhs
    ErasePattern(prog, rootPc, getMatchProg(matchLeft));

    IF_VERBOSE { std::cerr << "-- after erase at " << rootPc << " --\n"; prog.dump(); }

    // make space for rewrite rule
    ReMap reMap;
    int afterInsertPc = prog.make_space(rootPc + 1, getRewriteProg(matchLeft).size(), reMap);
    // updated match root
    int mappedRootPc = reMap.count(rootPc) ? reMap[rootPc] : rootPc;

    IF_VERBOSE { std::cerr << "-- after make_space --\n"; prog.dump(); }

    // insert @rhs
    // patch holes (some positions were remapped)
    IF_VERBOSE { std::cerr << "Adjust remap\n"; }
    for (int i = 0; i < getMatchProg(matchLeft).num_Params(); ++i) {
      IF_VERBOSE { std::cerr << i << " -> " << holes[i]; }
      if (reMap.count(holes[i])) { // re-map hole indices for moved statements
        holes[i] = reMap[holes[i]];
      }
      IF_VERBOSE { std::cerr << "  " << holes[i] << "\n"; }
    }

    // rmappedRootPc replacement after the insertion of this pattern
    int rootSubstPc = getRewriteProg(matchLeft).link(prog, afterInsertPc - (getRewriteProg(matchLeft).size() - 1), holes);

    IF_VERBOSE {
      std::cerr << "-- after linking (root subst " << rootSubstPc << ") -- \n";
      prog.dump();
    }

    // prog.applyAfter(afterInsertPc, lambda[=](Statement & stat) { .. }
    for (node_t i = std::max(0, rootSubstPc + 1); i < prog.size(); ++i) {
      for (int o = 0; o < prog.code[i].num_Operands(); ++o) {
        if (prog.code[i].getOperand(o) == mappedRootPc) { prog.code[i].setOperand(o, rootSubstPc); }
      }
    }

    IF_VERBOSE {
      std::cerr << "-- after insert -- \n";
      prog.dump();
    }
    prog.compact();
  }
};


using RuleVec = std::vector<Rule>;
// create a basic rule set
static
RuleVec
BuildRules() {
  RuleVec rules;

  // (%b + %a) - %b --> %a
  {
    // try some matching
    Program lhs (2, {Statement(OpCode::Add, -2, -1), Statement(OpCode::Sub, 0, -2), build_ret(1) });
    Program rhs (1, {build_ret(-1)});
    rules.emplace_back(lhs, rhs);
  }

  // commutative rules (oc %a %b --> oc %b %a)
  for_such(IsCommutative, [&](OpCode oc){
    rules.emplace_back(
      Program(2, {Statement(oc, -1, -2), build_ret(0)}),
      Program(2, {Statement(oc, -2, -1), build_ret(0)})
    );
  });

  // associative rules (oc (%a %b) %c --> %a (%b %c)
  for_such(IsAssociative, [&](OpCode oc){
    rules.emplace_back(
      Program(3, {Statement(oc, -1, -2),
                  Statement(oc, 0, -3),
                  build_ret(1)
                  }),
      Program(3, {Statement(oc, -2, -3),
                  Statement(oc, -1, 0),
                  build_ret(1)
                  })
    );
  });

  // %a * (%b + %c) --> (%a * %b) + (%a * %c)
  rules.emplace_back(
      Program(3, {Statement(OpCode::Add, -2, -3),
                  Statement(OpCode::Mul, -1, 0),
                  build_ret(1)
                  }),
      Program(3, {Statement(OpCode::Mul, -1, -2),
                  Statement(OpCode::Mul, -1, -3),
                  Statement(OpCode::Add, 0, 1),
                  build_ret(2)
                  })
  );

  // (%a <oc> neutral) --> %a
  for_such(HasNeutralRHS, [&](OpCode oc){
    data_t neutral = GetNeutralRHS(oc);
    rules.emplace_back(
      Program(1, {build_const(neutral),
                  Statement(oc, -1, 0),
                  build_ret(1)
                  }),
      Program(1, {build_ret(-1)
                  })
    );
  });

  //  paranoid rule consistency checking
  for (int i = 0; i < rules.size(); ++i) {
    const auto & rule = rules[i];
    bool lhs = true;
    auto handler = [&rule,&lhs](int pc, std::string msg) -> bool {
      std::cerr << "Error at " << pc << " : " <<msg << "\n";
      return true;
    };

    lhs = true; if (!rule.lhs.verify(handler)) { std::cerr << "in rule lhs: " << i << "\n"; exit(-1); }
    lhs = false; if (!rule.rhs.verify(handler))  { std::cerr << "in rule rhs: " << i << "\n"; exit(-1); }
  }

  // arithmetic simplifaction rules
  return rules;
}

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
      return numUses > right.numUses; // prioritize unused elements
    }
  };

  std::uniform_real_distribution<float> constantRand;

  struct Sampler {
    std::vector<int> unused;
    std::priority_queue<Elem, std::vector<Elem>> opQueue;

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
      if (num_Unused() > 0) {
        bool allowPeek = !opQueue.empty() && (distToLimit > unused.size());
        std::uniform_int_distribution<int> opRand(0, unused.size() - 1 + allowPeek);

        // pick element from unused vector
        int idx = opRand(randGen());
        // std::cerr << "UNUED ID " << idx << "\n";
        assert(idx >= 0);
        if (idx < unused.size()) {
          int valIdx = unused[idx];
          unused.erase(unused.begin() + idx);
          opQueue.emplace(1, valIdx); // add to used-tracked queue for potential re-use
          return valIdx;
        }
      }

      // Otw, take element from queue
      Elem elem = opQueue.top();
      opQueue.pop(); // TODO add randomness
      int valIdx = elem.valIdx;
      elem.numUses++;
      opQueue.push(elem); // re-insert element

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

  RPG(const RuleVec & rules, int _numParams)
  : numParams(_numParams)
  {
    std::set<data_t> seen;
    for (auto & rule : rules) {
      collectConstants(rule.lhs, seen);
      collectConstants(rule.rhs, seen);
    }
    std::cerr << "found " << constVec.size() << " different constants in rule set!\n";
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
      bool forceOperand = length - i < S.num_Unused();

      if (!forceOperand && // hard criterion to avoid dead code
          (S.empty() || (constantRand(randGen()) <= pConstant))) { // soft preference criterion
        // random constant
        std::uniform_int_distribution<int> constIdxRand(0, constVec.size() - 1);
        int idx = constIdxRand(randGen());
        P.push(build_const(constVec[idx]));

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

    return &P;
  }
};





using DataVec = std::vector<data_t>;
struct RandExecutor {
  std::vector<DataVec> paramSets;

  RandExecutor(int numParams, int numSets)
  : paramSets()
  {
    std::uniform_int_distribution<data_t> argRand(0, std::numeric_limits<data_t>::max());

    for (int s = 0; s < numSets; ++s) {
      DataVec params;
      for (int i = 0; i< numParams; ++i) {
        params.push_back(argRand(randGen()));
      }
      paramSets.push_back(params);
    }
  }

  DataVec run(const Program & P) {
    DataVec results;
    for (const auto & params : paramSets) {
      results.push_back(Evaluate(P, params));
    }
    return results;
 }
};

static void
Print(std::ostream & out, const DataVec & D) {
  for (auto & d : D) out << ", " << d;
}

static bool Equal(const DataVec & A, const DataVec & B) {
  if (A.size() != B.size()) return false;
  for (size_t i = 0; i < A.size(); ++i) {
    if (A[i] != B[i]) return false;
  }
  return true;
}


} // namespace apo

#endif // APO_APO_H
