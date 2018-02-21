#ifndef APO_REWRITING_H
#define APO_REWRITING_H

#include "apo/program.h"
#include <vector>
#include <set>
#include <cstdint>
#include "apo/ADT/SmallSet.h"

namespace apo {

using NodeSet = llvm::SmallSet<node_t, 8>;

static bool
rec_MatchPattern(const Program & prog, int pc, const Program & pattern, int patternPc, NodeVec & holes, std::vector<bool> & defined, NodeSet & nodes, int & oMinMatched) {
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
    nodes.insert(pc); // keep track of matched nodes // SLOW!! (7.21)
    oMinMatched = std::min(oMinMatched, pc);

    OpCode oc = prog.code[pc].oc;
    if (oc != pattern.code[patternPc].oc) return false; // 3.38
    if (oc == OpCode::Constant) { // 5.05
      // constant matching
      return pattern.code[patternPc].getValue() == prog.code[pc].getValue(); // matching constant

    } else {
      // generic statement matching
      for (int i = 0; i < 2; ++i) {
        if (!rec_MatchPattern(prog, prog.code[pc].getOperand(i), pattern, pattern.code[patternPc].getOperand(i), holes, defined, nodes, oMinMatched)) return false;
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
  int minMatchedNode = pc; // (minimally the match root itself)
  bool ok = rec_MatchPattern(prog, pc, pattern, retIndex, holes, defined, matchedNodes, minMatchedNode);
  if (Verbose) {
    std::cerr << "op match: " << ok << "\n";
  }

  // make sure that holes do not refer to nodes that are part of the match
  for (int i = 0; i < pattern.num_Params(); ++i) {
    if (!IsStatement(holes[i])) continue;
    if (holes[i] < minMatchedNode) continue; // below matched nodes
    if (matchedNodes.count(holes[i])) { ok = false; break; }
  }

  // verify that there is no user of a matched node (except for the root)
  for (int i = minMatchedNode + 1; ok && (i < prog.size()); ++i) {
    if (!prog.code[i].isOperator()) continue;

    for (int o = 0; ok && (o < prog.code[i].num_Operands()); ++o) {
      int opIdx = prog.code[i].getOperand(o);
      if (opIdx < 0 || opIdx == pc) continue; // operand is parameter or the match root (don't care)
      if (opIdx >= minMatchedNode && matchedNodes.count(opIdx)) { // uses part of the match
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


struct RewritePair {
  Program lhs;
  Program rhs;
  bool symmetric;

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

  // return match root opcode (for indexing)
  bool getRootOpCode(bool matchLeft, OpCode & rootOc) const {
    const auto & matchProg = getMatchProg(matchLeft);
    int retPc = matchProg.getReturnIndex();
    if (!IsStatement(retPc)) return false;
    else {
      rootOc = matchProg(retPc).oc;
      return true;
    }
  }

  RewritePair(Program _lhs, Program _rhs)
  : lhs(_lhs)
  , rhs(_rhs)
  , symmetric(false)
  {}

  RewritePair(Program _lhs, Program _rhs, bool isSymmetric)
  : lhs(_lhs)
  , rhs(_rhs)
  , symmetric(isSymmetric)
  {}

  bool match_ext(bool matchLeft, const Program & prog, int pc, NodeVec & holes, NodeSet & matchedNodes) const {
    return MatchPattern(prog, pc, getMatchProg(matchLeft), holes, matchedNodes);
  }

  inline bool match(bool matchLeft, const Program & prog, int pc, NodeVec & holes) const {
    NodeSet matchedNodes;
    return match_ext(matchLeft, prog, pc, holes, matchedNodes);
  }

  void print(std::ostream & out) const {
    out << "RewritePair " ;
    if (symmetric) out << "S ";
    out << "[[ lhs = "; lhs.print(out);
    out << "rhs = ";
    rhs.print(out);
    out << "]]\n";
  }

  void print(std::ostream & out, bool leftMatch) const {
    out << "RewritePair ";
    if (symmetric) out << "S ";
    out << "[[ from = "; getMatchProg(leftMatch).print(out);
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
    const int maxExpand = getRewriteProg(matchLeft).size();
    ReMap reMap(prog.size() + maxExpand);

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

using RewritePairVec = std::vector<RewritePair>;


// create a basic rule set
static
RewritePairVec
BuildRewritePairs() {
  RewritePairVec rules;

#if 0
  // (%a + %b) - %c --> (%a - %c) + %b
  {
    Program lhs (3, {Statement(OpCode::Add, -1, -2), Statement(OpCode::Sub, 1, -3), build_ret(1) });
    Program rhs (3, {Statement(OpCode::Sub, -1, -3), Statement(OpCode::Add, 1, -2), build_ret(1) });
    rules.emplace_back(lhs, rhs);
  }
#else
  // old rule
  // (%b + %a) - %b --> %a
  {
    // try some matching
    Program lhs (2, {Statement(OpCode::Add, -2, -1), Statement(OpCode::Sub, 0, -2), build_ret(1) });
    Program rhs (1, {build_ret(-1)});
    rules.emplace_back(lhs, rhs);
  }
#endif

  // commutative rules (oc %a %b --> oc %b %a)
  for_such(IsCommutative, [&](OpCode oc){
    rules.emplace_back(
      Program(2, {Statement(oc, -1, -2), build_ret(0)}),
      Program(2, {Statement(oc, -2, -1), build_ret(0)}),
      true // symmetric
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




} // namespace apo

#endif // APO_REWRITING_H
