#ifndef APO_RULEBOOK_H
#define APO_RULEBOOK_H

#include "apo/rewriting.h"
#include "apo/modelConfig.h"
#include "apo/exec.h"

#include <unordered_set>
#include <unordered_map>

namespace apo {

// generic action wrapper
struct Action {
  int pc;
  int ruleId;

  std::ostream & print(std::ostream& out) const {
    out << "Action (pc=" << pc << ", ruleId=" << ruleId << ")"; return out;
  }
  void dump() const { print(std::cerr); } // TODO move to object file

  bool operator==(const Action & O) const {
    return O.pc == pc && O.ruleId == ruleId;
  }
};

// rule wrapper for RewritePairs
struct RewriteRule {
  int pairId;
  bool leftToRight;
  bool expanding; // cached result
  RewriteRule(int _pairId, bool _leftToRight, bool _expanding)
  : pairId(_pairId), leftToRight(_leftToRight), expanding(_expanding)
  {}
};

enum class BuiltinRules : int {
  PipeWrapOps = 0, // wrap all operands in pipes
  DropPipe = 1, // drop a pipe
  Clone = 2, // TODO clone operation for all pipe-users
  Fuse = 3, // TODO fuse with other instructions (erases all pipe annotated operations that yield the same result)
  Evaluate = 4, // TODO evaluate instruction and replace with constant
  Num = 5, // number of extra rules
  Invalid = -1, // error token
};

struct RuleBook {
  const RewritePairVec & rewritePairs;
  // [0, .. , rewriteRuleVec.size() - 1]  - rewrite rule range
  // [rewriteRuleVec.size(), .., += 2]  - pipe rules
  std::vector<RewriteRule> rewriteRuleVec;

  // start offset of builtin rules
  int getBuiltinStart() const { return rewriteRuleVec.size(); }
  int getBuiltinID(BuiltinRules rule) const { return getBuiltinStart() + (int) rule; }

  const ModelConfig & config;

  DataVec constVec; // recognized constants in the match rules
  std::map<data_t, int> constIndex; // constant number index

  bool getConstantIndex(data_t constVal, int & idx) const {
    auto it = constIndex.find(constVal);
    if (it == constIndex.end()) return false;
    idx = it->second;
    return true;
  }

  void collectConstants(const Program & prog) {
    for (auto & stat : prog.code) {
      if (!stat.isConstant()) continue;

      data_t constVal = stat.getValue();
      auto itIndex = constIndex.find(constVal);
      if (itIndex != constIndex.end()) continue;
      int nextNo = constVec.size();
      constIndex[constVal] = nextNo;
      constVec.push_back(constVal); // TODO redundant
    }
  }

  RuleBook(const ModelConfig & modelConfig, const RewritePairVec & _rewritePairs)
  : rewritePairs(_rewritePairs)
  , rewriteRuleVec()
  , config(modelConfig)
  {
  // mine rule constants
    for (auto & rule : rewritePairs) {
      collectConstants(rule.lhs);
      collectConstants(rule.rhs);
    }
    std::cerr << "found " << constVec.size() << " different constants in rule set!\n";

  // index rewrite rules
    for (int i = 0; i < rewritePairs.size(); ++i) {
      const auto & rewPair = rewritePairs[i];
      for (int j = 0; j < 2 - (rewPair.symmetric ? 1 : 0); ++j) {
        bool leftToRight = !(bool) j; // default to lhs -> rhs
        int ruleId = rewriteRuleVec.size();
        bool expanding = rewritePairs[i].isExpanding(leftToRight);
        rewriteRuleVec.emplace_back(i, leftToRight, expanding);
      }
    }

    std::cerr << "RuleBook::num_Rules = " << num_Rules() << "\n";
  }

  inline const RewriteRule * fetchRewriteRule(int ruleId) const {
    if (ruleId < rewriteRuleVec.size()) {
      return &rewriteRuleVec[ruleId];
    }
    return nullptr;
  }

  inline BuiltinRules fetchBuiltinRule(int ruleId) const {
    int extraId = ruleId - getBuiltinStart();
    assert(0 <= extraId && extraId < (int) BuiltinRules::Num);
    return (BuiltinRules) extraId;
  }

  decltype(rewriteRuleVec.cbegin()) begin() const { return rewriteRuleVec.begin(); }
  decltype(rewriteRuleVec.cend()) end() const { return rewriteRuleVec.end(); }

  // number of distinct actions
  int num_Rules() const { return rewriteRuleVec.size() + (int) BuiltinRules::Num; }

  // translate rewrite action to a flat moveID
  int toActionID(const Action rew) const {
    return num_Rules() * rew.pc + rew.ruleId;
  }

  // only support rewrite actions atm
  bool isRewriteAction(int actionId) const { return true; }

  // translate flat moveId to RewriteActions
  Action toRewriteAction(int actionId) const {
    // decode ruleEnumId/pc
    int ruleId = actionId % num_Rules();
    int pc = actionId / num_Rules();

    auto rew = Action{pc, ruleId};
    assert(toActionID(rew) == actionId);
    return rew;
  }

  // RewriteAction wrapping layer
  inline bool isExpanding(int ruleId) const {
    if (ruleId < getBuiltinStart()) {
      const RewriteRule* rewRule = fetchRewriteRule(ruleId);
      assert (rewRule);
      return rewRule->expanding; //rewritePairs[rewRule->pairId].isExpanding(rewRule->leftToRight);
    } else {
      switch (fetchBuiltinRule(ruleId)) {
        case BuiltinRules::PipeWrapOps: return true;
        case BuiltinRules::DropPipe: return false;
        case BuiltinRules::Clone: return true;
        case BuiltinRules::Fuse: return false;
        case BuiltinRules::Evaluate: return false;
        default:
          abort(); // TODO implement
      }
    }
  }

  inline bool matchRule(int ruleId, const Program & P, int pc, NodeVec &holes) const {
    NodeSet dummy;
    return matchRule_ext(ruleId, P, pc, holes, dummy);
  }


  inline bool matchRule_ext(int ruleId, const Program & P, int pc, NodeVec & holes, NodeSet & matchedNodes) const {
    const RewriteRule* rewRule = fetchRewriteRule(ruleId);
    if (rewRule) {
      if (pc == P.size() - 1) return false; // do not allow Ret (x) rewriting
      return rewritePairs[rewRule->pairId].match_ext(rewRule->leftToRight, P, pc, holes, matchedNodes);
    } else {
      switch (fetchBuiltinRule(ruleId)) {
        case BuiltinRules::PipeWrapOps:
          return P(pc).oc != OpCode::Pipe; // double piping not allowed
        case BuiltinRules::DropPipe:
          return P(pc).oc == OpCode::Pipe; // can only drop pipes
        case BuiltinRules::Clone: {
          bool foundFirst = false; // we can safely ignore the first user
          holes.clear();
          // use holes to keep track of pipe users
          for (int i = pc + 1; i < P.size(); ++i) {
            if ((P.code[i].oc == OpCode::Pipe) &&
               (P.code[i].getOperand(0) == pc))
            {
              if (foundFirst) {
                holes.push_back(i);
              }
              foundFirst = true;
            }
          }
          return !holes.empty();
        }

        case BuiltinRules::Fuse: {
          holes.clear();
          // look for identical operations (to the one @pc) that are #-tagged
          const int endPc = std::numeric_limits<int>::max();
          int minPc = endPc;
          for (int i = 0; i < pc; ++i) {
            if (P(i).oc != OpCode::Pipe) continue;
            int otherPc = P(i).getOperand(0);
            if (!IsStatement(otherPc)) continue; // arg match
            if (P(otherPc) != P(pc)) continue; // statement match
            if (otherPc < minPc) {
              // new min incumbent
              if (minPc != endPc) holes.push_back(minPc);
              minPc =  otherPc;
            } else if (otherPc != minPc) {
              holes.push_back(otherPc);
            }
          }

          // preserved slot at last position
          if (minPc != endPc) {
            holes.push_back(minPc);
          }

        } return !holes.empty();

        case BuiltinRules::Evaluate: {
          if (!P(pc).isOperator()) return false;
          for (int o = 0; o < P(pc).num_Operands(); ++o) {
            int opPc = P(pc).getOperand(o);
            if (!IsStatement(opPc) || P(opPc).oc != OpCode::Constant) return false;

          }
          return true;
        }

        default:
          abort(); // TODO implement
      }
    }
  }

  inline void transform(int ruleId, Program & P, int pc, NodeVec & holes) const {
    const RewriteRule* rewRule = fetchRewriteRule(ruleId);
    if (rewRule) {
      return rewritePairs[rewRule->pairId].rewrite(rewRule->leftToRight, P, pc, holes);
    } else {
      switch (fetchBuiltinRule(ruleId)) {
        case BuiltinRules::PipeWrapOps:
        {
#if 0
          std::cerr << "PIPE WRAP!\n";
          P.dump();
          std::cerr << pc << "\n";
#endif
          int numOps = P.code[pc].num_Operands();


          // wrap every operand in a pipe
          int matchPc = pc; // subject to movement
          for (int o = 0; o < numOps; ++o) {
            int origOpPc = P(matchPc).getOperand(o);
            int pipePc = std::max<>(0, origOpPc + 1); // insert below operand position (or at beginning for args)

            ReMap reMap;
            int newPc = P.make_space(pipePc, 2, reMap);
            // std::cerr << "after move at " << newPc << ":\n";
            matchPc++; // match root shifted
            // std::cerr << "match at " << matchPc << ":\n";
            // P.dump();

            // insert pipe at allocated position
            P(pipePc).oc = OpCode::Pipe;

            // tunnel data flow through pipe
            P(pipePc).setOperand(0, P(matchPc).getOperand(o));
            P(matchPc).setOperand(o, pipePc);
          }
#if 0
          std::cerr << "FINAL!\n";
          P.dump();
          abort();
#endif
        } return;

        case BuiltinRules::DropPipe:
        {
          assert(P(pc).oc == OpCode::Pipe);
          // replace all uses of pipe with pc
          int pipedVal = P(pc).getOperand(0);
          for (int i = pc + 1; i < P.size(); ++i) {
            for (int o = 0; o < P(i).num_Operands(); ++o) {
              if (P(i).getOperand(o) == pc) {
                P(i).setOperand(o, pipedVal);
              }
            }
          }
          P(pc).oc = OpCode::Nop;
          P.compact();
        } return;

        // clone the operation for all pipe users
        case BuiltinRules::Clone: {
          ReMap reMap;
          for (int i = 0; i < holes.size(); ++i) {
            int pipePc = holes[i];
            P.code[pipePc] = P.code[pc]; // clone operation for user
          }
        } return;

        // remove redundant operations
        case BuiltinRules::Fuse: {
          // std::cerr << "FUSE! " << pc << "\n";
          // P.dump();
          assert(!holes.empty());
          int lastSlot = holes.size() - 1;
          // preserved PC
          int minPc = holes[lastSlot]; // the one we keep
          NodeSet killSet; // all PCs that are remapped to @minPc
          for (int i = 0; i < lastSlot; ++i) {
            assert(0 <= holes[i] && holes[i] < P.size());
            assert((minPc != holes[i]) && "by match algo");
            killSet.insert(holes[i]);
            P(holes[i]).oc = OpCode::Nop; // fuse-and-erase
          }
          assert((minPc < pc) && "order violation");
          killSet.insert(pc);
          P(pc).oc = OpCode::Nop; //fuse-and-erase

          // replace all uses with the remaining instance @firstPc
          for (int j = minPc + 1; j < P.size(); ++j) {
            for (int o = 0; o < P(j).num_Operands(); ++o) {
              if (killSet.count(P(j).getOperand(o))) {
                P(j).setOperand(o, minPc);
              }
            }
          }


          // erase nops
          P.compact();
        } return;

        case BuiltinRules::Evaluate: {
          // fetch constant params
          DataVec dataVec; dataVec.reserve(P(pc).num_Operands());
          for (int o = 0; o < P(pc).num_Operands(); ++o) {
            int constPc = P(pc).getOperand(o);
            dataVec.push_back(P(constPc).getValue());
          }

          // evaluate and encode as constant
          data_t result = Evaluate(P(pc).oc, dataVec);
          P(pc) = build_const(result);
        } return;

        default:
          abort(); // TODO implement
      }
    }
  }

  inline int getLeftHandHoles(int ruleId) const {
    const RewriteRule* rewRule = fetchRewriteRule(ruleId);
    if (rewRule) {
      return rewritePairs[rewRule->pairId].getMatchProg(rewRule->leftToRight).num_Params();
    } else {
      return 0; // TODO
    }
  }

  inline int getRightHandHoles(int ruleId) const {
    const RewriteRule* rewRule = fetchRewriteRule(ruleId);
    if (rewRule) {
      return rewritePairs[rewRule->pairId].getRewriteProg(rewRule->leftToRight).num_Params();
    } else {
      return 0; // TODO
    }
  }
};

} // namespace apo

#endif // APO_ACTIONS_H

