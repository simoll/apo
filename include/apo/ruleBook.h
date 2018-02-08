#ifndef APO_RULEBOOK_H
#define APO_RULEBOOK_H

#include "apo/rewriting.h"
#include "apo/modelConfig.h"

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
  RewriteRule(int _pairId, bool _leftToRight)
  : pairId(_pairId), leftToRight(_leftToRight)
  {}
};

struct RuleBook {
  const RewritePairVec & rewritePairs;
  // [0, .. , rewriteRuleVec.size() - 1]  - rewrite rule range
  // [rewriteRuleVec.size(), .., += 2]  - pipe rules
  std::vector<RewriteRule> rewriteRuleVec;

  // start offset of builtin rules
  int getBuiltinStart() const { return rewriteRuleVec.size(); }

  enum class BuiltinRules : int {
    PipeWrapOps = 0, // wrap all operand positions in pipes
    DropPipe = 1, // drop a pipe
    Num = 2, // number of extra rules
    Invalid = -1, // TODO maybe use as token
  };

  const ModelConfig & config;

  RuleBook(const ModelConfig & modelConfig, const RewritePairVec & _rewritePairs)
  : rewritePairs(_rewritePairs)
  , rewriteRuleVec()
  , config(modelConfig)
  {
    for (int i = 0; i < rewritePairs.size(); ++i) {
      const auto & rewPair = rewritePairs[i];
      for (int j = 0; j < 2 - (rewPair.symmetric ? 1 : 0); ++j) {
        bool leftToRight = !(bool) j; // default to lhs -> rhs
        int ruleId = rewriteRuleVec.size();
        rewriteRuleVec.emplace_back(i, leftToRight);
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
    const RewriteRule* rewRule = fetchRewriteRule(ruleId);
    if (rewRule) {
      return rewritePairs[rewRule->pairId].isExpanding(rewRule->leftToRight);
    } else {
      switch (fetchBuiltinRule(ruleId)) {
        case BuiltinRules::PipeWrapOps: return true;
        case BuiltinRules::DropPipe: return false;
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
          return P.code[pc].oc != OpCode::Pipe; // double piping not allowed
        case BuiltinRules::DropPipe:
          return P.code[pc].oc == OpCode::Pipe; // can only drop pipes
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
#endif
          std::cerr << pc << "\n";
          int numOps = P.code[pc].num_Operands();

          ReMap reMap;

          // make space for sufficiently many pipes (TODO don't double pipe)
          int newPc = P.make_space(pc, numOps + 1, reMap);
          auto & theStat = P.code[newPc]; // moved statment
#if 0
          std::cerr << "after make_space (for " << numOps << ") at " << newPc << ":\n";
          P.dump();
#endif

          // wrap every operand in a pipe
          for (int o = 0; o < numOps; ++o) {
            int i = newPc - numOps + o;
            auto & pipeStat = P.code[i];
            pipeStat.oc = OpCode::Pipe;
            pipeStat.setOperand(0, theStat.getOperand(o)); // create a new dedicated pipe for that operand
            theStat.setOperand(o, i); // use the piped operand instead
          }
#if 0
          std::cerr << "FINAL!\n";
          P.dump();
          abort();
#endif
        } return;

        case BuiltinRules::DropPipe:
        {
          assert(P.code[pc].oc == OpCode::Pipe);
          // replace all uses of pipe with pc
          int pipedVal = P.code[pc].getOperand(0);
          for (int i = pc + 1; i < P.size(); ++i) {
            for (int o = 0; o < P.code[i].num_Operands(); ++o) {
              if (P.code[i].getOperand(o) == pc) {
                P.code[i].setOperand(o, pipedVal);
              }
            }
          }
          P.code[pc].oc = OpCode::Nop;
          P.compact();
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

