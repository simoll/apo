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
  std::vector<RewriteRule> rewriteRuleVec;

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

  inline const RewriteRule & getRewriteRule(int ruleId) const { return rewriteRuleVec[ruleId]; }
  decltype(rewriteRuleVec.cbegin()) begin() const { return rewriteRuleVec.begin(); }
  decltype(rewriteRuleVec.cend()) end() const { return rewriteRuleVec.end(); }

  // number of distinct actions
  int num_Rules() const { return rewriteRuleVec.size(); }

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
  inline bool isExpanding(const RewriteRule & rewRule) const {
    return rewritePairs[rewRule.pairId].isExpanding(rewRule.leftToRight);
  }

  inline bool matchRule(const RewriteRule & rewRule, const Program & P, int pc, NodeVec &holes) const {
    NodeSet dummy;
    return matchRule_ext(rewRule, P, pc, holes, dummy);
  }

  inline bool matchRule_ext(const RewriteRule & rewRule, const Program & P, int pc, NodeVec & holes, NodeSet & matchedNodes) const {
    return rewritePairs[rewRule.pairId].match_ext(rewRule.leftToRight, P, pc, holes, matchedNodes);
  }

  inline void transform(const RewriteRule & rewRule, Program & P, int pc, NodeVec & holes) const {
    return rewritePairs[rewRule.pairId].rewrite(rewRule.leftToRight, P, pc, holes);
  }

  inline int getLeftHandHoles(const RewriteRule & rewRule) const {
    return rewritePairs[rewRule.pairId].getMatchProg(rewRule.leftToRight).num_Params();
  }

  inline int getRightHandHoles(const RewriteRule & rewRule) const {
    return rewritePairs[rewRule.pairId].getRewriteProg(rewRule.leftToRight).num_Params();
  }
};

} // namespace apo

#endif // APO_ACTIONS_H

