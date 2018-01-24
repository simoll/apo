#ifndef APO_RULEBOOK_H
#define APO_RULEBOOK_H

#include "apo/rewriting.h"
#include "apo/modelConfig.h"

namespace apo {

// generic action wrapper
struct Action {
  int pc;
  int ruleId;
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
  std::unordered_map<unsigned, int> rewriteRuleIndex;

  static int GetRewriteRuleKey(int pairId, bool leftMatch) {
    return (((unsigned) pairId) << 1) | (leftMatch ? 1 : 0);
  }

  inline int searchRewriteRuleIndex(int pairId, bool leftMatch) const {
    auto it = rewriteRuleIndex.find(GetRewriteRuleKey(pairId, leftMatch));
    assert(it != rewriteRuleIndex.end());
    return it->second;
  }

  ModelConfig config;

  RuleBook(ModelConfig modelConfig, const RewritePairVec & _rewritePairs)
  : rewritePairs(_rewritePairs)
  , config(modelConfig)
  {
    for (int i = 0; i < rewritePairs.size(); ++i) {
      const auto & rewPair = rewritePairs[i];
      for (int j = 0; j < 2 - (rewPair.symmetric ? 1 : 0); ++j) {
        bool leftToRight = !(bool) j; // default to lhs -> rhs
        int ruleId = rewriteRuleVec.size();
        rewriteRuleVec.emplace_back(i, leftToRight);
        rewriteRuleIndex[GetRewriteRuleKey(i, leftToRight)] = ruleId;
      }
    }
  }

  const RewriteRule & getRewriteRule(int ruleId) const { return rewriteRuleVec[ruleId]; }
  decltype(rewriteRuleVec.cbegin()) begin() const { return rewriteRuleVec.begin(); }
  decltype(rewriteRuleVec.cend()) end() const { return rewriteRuleVec.end(); }

  // number of distinct actions
  int num_Rules() const { return rewriteRuleVec.size(); }

  // translate rewrite action to a flat moveID
  int toActionID(const RewriteAction rew) const {
    int ruleId = searchRewriteRuleIndex(rew.pairId, rew.leftMatch);
    return num_Rules() * rew.pc + ruleId;
  }

  // only support rewrite actions atm
  bool isRewriteAction(int actionId) const { return true; }

  // translate flat moveId to RewriteActions
  RewriteAction toRewriteAction(int actionId) const {
    // decode ruleEnumId/pc
    int ruleId = actionId % num_Rules();
    int pc = actionId / num_Rules();

    const int pairId = rewriteRuleVec[ruleId].pairId;
    const bool leftMatch = rewriteRuleVec[ruleId].leftToRight;

    auto rew = RewriteAction{pc, pairId, leftMatch};
    assert(toActionID(rew) == actionId);
    return rew;
  }
};

} // namespace apo

#endif // APO_ACTIONS_H

