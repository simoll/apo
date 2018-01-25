#ifndef APO_MCTS_H
#define APO_MCTS_H

#include "apo/shared.h"
#include "apo/ml.h"

#include "apo/parser.h"
#include "apo/extmath.h"

#include "apo/program.h"
#include "apo/rpg.h"
#include "apo/mutator.h"
#include "apo/task.h"
#include "apo/ruleBook.h"

namespace apo {

using IntVec = std::vector<int>;

int GetProgramScore(const Program &P);

struct Derivation {
  int bestScore;
  int shortestDerivation;

  Derivation(int _score, int _der)
  : bestScore(_score), shortestDerivation(_der) {}

  Derivation(const Program & origP)
  : bestScore(GetProgramScore(origP))
  , shortestDerivation(0)
  {}

  Derivation()
  : bestScore(std::numeric_limits<int>::max())
  , shortestDerivation(0)
  {}

  std::ostream& print(std::ostream & out) const;

  void dump() const;

  bool betterThan(const Derivation & o) const;
  bool operator== (const Derivation& o) const;
  bool operator!= (const Derivation& o) const;
};
using DerivationVec = std::vector<Derivation>;

// optimize the given program @P using @model (or a uniform random rule application using @maxDist)
// maximal derivation length is @maxDist
// will return the sequence to the best-seen program (even if the model decides to go on)
struct MonteCarloOptimizer {
#define IF_DEBUG_MC if (false)

  // some statistics
  struct Stats {
    size_t sampleActionFailures; // failures to sample actions
    size_t invalidModelDists; // # detected invalid rule/dist distributions from model
    size_t derivationFailures; // # failed derivations
    size_t validModelDerivations; // # successful model-driven derivations

    Stats()
    : sampleActionFailures(0)
    , invalidModelDists(0)
    , derivationFailures(0)
    , validModelDerivations(0)
    {}

    std::ostream& print(std::ostream& out) const;
  };
  Stats stats;

  // IF_DEBUG
  const RuleBook & ruleBook;
  Model & model;

  int maxGenLen;
  Mutator mut;
  const float pSearchExpand = 0.1; // expanding rule probability
  const float stopThreshold = 0.8;

  const int sampleAttempts = 2; // number of attemps until tryApplyModel fails

  MonteCarloOptimizer(RuleBook & _ruleBook, Model & _model)
  : stats()
  , ruleBook(_ruleBook)
  , model(_model)
  , maxGenLen(model.config.prog_length - model.config.num_Params - 1)
  , mut(ruleBook.rewritePairs) // greedy shrinking mutator
  {}

  bool
  greedyApplyModel(Program & P, RewriteAction & rew, ResultDist & res, bool & signalsStop);

  bool
  tryApplyModel(Program & P, RewriteAction & rewrite, ResultDist & res, bool & signalsStop);

  // run greedy model based derivation
  struct GreedyResult {
    DerivationVec greedyVec; // stop at STOP
    DerivationVec bestVec; // best derivation within @maxDist
  };

  GreedyResult
  greedyDerivation(const ProgramVec & origProgVec, const IntVec & maxDistVec);

  // random trajectory based model (or uniform dist) sampling
  DerivationVec
  searchDerivations(const ProgramVec & progVec, const double pRandom, const IntVec & maxDistVec, const int numOptRounds, bool allowFallback);

  // optimized version for model-based seaerch
  DerivationVec
  searchDerivations_ModelDriven(const ProgramVec & progVec, const double pRandom, const IntVec & maxDist, const int numOptRounds, const bool useRandomFallback);

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  DerivationVec
  searchDerivations_Default(const ProgramVec & progVec, const IntVec & maxDist, const int numOptRounds);

  using CompactedRewrites = const std::vector<std::pair<int, RewriteAction>>;

  // convert detected derivations to refernce distributions
  void
  encodeBestDerivation(ResultDist & refResult, Derivation baseDer, const DerivationVec & derivations, const CompactedRewrites & rewrites, int startIdx, int progIdx) const;
  void
  populateRefResults(ResultDistVec & refResults, const DerivationVec & derivations, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, const ProgramVec & progVec) const;
  // sample a target based on the reference distributions (discards STOP programs)
  int sampleActions(ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, const IntVec & nextMaxDerVec, ProgramVec & oProgs, IntVec & oMaxDer);

#undef IF_DEBUG_MV
};

// compute a scaore for the sample derivations (assuming refDef contains reference derivations)
struct DerStats {
  double matched; // ref == model
  double longerDer; // ref.score == model.score BUT model.der > ref.der
  double betterScore; // model.score < ref.score
  double shorterDer; // ref.score == model.score AND model.der < ref.der

  double getClearedScore() const { return matched + longerDer + betterScore + shorterDer; }
  double getMisses() const { return 1.0 - getClearedScore(); }

  void print(std::ostream&out) const;
};

DerStats
ScoreDerivations(const DerivationVec & refDer, const DerivationVec & sampleDer);

} // namespace apo

#endif // APO_MCTS_H
