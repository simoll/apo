#ifndef APO_APO_H
#define APO_APO_H

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <cassert>
#include <set>
#include <initializer_list>
#include <functional>
#include <queue>

#include "apo/shared.h"
#include "apo/ml.h"

#include "apo/parser.h"
#include "apo/extmath.h"

#include "apo/program.h"
#include "apo/rpg.h"
#include "apo/mutator.h"
#include "apo/task.h"

#include <sstream>

#include <time.h>
#include <vector>
// #include "tensorflow/cc/client/client_session.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/cc/framework/tensor.h"

namespace apo {

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
  RuleVec & rules;
  Model & model;

  int maxGenLen;
  Mutator mut;
  const float stopThreshold = 0.8;

  const int sampleAttempts = 2; // number of attemps until tryApplyModel fails

  MonteCarloOptimizer(RuleVec & _rules, Model & _model)
  : stats()
  , rules(_rules)
  , model(_model)
  , maxGenLen(model.prog_length - model.num_Params - 1)
  , mut(rules, 0.1) // greedy shrinking mutator
  {}

  bool
  greedyApplyModel(Program & P, Rewrite & rew, ResultDist & res, bool & signalsStop);

  bool
  tryApplyModel(Program & P, Rewrite & rewrite, ResultDist & res, bool & signalsStop);

  // run greedy model based derivation
  struct GreedyResult {
    DerivationVec greedyVec; // stop at STOP
    DerivationVec bestVec; // best derivation within @maxDist
  };

  GreedyResult
  greedyDerivation(const ProgramVec & origProgVec, const int maxDist);

  // random trajectory based model (or uniform dist) sampling
  DerivationVec
  searchDerivations(const ProgramVec & progVec, const double pRandom, const int maxDist, const int numOptRounds, bool allowFallback);

  // optimized version for model-based seaerch
  DerivationVec
  searchDerivations_ModelDriven(const ProgramVec & progVec, const double pRandom, const int maxDist, const int numOptRounds, const bool useRandomFallback);

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  DerivationVec
  searchDerivations_Default(const ProgramVec & progVec, const int maxDist, const int numOptRounds);

  using CompactedRewrites = const std::vector<std::pair<int, Rewrite>>;

  // convert detected derivations to refernce distributions
  void
  encodeBestDerivation(ResultDist & refResult, Derivation baseDer, const DerivationVec & derivations, const CompactedRewrites & rewrites, int startIdx, int progIdx) const;
  void
  populateRefResults(ResultDistVec & refResults, const DerivationVec & derivations, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, const ProgramVec & progVec) const;
  // sample a target based on the reference distributions (discards STOP programs)
  int
  sampleActions(ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, ProgramVec & oProgs);

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


struct APO {
  RuleVec rules;
  Model model;
  std::string cpPrefix; // checkpoint prefix
  MonteCarloOptimizer montOpt;
  RPG rpg;
  Mutator expMut;

  std::string taskName; // name of task

  int minStubLen; //3; // minimal progrm stub len (excluding params and return)
  int maxStubLen; //4; // maximal program stub len (excluding params and return)
  int minMutations;// 1; // max number of program mutations
  int maxMutations;// 1; // max number of program mutations
  static constexpr double pExpand = 0.7; //0.7; // mutator expansion ratio

// mc search options
  int maxExplorationDepth; //maxMutations + 1; // best-effort search depth
  double pRandom; //1.0; // probability of ignoring the model for inference
  int numOptRounds; //50; // number of optimization retries
  int numEvalOptRounds; //50; // number of optimization retries

// eval round interval
  int logRate;
  size_t numRounds; // total training rounds
  size_t racketStartRound; // round when the racket should start (model based query)

  bool saveCheckpoints; // save model checkpoints at @logRate

// training
  int numSamples;//

// number of simulation batches
  APO(const std::string & taskFile, const std::string & _cpPrefix);

  void
  generatePrograms(ProgramVec & progVec, size_t startIdx, size_t endIdx);

  void train();
};


} // namespace apo

#endif // APO_APO_H
