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

#include "apo/devices.h"
#include "apo/score.h"

#include <thread>

namespace apo {

using IntVec = std::vector<int>;

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

static DerivationVec
FilterBest(const DerivationVec & A, const DerivationVec & B) {
  DerivationVec res;
  res.reserve(A.size());
  for (int i = 0; i < A.size(); ++i) {
    if (A[i].betterThan(B[i])) {
      res.push_back(A[i]);
    } else {
      res.push_back(B[i]);
    }
  }
  return res;
}

// reasons to sample STOP
enum StopReason {
  Choice = 0, // STOP-ped by model choice
  InvalidDist = 1, // invalid action distribution
  DerivationFailure = 2, // failed to deriva a valid action (eventhough the action distribution is legal)
  NoPossibleAction = 3, // there is no applicable action
};

struct
SearchPerfStats {
  size_t numProgs; // workload
  double totalDerTime; // total time spent in searchDerivation
  double initialInferStall; // time spent in initial inference
  double totalInferStall; // total time stalled for pending inference

  void dump() const {
    std::cerr << "PerfStats { numProgs = " << numProgs << ", totalDerTime = " << totalDerTime << ", initialStall = " << initialInferStall << ", totalInferStall = " << totalInferStall << " }\n";
  }
};

// optimize the given program @P using @model (or a uniform random rule application using @maxDist)
// maximal derivation length is @maxDist
// will return the sequence to the best-seen program (even if the model decides to go on)
struct MonteCarloOptimizer {

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
  const float stopThreshold = 0.5; // FIXME this should be a parameter

  const int sampleAttempts = 2; // number of attemps until tryApplyModel fails

  MonteCarloOptimizer(RuleBook & _ruleBook, Model & _model)
  : stats()
  , ruleBook(_ruleBook)
  , model(_model)
  , maxGenLen(model.config.prog_length - model.config.num_Params - 1)
  , mut(ruleBook) // greedy shrinking mutator
  {}

  bool
  greedyApplyModel(Program & P, Action & rew, ResultDist & res, bool & signalsStop);

  bool
  tryApplyModel(Program & P, Action & rewrite, ResultDist & res, bool & signalsStop);

  // run greedy model based derivation
  struct GreedyResult {
    DerivationVec greedyVec; // stop at STOP
    DerivationVec bestVec; // best derivation within @maxDist

    GreedyResult(size_t numElems)
    : greedyVec(numElems)
    , bestVec(numElems)
    {}
  };

  // actual greedyDerivation func
  GreedyResult
  greedyDerivation(const ProgramVec & origProgVec, const IntVec & maxDistVec, const DeviceVec & devices);

  // helper func to distribute onto devices (mutates progs in progVec)
  void
  greedyDerivation(DerivationVec & oBestVec, DerivationVec & oStopVec, ProgramVec & progVec, const IntVec & maxDistVec, int startId, int endId, std::string towerName);

  // optimize the progarms in @progVec greedily
  void greedyOptimization(ProgramVec & oBestVec, ProgramVec & oStopVec, ProgramVec & progVec, const IntVec & maxDistVec, int startId, int endId, std::string towerName);

  // random trajectory based model (or uniform dist) sampling
  DerivationVec
  searchDerivations(const ProgramVec & progVec, const double pRandom, const IntVec & maxDistVec, const int numOptRounds, bool allowFallback, const DeviceVec & inferDevices, SearchPerfStats * oPerfStats = nullptr);

  // optimized version for model-based seaerch (improves results in @states)
  void
  searchDerivations_ModelDriven(DerivationVec & states, int startSlice, int endSlice, const ProgramVec & progVec, const double pRandom, const IntVec & maxDist, const int numOptRounds, const bool useRandomFallback, std::string towerName, SearchPerfStats * oPerfStats);

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  DerivationVec
  searchDerivations_Default(const ProgramVec & progVec, const IntVec & maxDist, const int numOptRounds);

  using CompactedRewrites = const std::vector<std::pair<int, Action>>;

  // convert detected derivations to refernce distributions
  void
  encodeBestDerivation(ResultDist & refResult, const DerivationVec & derivations, const CompactedRewrites & rewrites, const Derivation stopDer, int startIdx, int progIdx) const;
  ResultDistVec
  populateRefResults(const DerivationVec & derivations, const CompactedRewrites & rewrites, const ProgramVec & progVec) const;

  using ActionCallback = std::function<bool(int sampleIdx, int rewriteIdx)>;
  using StopCallback = std::function<bool(int sampleIdx, StopReason reason)>;

  // sample a target based on the reference distributions (discards STOP programs)
  void sampleActions(const ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, ActionCallback &&actionHandler, StopCallback &&stopHandler);

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
