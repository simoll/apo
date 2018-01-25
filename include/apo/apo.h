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

#include "apo/shared.h"
#include "apo/ml.h"

#include "apo/parser.h"
#include "apo/extmath.h"

#include "apo/program.h"
#include "apo/rpg.h"
#include "apo/mutator.h"
#include "apo/task.h"
#include "apo/ruleBook.h"
#include "apo/mcts.h"

#include <sstream>

#include <time.h>
#include <vector>
// #include "tensorflow/cc/client/client_session.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/cc/framework/tensor.h"

namespace apo {

struct APO {
  ModelConfig modelConfig;
  RewritePairVec rewritePairs;
  RuleBook ruleBook;
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
  static constexpr double pGenExpand = 0.7; //0.7; // mutator expansion ratio

// mc search options
  int extraExplorationDepth; // number of derivation steps beyond applied number of mutations
  int maxExplorationDepth; // maximal exploration depth in any case
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
  generatePrograms(ProgramVec & progVec, IntVec & maxDistVec, size_t startIdx, size_t endIdx);

  void train();
};


} // namespace apo

#endif // APO_APO_H
