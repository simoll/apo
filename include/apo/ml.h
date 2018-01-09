#ifndef APO_ML_H
#define APO_ML_H

#include <string>
#include <vector>

#include "apo/program.h"
#include "apo/mutator.h"
#include "apo/extmath.h"

namespace tensorflow {
  class Session;
}


namespace apo {


struct ResultDist {
  CatDist ruleDist;
  CatDist targetDist;
  ResultDist(int numRules, int numTargets)
  : ruleDist(numRules, 0.0)
  , targetDist(numTargets, 0.0)
  {}
  void normalize();
  void print(std::ostream & out) const;
  void dump() const;
  bool isStop() const;
};

struct Result {
  int rule;   // rule index (0 == STOP)
  int target; // target index (match root)
};

using ProgramVec = std::vector<Program*>;
using ResultVec = std::vector<Result>;
using ResultDistVec = std::vector<ResultDist>;

class Model {
// tensorflow state
  static tensorflow::Session * session;
  static bool initialized;
  // initialize tensorflow
  static int init_tflow();

// graph definition

  // TODO read from shared config file
public:
  int batch_size; // = 4;
  int max_Time; // = 4;
  int num_Params; // = 5;
  int num_Rules;
  const int max_Operands = 2;

  int translateOperand(node_t idx) const;
  int encodeOperand(const Statement & stat, node_t opIdx) const;
  int encodeOpCode(const Statement & stat) const;

  ResultDist createResultDist() { return ResultDist(num_Rules, max_Time + 1); }

public:
  Model(const std::string & fileName, const std::string & configFile);

  // train model on a batch of programs (returns loss)
  // double train(const ProgramVec& progs, const ResultVec& results, int num_steps);

  // train model on a batch of programs (returns loss)
  double train_dist(const ProgramVec& progs, const ResultDistVec& results, int num_steps);

  // most likely selection
  ResultVec infer_likely(const ProgramVec& progs);

  // distribution over selections
  ResultDistVec infer_dist(const ProgramVec& progs, bool failSilently=false);

  // returns a plain STOP result
  ResultDist createStopResult() const;

  // create all-zero distributions
  ResultDist createEmptyResult() const;

  static void shutdown();
};

} // namespace apo

#endif // APO_ML_H

