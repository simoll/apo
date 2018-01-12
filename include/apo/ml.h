#ifndef APO_ML_H
#define APO_ML_H

#include <string>
#include <vector>

#include "apo/program.h"
#include "apo/mutator.h"
#include "apo/extmath.h"

// MetaGraph
#include <tensorflow/core/protobuf/meta_graph.pb.h>

namespace tf = tensorflow;

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

using ResultVec = std::vector<Result>;
using ResultDistVec = std::vector<ResultDist>;

class Model {
// tensorflow state
  static tensorflow::Session * session;
  static bool initialized;
  // initialize tensorflow
  static int init_tflow();

  tf::MetaGraphDef graph_def;

// graph definition

  // TODO read from shared config file
public:
  int max_batch_size; // = 4;
  int prog_length; // = 4;
  int num_Params; // = 5;
  int num_Rules;
  const int max_Operands = 2;

  int translateOperand(node_t idx) const;
  int encodeOperand(const Statement & stat, node_t opIdx) const;
  int encodeOpCode(const Statement & stat) const;

  ResultDist createResultDist() { return ResultDist(num_Rules, prog_length); }

  // internal learning statistics
  struct Statistics {
    size_t global_step;
    double learning_rate;
    void print(std::ostream & out) const;
  };

public:
  Model(const std::string & saverPrefix, const std::string & configFile);

  void loadCheckpoint(const std::string & checkPointFile);
  void saveCheckpoint(const std::string & checkPointFile);

  // train model on a batch of programs (returns loss)
  // double train(const ProgramVec& progs, const ResultVec& results, int num_steps);

  // train model on a batch of programs (returns loss)
  struct Losses {
    double ruleLoss;
    double targetLoss;

    std::ostream& print(std::ostream & out) const;
  };

  void train_dist(const ProgramVec& progs, const ResultDistVec& results, int num_steps, Losses * oLoss);

  Statistics query_stats();

  // most likely selection
  ResultVec infer_likely(const ProgramVec& progs);

  // distribution over selections
  ResultDistVec infer_dist(const ProgramVec& progs, bool failSilently=false);

  // returns a plain STOP result
  ResultDist createStopResult() const;

  // create all-zero distributions
  ResultDist createEmptyResult() const;

  // set learning rate
  void setLearningRate(float v);

  static void shutdown();
};

} // namespace apo

#endif // APO_ML_H

