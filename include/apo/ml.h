#ifndef APO_ML_H
#define APO_ML_H

#include <string>
#include <vector>

#include "apo/program.h"
#include "apo/mutator.h"
#include "apo/extmath.h"

#include "apo/task.h"

// MetaGraph
#include <tensorflow/core/protobuf/meta_graph.pb.h>

namespace tf = tensorflow;

namespace tensorflow {
  class Session;
}


namespace apo {


struct ResultDist {
  float stopDist; // [0,1]
  CatDist actionDist; // [prog_length x num_Rules] -> [0,1]

  ResultDist() {}

  ResultDist(int numRules, int numTargets)
  : stopDist(0.0)
  , actionDist(numTargets * numRules, 0.0)
  {}

  void normalize();
  void print(std::ostream & out) const;
  void dump() const;
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

  // asynchronous training support
  TaskMutex modelMutex;

// graph definition

  // TODO read from shared config file
public:
  int infer_batch_size; // = 4;
  int train_batch_size; // = 4;
  int prog_length; // maximal program length
  int num_Params; // = 5;
  int max_Rules; //  maximal number of rules supported by model
  int num_Rules; // number of active rules
  const int max_Operands = 2;

  int batch_train_steps; // number of updates per training

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

  // wait until workerThread has finished (modelMutex is avilable)
  void flush();

public:
  Model(const std::string & saverPrefix, const std::string & configFile, int num_Rules);
  ~Model();

  void loadCheckpoint(const std::string & checkPointFile);
  void saveCheckpoint(const std::string & checkPointFile);

  // train model on a batch of programs (returns loss)
  // double train(const ProgramVec& progs, const ResultVec& results, int num_steps);

  // train model on a batch of programs (returns loss)
  struct Losses {
    double stopLoss;
    double targetLoss;
    double actionLoss;

    std::ostream& print(std::ostream & out) const;
  };

  Task train_dist(const ProgramVec& progs, const ResultDistVec& results, Losses * oLoss);

  Statistics query_stats();

  // most likely selection
#if 0
  ResultVec infer_likely(const ProgramVec& progs);
#endif

  // distribution over selections
  Task infer_dist(ResultDistVec & oResultDist, const ProgramVec& progs, size_t startIdx, size_t endIdx);

  // returns a plain STOP result
  ResultDist createStopResult() const;

  // create all-zero distributions
  ResultDist createEmptyResult() const;

  // set learning rate
  void setLearningRate(float v);

  int toActionID(const Rewrite rew) const {
    int ruleEnumId = rew.getEnumId();
    return rew.pc * num_Rules + ruleEnumId;
  }

  // translate flat actionId to Rewrite
  Rewrite toRewrite(int actionId) const {
    // decode ruleEnumId/pc
    int ruleEnumId = actionId % num_Rules;
    int pc = actionId / num_Rules;

    // decode leftMatch / rid
    int ruleId = ruleEnumId / 2;
    bool ruleLeftMatch = (ruleEnumId % 2 == 1);

    auto rew = Rewrite{pc, ruleId, ruleLeftMatch};
    assert(toActionID(rew) == actionId);
    return rew;
  }

  static void shutdown();
};

} // namespace apo

#endif // APO_ML_H

