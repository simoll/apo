#ifndef APO_ML_H
#define APO_ML_H

#include <string>
#include <vector>

#include "apo/program.h"
#include "apo/extmath.h"
#include "apo/modelConfig.h"
#include "apo/ruleBook.h"

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

  float& pAction(int i) { return actionDist[i]; }
  float pAction(int i) const { return actionDist[i]; }
  size_t size() const { return actionDist.size(); }

  void normalize();
  void print(std::ostream & out) const;
  void dump() const;
};

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

public:
  const ModelConfig & config;
  const RuleBook & ruleBook;

  const int max_Operands = 2;

  int translateOperand(node_t idx) const;
  int encodeOperand(const Statement & stat, node_t opIdx) const;
  int encodeOpCode(const Statement & stat) const;

  int num_Rules() const { return ruleBook.num_Rules(); }

  ResultDist createResultDist() { return ResultDist(num_Rules(), config.prog_length); }

  // internal learning statistics
  struct Statistics {
    size_t global_step;
    double learning_rate;
    void print(std::ostream & out) const;
  };

  // wait until workerThread has finished (modelMutex is avilable)
  void flush();

public:
  Model(const std::string & saverPrefix, const ModelConfig & _modelConfig, const RuleBook & _ruleBook);
  ~Model();

  void loadCheckpoint(const std::string & checkPointFile);
  void saveCheckpoint(const std::string & checkPointFile);

  // train model on a batch of programs (returns loss)
  struct Losses {
    double stopLoss;
    double targetLoss;
    double actionLoss;

    std::ostream& print(std::ostream & out) const;
  };

  ATTR_WARN_UNUSED
  Task train_dist(const ProgramVec& progs, const ResultDistVec& results, std::string towerName, Losses * oLoss);

  Statistics query_stats();

  // distribution over selections
  ATTR_WARN_UNUSED
  Task infer_dist(ResultDistVec & oResultDist, const ProgramVec& progs, size_t startIdx, size_t endIdx, std::string towerName);

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

