#include "apo/modelConfig.h"
#include "apo/parser.h"

namespace apo {

ModelConfig::ModelConfig(const std::string & configFile, const std::string & trainFile) {
  Parser trainParser(trainFile);
  Parser confParser(configFile);

  // soft configuration
  train_batch_size = trainParser.get_or_fail<int>("train_batch_size"); // training mini batch size
  infer_batch_size = trainParser.get_or_fail<int>("infer_batch_size"); // inference mini batch size
  batch_train_steps = trainParser.get_or_fail<int>("batch_train_steps");
  self_organizing = trainParser.get_or_fail<int>("self_organizing");

  // hard configuration (requires model rebuilding)
  prog_length = confParser.get_or_fail<int>("prog_length");
  num_Params = confParser.get_or_fail<int>("num_Params");
  max_Rules = confParser.get_or_fail<int>("max_Rules");
  max_OpCodes = confParser.get_or_fail<int>("max_OpCodes");
}

std::ostream &
ModelConfig::print(std::ostream & out) const {
  out
    << "ModelConfig (apo).  train_batch_size=" << train_batch_size
               << ", infer_batch_size=" << infer_batch_size
               << ", batch_train_steps=" << batch_train_steps
               << ", prog_length=" << prog_length
               << ", max_Rules=" << max_Rules
               << ", num_Params=" << num_Params
               << ", self_organizing=" << self_organizing << ").";
  return out;
}

} // namespace apo
