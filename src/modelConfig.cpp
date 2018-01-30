#include "apo/modelConfig.h"
#include "apo/parser.h"

namespace apo {

ModelConfig::ModelConfig(const std::string & configFile) {
  Parser confParser(configFile);
  // soft configuration
  train_batch_size = confParser.get_or_fail<int>("train_batch_size"); // training mini batch size
  infer_batch_size = confParser.get_or_fail<int>("infer_batch_size"); // inference mini batch size
  batch_train_steps = confParser.get_or_fail<int>("batch_train_steps");

  // hard configuration (requires model rebuilding)
  prog_length = confParser.get_or_fail<int>("prog_length");
  num_Params = confParser.get_or_fail<int>("num_Params");
  max_Rules = confParser.get_or_fail<int>("max_Rules");
}

std::ostream &
ModelConfig::print(std::ostream & out) const {
  out
    << "ModelConfig (apo).  train_batch_size=" << train_batch_size
               << ", infer_batch_size=" << infer_batch_size
               << ", batch_train_steps=" << batch_train_steps
               << ", prog_length=" << prog_length
               << ", max_Rules=" << max_Rules
               << ", num_Params=" << num_Params << ").";
  return out;
}

} // namespace apo