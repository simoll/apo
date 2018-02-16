#ifndef APO_MODELCONFIG_H
#define APO_MODELCONFIG_H

#include <iostream>

namespace apo {

struct ModelConfig {
  // soft configuration (no model re-compile)
  int train_batch_size;
  int infer_batch_size;
  int batch_train_steps;

  // hard configuration (requires model rebuilding)
  int prog_length;
  int num_Params;
  int max_Rules;
  int max_OpCodes;

  ModelConfig(const std::string & configFile);
  std::ostream & print(std::ostream & out) const;
  void dump() const { print(std::cerr); }
};

} // namespace apo

#endif // APO_MODELCONFIG_H

