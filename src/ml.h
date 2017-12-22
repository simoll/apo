#ifndef APO_ML_H
#define APO_ML_H

#include <string>
#include <vector>

#include "program.h"

namespace tensorflow {
  class Session;
}

namespace apo {


  struct Result {
    int value; // number of adds in the program
  };

  using ProgramVec = std::vector<const Program*>;
  using ResultVec = std::vector<Result>;

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
    const int max_Operands = 2;

    int translateOperand(node_t idx) const;
    int encodeOperand(const Statement & stat, node_t opIdx) const;
    int encodeOpCode(const Statement & stat) const;

  public:
    Model(const std::string & fileName, const std::string & configFile);

    // train model on a batch of programs (returns loss)
    double train(const ProgramVec& progs, const ResultVec& results, int num_steps);

    ResultVec infer(const ProgramVec& progs);

    static void shutdown();
  };
}

#endif // APO_ML_H

