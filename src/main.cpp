#include "apo/apo.h"
#include "apo/shared.h"
#include "apo/ml.h"
#include <iostream>

using namespace apo;

int main(int argc, char ** argv) {
  // initialize thread safe random number generators
  InitRandom();

  // pars command
  std::string cmd = "help";
  if (argc >= 2) {
    cmd = argv[1];
  }

  // help command
  if (cmd == "help") {
    std::cerr << argv[0] << " <command>\nAvailable commands:\n"
                             << "\ttrain <scenario.task>\n";
    return 0;
  }

  // train command
  if (cmd == "train") {
    const std::string cpPrefix = "cp/";
    std::string taskFile = argv[2];
    APO apo(taskFile, cpPrefix);

    const size_t numGames = 1000000;
    apo.train(numGames);
  }

  Model::shutdown();
}
