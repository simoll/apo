#include "apo/apo.h"
#include "apo/shared.h"
#include "apo/ml.h"
#include <iostream>
#include <sstream>
#include <ctime>
#include <iomanip>

using namespace apo;

std::string
GetCheckpointPath() {
  std::time_t T = std::time(nullptr);
  std::stringstream buffer;
  buffer << "cp/" << std::put_time(std::gmtime(&T), "%Y-%m-%d_%H:%M:%S");
  return buffer.str();
}

int main(int argc, char ** argv) {
  // initialize thread safe random number generators
  InitRandom();

  // pars command
  std::string cmd = "help";
  if (argc >= 2) {
    cmd = argv[1];
  }

  // train command
  if (cmd == "train") {
    const std::string taskFile = argv[2];

    // create time-stamped checkpoint (model state) path
    const std::string cpPrefix = GetCheckpointPath();
    std::stringstream ss;
    ss << "mkdir -p " << cpPrefix;
    system(ss.str().c_str());

    APO apo(taskFile, cpPrefix);

    apo.train();

    Model::shutdown();
    return 0;
  }

  // help command
  std::cerr << argv[0] << " <command>\nAvailable commands:\n"
                           << "\ttrain <scenario.task>\n";
  return 0;
}
