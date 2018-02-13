#include "apo/apo.h"
#include "apo/shared.h"
#include "apo/ml.h"
#include "apo/program.h"

#include <iostream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <fstream>

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
    if (argc != 3) return -1;

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

  } else if (cmd == "parse") {
    if (argc != 3) return -1;

    // parse a program
    const std::string progFile(argv[2]);
    std::ifstream in(progFile);
    auto * P = Program::Parse(in);
    if (!P) return -1;
    P->dump();
    abort(); // TODO do something about this program
  }

  // help command
  std::cerr << argv[0] << " <command>\nAvailable commands:\n"
                           << "\ttrain <scenario.task>\n"
                           << "\tparse <program.p>\n";
  return 0;
}
