#include "apo/apo.h"
#include "apo/shared.h"
#include "apo/ml.h"
#include "apo/program.h"
#include "apo/score.h"

#include "apo/timers.h"

#include <iostream>
#include <sstream>
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


  // parse command
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

    APO::Job job(taskFile, cpPrefix);
    APO apo;

    apo.train(job);
    Model::shutdown();
    return 0;

  } else if (cmd == "rand") { // "mcts <samples> <prog-name>'
    if (argc != 3) return -1;

    const std::string progFile(argv[2]);

    // parse program
    std::ifstream in(progFile);
    ProgramPtr P(Program::Parse(in));
    if (!P) {
      std::cerr << "Could not load program!\n";
      return -1;
    }

    if (!P->verify()) {
      std::cerr << "Program is not well formed!\n";
      return -2;
    }

    // initial prog.
    P->dump();
    int startScore = GetProgramScore(*P);

    // set-up engine
    APO apo;

    // optimize
    const int stepLimit = 256;
    ProgramVec progVec(1, P);
    double startOpt = get_wall_time();
    apo.optimize(progVec, APO::Strategy::Random, stepLimit);
    double endOpt = get_wall_time();

    // optimized prog.
    // P->dump();
    // auto endScore = GetProgramScore(*P);
    double optTime = (endOpt - startOpt);

    std::cerr << "Opt time= " << optTime << "s\n";
    //  std::cerr << "Start score " << startScore << ", end score: " << endScore << ". opt time= " << optTime << "s\n";

    Model::shutdown();
    return 0;
  } else if (cmd == "run") { // "run <model-cp> <prog-name>"
    if (argc != 4) return -1;

    const std::string cpFile(argv[2]);
    const std::string progFile(argv[3]);

    // parse program
    std::ifstream in(progFile);
    ProgramPtr P(Program::Parse(in));
    if (!P) {
      std::cerr << "Could not load program!\n";
      return -1;
    }

    if (!P->verify()) {
      std::cerr << "Program is not well formed!\n";
      return -2;
    }

    // initial prog.
    P->dump();
    int startScore = GetProgramScore(*P);

    // set-up engine
    APO apo;
    apo.loadCheckpoint(cpFile);

    // optimize
    const int stepLimit = 256;
    ProgramVec progVec(1, P);
    double startOpt = get_wall_time();
    apo.optimize(progVec, APO::Strategy::BestGreedy, stepLimit);
    double endOpt = get_wall_time();

    // optimized prog.
    P->dump();
#if 0
    // redundant
    P->dce();
    std::cerr << "after DCE:\n";
    P->dump();
#endif

    auto endScore = GetProgramScore(*P);
    double optTime = (endOpt - startOpt) / (double) CLOCKS_PER_SEC;

    std::cerr << "Start score " << startScore << ", end score: " << endScore << ". opt time= " << optTime << "s\n";

    Model::shutdown();
    return 0;
  }

  // help command
  std::cerr << argv[0] << " <command>\nAvailable commands:\n"
                           << "\ttrain <scenario.task>\n"
                           << "\trun <modelCheckpoint.cp> <program.p>\n";
  return 0;
}
