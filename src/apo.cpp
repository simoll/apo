#include "apo.h"
#include "ml.h"
#include "parser.h"

using namespace apo;



// RPG + Mutator fuzzing tests
void
TestGenerators() {
  std::cerr << "TEST: RPG + mutator fuzzing!\n";

  RuleVec rules = BuildRules();
  std::cerr << "Loaded " << rules.size() << " rules!\n";

  std::cerr << "Generating some random programs:\n";
  const int stubLen = 10;
  const int mutSteps = 100;

  const float pExpand = 0.05;
  const int numParams = 3;
  RPG rpg(rules, numParams);

  const int numSets = 3;
  RandExecutor Exec(numParams, numSets);
  Mutator mut(rules, pExpand);
  const int numRounds = 10000;
  for (int i = 0; i < numRounds; ++i) {
    Program * P = rpg.generate(stubLen);

    IF_VERBOSE {
      std::cerr << "Rand " << i << " ";
      P->dump();
    }
    DataVec refResult = Exec.run(*P);
    IF_VERBOSE { std::cerr << "--> Result: "; Print(std::cerr, refResult); std::cerr << "\n"; }

    for (int m = 0; m < mutSteps; ++m) {
      mut.mutate(*P, 1);
      IF_VERBOSE {
        std::cerr << "Mutated " << i << " at " << m << ": ";
        P->dump();
      }
      DataVec mutResult = Exec.run(*P);
      IF_VERBOSE { std::cerr << "--> Result: "; Print(std::cerr, mutResult); std::cerr << "\n"; }
      assert(Equal(refResult, mutResult));
    }

    delete P;
  }
}

void
RunTests() {
  auto rules = BuildRules();

  {
    NodeVec holes;
    std::cerr << "\nTEST: Shrinking re-write:\n";
    Program prog(2, {
        Statement(OpCode::Add, -1, -2),
        Statement(OpCode::Nop, 0, 0),
        Statement(OpCode::Sub, 0, -1),
        Statement(OpCode::Return, 2)
    });
    // define a simple program
    prog.compact();
    prog.dump();

    bool ok = rules[0].match(true, prog, 1, holes);
    assert(ok);

    // rewrite test
    rules[0].rewrite(true, prog, 1, holes);
    std::cerr << "after rewrite:\n";
    prog.dump();
  }

  {
    std::cerr << "\nTEST: Expanding re-write:\n";
    Program prog(2, {
        build_pipe(-1),
        Statement(OpCode::Mul, -1, 0),
        build_ret(0)
    });

    // define a simple program
    prog.compact();
    prog.dump();

    NodeVec holes;
    bool ok = rules[0].match(false, prog, 0, holes);
    assert(ok);

    assert(holes.size() == 1);
    holes.resize(2, 0);
    holes[0] = -3;
    holes[1] = -2;

    // rewrite test
    rules[0].rewrite(false, prog, 0, holes);
    std::cerr << "after rewrite:\n";
    prog.dump();
  }

  std::cerr << "END_OF_TESTS\n";
}

static
int
CountOpCode(const Program & P, OpCode oc) {
  int total = 0;
  for (const auto & stat : P.code) {
    total += (int) (stat.oc == oc);
  }
  return total;
}

void
ModelTest() {
  Model model("build/apo_graph.pb", "model.conf");
#if 0
#if 0
  ProgramVec progVec = {
    new Program(model.num_Params, {Statement(OpCode::Add, -1 , -2), build_ret(0)}),
    new Program(model.num_Params, {Statement(OpCode::Sub, -1 , -2), build_ret(0)}),
    new Program(model.num_Params, {Statement(OpCode::Xor, -1 , -2), Statement(OpCode::And, 0 , -1), build_ret(1)}),
    new Program(model.num_Params, {Statement(OpCode::And, -1 , -2), Statement(OpCode::Mul, -1, -2), Statement(OpCode::Add, 0, 1), build_ret(2)})
  };
#else
  ProgramVec progVec = {
    new Program(model.num_Params, {Statement(OpCode::Add, -1 , -2), Statement(OpCode::Sub, 0, -1), build_ret(0)}),
    new Program(model.num_Params, {Statement(OpCode::Sub, -1 , -2), Statement(OpCode::Sub, 0, -1), Statement(OpCode::Add, -3, 0)}),
    new Program(model.num_Params, {Statement(OpCode::Sub, -1 , -2), Statement(OpCode::Sub, 0 , -1), Statement(OpCode::Sub, 0, 1)}),
    new Program(model.num_Params, {Statement(OpCode::Add, -1 , -2), Statement(OpCode::Add, -1, -2), Statement(OpCode::Add, 0, 1)})
  };
#endif
#endif
  ProgramVec progVec;
  auto rules = BuildRules();
  RPG rpg(rules, model.num_Params);

  int genLen = model.max_Time - model.num_Params - 1;
  assert(genLen > 0 && "can not generate program within constraints");

// synthesize inputs
  const int numSamples = model.batch_size * 32;
  std::cout << "Generating " << numSamples << " programs..\n";

  for (int i = 0; i < numSamples; ++i) {
    auto * P = rpg.generate(genLen);
    assert(P->size() < model.max_Time);
    progVec.push_back(P);
  }

// reference results
  ResultVec results;
  for (const auto * P : progVec) {
    results.push_back(Result{CountOpCode(*P, OpCode::Add)});
  }

  const int numBatchSteps = 10;
  const int numEpochs = 100;

  for (int epoch = 0; epoch < numEpochs; ++epoch) {
    double avgLoss = model.train(progVec, results, numBatchSteps);
    std::cout << "Loss after epoch " << epoch << " " << avgLoss << "\n";
  }

  for (auto * P : progVec) delete P;
}

int main(int argc, char ** argv) {
  ModelTest();
  return 0;

  RunTests();


  // TestGenerators();
  // return 0;
  //
  Model::shutdown();
}
