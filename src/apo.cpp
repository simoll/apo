#include "apo.h"
#include "ml.h"

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
    Program p = rpg.generate(stubLen);

    IF_VERBOSE {
      std::cerr << "Rand " << i << " ";
      p.dump();
    }
    DataVec refResult = Exec.run(p);
    IF_VERBOSE { std::cerr << "--> Result: "; Print(std::cerr, refResult); std::cerr << "\n"; }

    for (int m = 0; m < mutSteps; ++m) {
      mut.mutate(p, 1);
      IF_VERBOSE {
        std::cerr << "Mutated " << i << " at " << m << ": ";
        p.dump();
      }
      DataVec mutResult = Exec.run(p);
      IF_VERBOSE { std::cerr << "--> Result: "; Print(std::cerr, mutResult); std::cerr << "\n"; }
      assert(Equal(refResult, mutResult));
    }
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

int main(int argc, char ** argv) {

  test_tf();

  RunTests();

  TestGenerators();
  // return 0;

}
