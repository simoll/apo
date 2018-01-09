#include "apo/apo.h"
#include "apo/ml.h"
#include "apo/parser.h"
#include "apo/program.h"
#include "apo/mutator.h"
#include "apo/extmath.h"

#include <vector>

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

static
int
CountEdge(const Program & P, OpCode userOc, OpCode defOc, int opIdx) {
  int total = 0;
  for (const auto & stat : P.code) {
    if (stat.oc != userOc) continue;
    if (opIdx >= stat.num_Operands()) continue;
    // for (int opIdx = 0; opIdx < stat.num_Operands(); ++opIdx)

    {
      int i = stat.getOperand(opIdx);
      if (!IsStatement(i)) continue;
      const auto & defStat = P.code[i];
      total += (defStat.oc == defOc);
    }
  }

  return total;
}

static
int
CountMatches(const Program & P, const Rule & rule) {
  int total = 0;
  NodeVec holes;
  for (int pc = 0; pc < P.size(); ++pc) {
    if (rule.match(true, P, pc, holes)) total++;
  }
  return total;
}

static
int
MostLikely(const Program & P, const RuleVec & rules) {
  int likelyRule = 0; // STOP rule
  int likelyHits = 0;
  for (int r = 0; r < rules.size(); ++r) {
    const auto & R = rules[r];
    // results.push_back(Result{CountOpCode(*P, OpCode::Add)}); // WORKS
    // results.push_back(Result{CountEdge(*P, OpCode::Mul, OpCode::Sub, 0)}); // number of Subs in operand position "0" of any Mul // WORKS
    int numMatched = CountMatches(P, R);
    if (numMatched > likelyHits) {
      likelyHits = numMatched;
      likelyRule = r + 1; // skip virtual STOP rule
    }
  }
  return likelyRule;
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
  const int numSamples = model.batch_size * 64;
  std::cout << "Generating " << numSamples << " programs..\n";

  for (int i = 0; i < numSamples; ++i) {
    auto * P = rpg.generate(genLen);
    assert(P->size() < model.max_Time);
    progVec.push_back(P);
  }

// reference results
  ResultVec results;
  for (const auto * P : progVec) {
    int likelyRule = MostLikely(*P, rules);
    results.push_back(Result{likelyRule, 0});
  }

#if 0
  const auto & R = rules[0];
  // TODO most likely rule prediction
  double randFrac = 1.0 - (t / (double) numSamples);
  std::cout << "Found " << t << " matches for rule (0-baseline " << randFrac << "):\n";
  R.dump();
#endif

// Training
  const int numBatchSteps = 4;
  const int numEpochs = 10000;

#if 0
  std::cout << "Training:\n";
  for (int epoch = 0; epoch < numEpochs; ++epoch) {
    double fracCorrect = model.train(progVec, results, numBatchSteps);
    std::cout << "correct fraction after epoch " << epoch << " :  " << fracCorrect << "\n";
    if (fracCorrect >= .9999) {
      std::cout << "Training complete!\n";
      break;
    }
  }

  for (auto * P : progVec) delete P;
#endif

// Validation
  progVec.clear();
  results.clear();
  std::cout << "Re-generating " << numSamples << " programs..\n";

  for (int i = 0; i < numSamples; ++i) {
    auto * P = rpg.generate(genLen);
    assert(P->size() < model.max_Time);
    progVec.push_back(P);
  }

#if 0
  ResultVec predicted = model.infer_likely(progVec);

  int hits = 0;
  for (int i = 0; i < numSamples; ++i) {
    int refResult = MostLikely(*progVec[i], rules);
    hits += (predicted[i].rule == refResult);
  }
  double pCorrect = hits / (double) numSamples;

  std::cout << "Validated: " << pCorrect << "\n";
#else
  ResultDistVec predicted = model.infer_dist(progVec);

  for (int i = 0; i < numSamples; ++i) {
    predicted[i].dump();
  }
  abort();
#endif
}

// optimization recipe

int
GetProgramScore(const Program & P) {
  return P.size();
}

static
ProgramVec
Clone(const ProgramVec & progVec) {
  ProgramVec cloned;
  cloned.reserve(progVec.size());
  for (const auto * P : progVec) {
    cloned.push_back(new Program(*P));
  }
  return cloned;
}

// optimize the given program @P using @model (or a uniform random rule application using @maxDist)
// maximal derivation length is @maxDist
// will return the sequence to the best-seen program (even if the model decides to go on)
struct MonteCarloOptimizer {
#define IF_DEBUG_MC IF_DEBUG
  RuleVec & rules;
  Model & model;

  int maxGenLen;
  Mutator mut;

  MonteCarloOptimizer(RuleVec & _rules, Model & _model)
  : rules(_rules)
  , model(_model)
  , maxGenLen(model.max_Time - model.num_Params - 1)
  , mut(rules, 0.1) // greedy shrinking mutator
  {}


  bool
  tryApplyModel(Program & P, Rewrite & rewrite, ResultDist & res, bool signalsStop) {
  // sample a random rewrite at a random location (product of rule and target distributions)
    std::uniform_real_distribution<float> pRand(0, 1.0);

    int ruleId = SampleCategoryDistribution(res.ruleDist, pRand(randGen));
    int targetId = SampleCategoryDistribution(res.targetDist, pRand(randGen));

    if (ruleId == 0) {
      // magic STOP rule
      signalsStop = true;
      return true;
    }

  // translate to internal rule representation
    int ruleListIdx = (ruleId - 1) / 2;
    bool matchLeft = (ruleId - 1) % 2 == 0;

  // Otw, rely on the mutator to do the job
    return mut.tryApply(P, targetId, ruleListIdx, matchLeft);
  }

   struct Derivation {
     int bestScore;
     int shortestDerivation;

     Derivation(const Program & origP)
     : bestScore(GetProgramScore(origP))
     , shortestDerivation(0)
     {}

     void dump() const {
       std::cerr << "Derivation (bestScore=" << bestScore << ", dist=" << shortestDerivation << ")";
     }

     bool betterThan(const Derivation & o) const {
       if (bestScore < o.bestScore) {
         return true;
       } else if (bestScore == o.bestScore && (shortestDerivation < o.shortestDerivation)) {
         return true;
       }
       return false;
     }

     bool operator== (const Derivation& o) const { return bestScore == o.bestScore && shortestDerivation == o.shortestDerivation; }
     bool operator!= (const Derivation& o) const { return !(*this == o); }
   };
   using DerivationVec = std::vector<Derivation>;

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  std::vector<Derivation>
  searchDerivations(const ProgramVec & progVec, double pRandom, int maxDist) {
    const int numSamples = progVec.size();
    std::uniform_real_distribution<float> ruleRand(0, 1);

    std::vector<Derivation> states; // thread state
    for (int i = 0; i < progVec.size(); ++i) {
      states.emplace_back(*progVec[i]);
    }

#define IF_DEBUG_DER if (false)

    // pre-compute initial program distribution
    ResultDistVec initialProgDist = model.infer_dist(progVec, true);

    // number of derivation walks
    const int numRounds = 100;
    for (int r = 0; r < numRounds; ++r) {

      // re-start from initial program
      ProgramVec roundProgs = Clone(progVec);

      ResultDistVec modelRewriteDist = initialProgDist;
      for (int derStep = 0; derStep < maxDist; ++derStep) {

        // use cached probabilities if possible
        if (derStep > 0) modelRewriteDist = model.infer_dist(roundProgs, true); // fail silently

        int frozen = 0;

        for (int t = 0; t < numSamples; ++t) {
          // freeze if derivation exceeds model
          if (roundProgs[t]->size() >= model.max_Time) {
            ++frozen;
            continue;
          }

          IF_DEBUG_DER if (derStep == 0) {
            std::cerr << "Initial prog " << t << ":\n";
            roundProgs[t]->dump();
          }

        // pick & apply a rewrite
          Rewrite rewrite;
          bool success = false;
          bool signalsStop = false;

        // loop until rewrite succeeds (or stop)
          while (!signalsStop && !success) {
            bool uniRule;
            uniRule = ruleRand(randGen) <= pRandom;

            bool signalsStop = false;

            if (uniRule) {
              // uniform random rewrite
              rewrite = mut.mutate(*roundProgs[t], 1);
              IF_DEBUG_DER {
                std::cerr << "after random rewrite!\n";
                roundProgs[t]->dump();
              }
              success = true; // mutation always succeeeds
              signalsStop = false;
            } else {
              // learned rewrite distribution
              success = tryApplyModel(*roundProgs[t], rewrite, modelRewriteDist[t], signalsStop);
            }
          }

          if (signalsStop) break;

        // derived program to large for model -> freeze
          if (roundProgs[t]->size() >= model.max_Time) {
            ++frozen;
            continue;
          }

        // Otw, update incumbent
          // mutated program
          int currScore = GetProgramScore(*roundProgs[t]);
          if (states[t].bestScore < currScore) continue;

          if ((states[t].bestScore > currScore) || // found a better candidate program
              (states[t].shortestDerivation > derStep) // reached known best program with shorter derivation
          ) {
            states[t].bestScore = currScore;
            states[t].shortestDerivation = derStep + 1;
          }
        }

        if (frozen == numSamples) {
          break; // all frozen -> early exit
        }
      }
    }

#undef IF_DEBUG_DER
    return states;
  }

  using CompactedRewrites = const std::vector<std::pair<int, Rewrite>>;

  // convert detected derivations to refernce distributions
  void
  encodeBestDerivation(ResultDistVec & refResultVec, const DerivationVec & derivations, const CompactedRewrites & rewrites, int startIdx, int progIdx) const {
  // find best-possible rewrite
    Derivation bestDer = derivations[startIdx];
    for (int i = startIdx + 1;
         i < rewrites.size() && (rewrites[i].first == progIdx);
         ++i)
    {
      const auto & der = derivations[i];
      if (der.betterThan(bestDer)) { bestDer = der; }
    }

  // activate all positions with best rewrites
    for (int i = startIdx + 1;
         i < rewrites.size() && (rewrites[i].first == progIdx);
         ++i)
    {
      if (derivations[i] != bestDer) { continue; }
      const auto & rew = rewrites[i].second;
      int ruleEnumId = rew.getEnumId();
      refResultVec[progIdx].targetDist[rew.pc] += 1.0;
      refResultVec[progIdx].ruleDist[ruleEnumId] += 1.0;
    }
  }

  void
  populateRefResults(ResultDistVec & refResults, const DerivationVec & derivations, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, const ProgramVec & progVec) const {
    int rewriteIdx = 0;
    int nextSampleWithRewrite = rewrites[rewriteIdx].first;
    for (int s = 0; s < progVec.size(); ++s) {
      // program without applicable rewrites
      if (s < nextSampleWithRewrite) {
        refResults[s] = model.createStopResult();
        continue;
      }

      // convert to a reference distribution
      encodeBestDerivation(refResults, derivations, rewrites, rewriteIdx, s);

      // advance to next progam with rewrites
      ++rewriteIdx;
      if (rewriteIdx >= rewrites.size()) {
        nextSampleWithRewrite = std::numeric_limits<int>::max(); // no more rewrites -> mark all remaining programs as STOP
      } else {
        nextSampleWithRewrite = rewrites[rewriteIdx].first; // program with applicable rewrite in sight
      }
    }

    // normalize distributions
    for (int s = 0; s < progVec.size(); ++s) {
      auto & result = refResults[s];
      result.normalize();
      IF_DEBUG_MC {
        std::cerr << "\n Sample " << s << ":\n";
        progVec[s]->dump();
        result.dump();
      }
    }
  }

  ProgramVec
  sampleActions(const ProgramVec & roundProgs, ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, bool & allStop) {
    allStop = true;
    std::uniform_real_distribution<float> pRand(0, 1.0);

    ProgramVec actionProgs;

    int rewriteIdx = 0;
    int nextSampleWithRewrite = rewrites[rewriteIdx].first;
    for (int s = 0; s < refResults.size(); ++s) {
      if (s < nextSampleWithRewrite) {
        // no rewrite available -> STOP
        actionProgs.push_back(roundProgs[s]);
        continue;
      }

      // Otw, sample an action
      bool hit = false;
      do {
        int ruleEnumId = SampleCategoryDistribution(refResults[s].ruleDist, pRand(randGen));
        int targetId = SampleCategoryDistribution(refResults[s].targetDist, pRand(randGen));

        // try to apply the action
        if (ruleEnumId > 0) {
          // scan through legal actions until hit
          for (int i = rewriteIdx;
              i < rewrites.size() && rewrites[i].first == s;
              ++i)
          {
            if ((rewrites[i].second.pc == targetId) &&
                (rewrites[i].second.getEnumId() == ruleEnumId)
            ) {
              actionProgs.push_back(nextProgs[i]);
              hit = true;
              break;
            }
          }

          // proper action
          allStop = false;
        } else {
          // STOP action
          actionProgs.push_back(roundProgs[s]);
          hit = true;
          break;
        }
      } while (!hit);

      // advance to next progam with rewrites
      ++rewriteIdx;
      if (rewriteIdx >= rewrites.size()) {
        nextSampleWithRewrite = std::numeric_limits<int>::max(); // no more rewrites -> mark all remaining programs as STOP
      } else {
        nextSampleWithRewrite = rewrites[rewriteIdx].first; // program with applicable rewrite in sight
      }
    }

    assert(actionProgs.size() == nextProgs.size());
    return actionProgs;
  }

#undef IF_DEBUG_MV
};




void
MonteCarloTest() {
  Model model("build/apo_graph.pb", "model.conf");
  auto rules = BuildRules();
  MonteCarloOptimizer montOpt(rules, model);

// generate sample programs
  ProgramVec progVec;

  int genLen = 2; //model.max_Time - model.num_Params - 1;
  assert(genLen > 0 && "can not generate program within constraints");
  RPG rpg(rules, model.num_Params);

// generate randomly mutated programs
  const int numSamples = model.batch_size;
  std::cout << "Generating " << numSamples << " programs..\n";

  const int maxMutations = 3;
  Mutator expMut(rules, .7); // expanding rewriter
  for (int i = 0; i < numSamples; ++i) {
    auto * P = rpg.generate(genLen);
    assert(P->size() < model.max_Time);
    progVec.push_back(P);

    std::uniform_int_distribution<int> mutRand(0, maxMutations); // FIXME geometric distribution
    int mutSteps = mutRand(randGen);
    expMut.mutate(*P, mutSteps); // mutate at least once
  }

// explore all actions from current program
  // TODO factor out into MCOptimizer
  const int mcDerivationSteps = 3;
  const int maxExplorationDepth = 3;

  for (int depth = 0; depth < mcDerivationSteps; ++depth) {

  // compute all one-step derivations
    std::vector<std::pair<int, Rewrite>> rewrites;
    ProgramVec nextProgs;

    #pragma omp parallel
    for (int t = 0; t < progVec.size(); ++t) {
      for (int r = 0; r < rules.size(); ++r) {
        for (int j = 0; j < 2; ++j) {
          for (int pc = 0; pc < progVec[t]->size(); ++pc) {
            bool leftMatch = (bool) j;

            auto * clonedProg = new Program(*progVec[t]);
            if (!expMut.tryApply(*clonedProg, pc, r, leftMatch)) {
              // TODO clone after match (or render into copy)
              delete clonedProg;
              continue;
            }

            // compact list of programs resulting from a single action
            #pragma omp ordered
            {
              nextProgs.push_back(clonedProg);
              rewrites.emplace_back(t, Rewrite{pc, r, leftMatch});
            }
          }
        }
      }
    }

  // best-effort search for optimal program
    const double pRandom = 1.0; // probability of ignoring the model for inference
    auto derVec = montOpt.searchDerivations(nextProgs, pRandom, maxExplorationDepth);

  // decode reference ResultDistVec from detected derivations
    ResultDistVec refResults;
    montOpt.populateRefResults(refResults, derVec, rewrites, nextProgs, progVec);

  // train model
    const int batchSteps = 10;
    double loss = model.train_dist(progVec, refResults, batchSteps);
    std::cerr << "Loss at " << depth << "\n";

  // pick an action per program and advance
    bool allStop;
    progVec = montOpt.sampleActions(progVec, refResults, rewrites, nextProgs, allStop);

    if (allStop) break; // early exit if no progress was made
  }
}

int main(int argc, char ** argv) {
  MonteCarloTest();
  return 0;

  RunTests();


  // TestGenerators();
  // return 0;
  //
  Model::shutdown();
}
