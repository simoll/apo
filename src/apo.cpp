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
  const int numRounds = 100;
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

  int genLen = model.prog_length - model.num_Params - 2;
  assert(genLen > 0 && "can not generate program within constraints");

// synthesize inputs
  const int numSamples = model.batch_size * 4;
  std::cout << "Generating " << numSamples << " programs..\n";

  for (int i = 0; i < numSamples; ++i) {
    auto * P = rpg.generate(genLen);
    assert(P->size() <= model.prog_length);
    progVec.emplace_back(P);
  }

// reference results
  ResultVec results;
  for (const auto & P : progVec) {
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
    assert(P->size() <= model.prog_length);
    progVec.emplace_back(P);
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
  for (const auto & P : progVec) {
    cloned.emplace_back(new Program(*P));
  }
  return cloned;
}

// optimize the given program @P using @model (or a uniform random rule application using @maxDist)
// maximal derivation length is @maxDist
// will return the sequence to the best-seen program (even if the model decides to go on)
struct MonteCarloOptimizer {
#define IF_DEBUG_MC if (false)


  // IF_DEBUG
  RuleVec & rules;
  Model & model;

  int maxGenLen;
  Mutator mut;

  int numOptRounds;

  MonteCarloOptimizer(RuleVec & _rules, Model & _model, int _numOptRounds)
  : rules(_rules)
  , model(_model)
  , maxGenLen(model.prog_length - model.num_Params - 1)
  , mut(rules, 0.1) // greedy shrinking mutator
  , numOptRounds(_numOptRounds)
  {}


  bool
  tryApplyModel(Program & P, Rewrite & rewrite, ResultDist & res, bool & signalsStop) {
  // sample a random rewrite at a random location (product of rule and target distributions)
    std::uniform_real_distribution<float> pRand(0, 1.0);

    int ruleEnumId = SampleCategoryDistribution(res.ruleDist, pRand(randGen()));

    int targetId;
    do {
      // TODO (efficiently!) normalize targetId to actual program length (apply cutoff after problem len)
      targetId = SampleCategoryDistribution(res.targetDist, pRand(randGen()));
    } while (targetId + 1 >= P.size());


    if (ruleEnumId == 0) {
      // magic STOP rule
      signalsStop = true;
      return true;
    }

  // translate to internal rule representation
    auto rew = Rewrite::fromModel(targetId, ruleEnumId);

  // Otw, rely on the mutator to do the job
    return mut.tryApply(P, rew.pc, rew.ruleId, rew.leftMatch);
  }

   struct Derivation {
     int bestScore;
     int shortestDerivation;

     Derivation(int _score, int _der)
     : bestScore(_score), shortestDerivation(_der) {}

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

     bool operator== (const Derivation& o) const { return (bestScore == o.bestScore) && (shortestDerivation == o.shortestDerivation); }
     bool operator!= (const Derivation& o) const { return !(*this == o); }
   };
   using DerivationVec = std::vector<Derivation>;

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  std::vector<Derivation>
  searchDerivations(const ProgramVec & progVec, const double pRandom, const int maxDist) {
    const int numSamples = progVec.size();
    std::uniform_real_distribution<float> ruleRand(0, 1);

    const bool useModel = pRandom != 1.0;

    std::vector<Derivation> states; // thread state
    for (int i = 0; i < progVec.size(); ++i) {
      states.emplace_back(*progVec[i]);
    }

#define IF_DEBUG_DER if (false)

    // pre-compute initial program distribution
    ResultDistVec initialProgDist;
    if (useModel) initialProgDist = model.infer_dist(progVec, true);

    // number of derivation walks
    for (int r = 0; r < numOptRounds; ++r) {

      // re-start from initial program
      ProgramVec roundProgs = Clone(progVec);

      ResultDistVec modelRewriteDist = initialProgDist;
      for (int derStep = 0; derStep < maxDist; ++derStep) {

        // use cached probabilities if possible
        if ((derStep > 0) && useModel) modelRewriteDist = model.infer_dist(roundProgs, true); // fail silently

        int frozen = 0;

        #pragma omp parallel for reduction(+:frozen)
        for (int t = 0; t < numSamples; ++t) {
          // freeze if derivation exceeds model
          if (roundProgs[t]->size() >= model.prog_length) {
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
            uniRule = !useModel || (ruleRand(randGen()) <= pRandom);

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

          if (signalsStop) continue;

        // derived program to large for model -> freeze
          if (roundProgs[t]->size() > model.prog_length) {
            ++frozen;
            continue;
          }

        // Otw, update incumbent
          // mutated program
          int currScore = GetProgramScore(*roundProgs[t]);
          Derivation thisDer(currScore, derStep + 1);
          if (thisDer.betterThan(states[t])) { states[t] = thisDer; }
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
  encodeBestDerivation(ResultDist & refResult, Derivation baseDer, const DerivationVec & derivations, const CompactedRewrites & rewrites, int startIdx, int progIdx) const {
  // find best-possible rewrite
    assert(startIdx < derivations.size());
    bool noBetterDerivation = true;
    Derivation bestDer = baseDer;
    for (int i = startIdx;
         i < rewrites.size() && (rewrites[i].first == progIdx);
         ++i)
    {
      const auto & der = derivations[i];
      if (der.betterThan(bestDer)) { noBetterDerivation = false; bestDer = der; }
    }

    if (noBetterDerivation) {
      // no way to improve over STOP
      refResult = model.createStopResult();
      return;
    }

    IF_DEBUG_MC { std::cerr << progIdx << " -> best "; bestDer.dump(); std::cerr << "\n"; }

  // activate all positions with best rewrites
    bool noBestDerivation = true;
    for (int i = startIdx;
         i < rewrites.size() && (rewrites[i].first == progIdx);
         ++i)
    {
      if (derivations[i] != bestDer) { continue; }
      noBestDerivation = false;
      const auto & rew = rewrites[i].second;
      int ruleEnumId = rew.getEnumId();
      assert(rew.pc < refResult.targetDist.size());
      refResult.targetDist[rew.pc] += 1.0;
      assert(ruleEnumId < refResult.ruleDist.size());
      refResult.ruleDist[ruleEnumId] += 1.0;

      IF_DEBUG_MC {
        std::cerr << "Prefix to best. pc=" << rew.pc << ", reid=" << ruleEnumId << "\n";
        rules[rew.ruleId].dump(rew.leftMatch);
      }
    }

    assert(!noBestDerivation);
  }

  void
  populateRefResults(ResultDistVec & refResults, const DerivationVec & derivations, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, const ProgramVec & progVec) const {
    int rewriteIdx = 0;
    int nextSampleWithRewrite = rewrites[rewriteIdx].first;
    for (int s = 0; s < progVec.size(); ++s) {
      // program without applicable rewrites
      if (s < nextSampleWithRewrite) {
        refResults.push_back(model.createStopResult());
        continue;
      } else {
        refResults.push_back(model.createEmptyResult());
      }

      // convert to a reference distribution
      encodeBestDerivation(refResults[s], Derivation(*progVec[s]), derivations, rewrites, rewriteIdx, s);

      // skip to next progam with rewrites
      for (;rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s; ++rewriteIdx) {}

      if (rewriteIdx >= rewrites.size()) {
        nextSampleWithRewrite = std::numeric_limits<int>::max(); // no more rewrites -> mark all remaining programs as STOP
      } else {
        nextSampleWithRewrite = rewrites[rewriteIdx].first; // program with applicable rewrite in sight
      }
    }
    assert(refResults.size() == progVec.size());

    // normalize distributions
    for (int s = 0; s < progVec.size(); ++s) {
      auto & result = refResults[s];
      result.normalize();

      IF_DEBUG_MC {
        std::cerr << "\n Sample " << s << ":\n";
        progVec[s]->dump();
        std::cerr << "Result ";
        if (result.isStop()) std::cerr << "STOP!\n";
        else result.dump();
      }
    }
  }

  // sample a target based on the reference distributions
  ProgramVec
  sampleActions(const ProgramVec & roundProgs, ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, bool & allStop) {
#define IF_DEBUG_SAMPLE if (false)
    allStop = true;
    std::uniform_real_distribution<float> pRand(0, 1.0);

    ProgramVec actionProgs;
    actionProgs.reserve(roundProgs.size());

    int rewriteIdx = 0;
    int nextSampleWithRewrite = rewrites.empty() ? std::numeric_limits<int>::max() : rewrites[rewriteIdx].first;
    for (int s = 0; s < refResults.size(); ++s) {
      IF_DEBUG_SAMPLE { std::cerr << "ACTION: " << actionProgs.size() << "\n"; }
      if (s < nextSampleWithRewrite) {
        // no rewrite available -> STOP
        actionProgs.push_back(roundProgs[s]);
        continue;
      }

      // Otw, sample an action
      const int numRetries = 100;
      bool hit = false;
      for (int t = 0; !hit && (t < numRetries); ++t) { // FIXME consider a greedy strategy
        int ruleEnumId = SampleCategoryDistribution(refResults[s].ruleDist, pRand(randGen()));
        int targetId = SampleCategoryDistribution(refResults[s].targetDist, pRand(randGen()));

        IF_DEBUG_SAMPLE { std::cerr << "PICK: " << targetId << " " << ruleEnumId << "\n"; }
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
              assert(i < nextProgs.size());
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
      }

      // could not hit
      if (!hit) {
        std::cerr << "---- Could not sample action!!! -----\n";
        roundProgs[s]->dump();
        refResults[s].dump();
        abort(); // this should never happen

        actionProgs.push_back(roundProgs[s]); // soft failure
      }

      // advance to next progam with rewrites
      for (;rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s; ++rewriteIdx) {}

      if (rewriteIdx >= rewrites.size()) {
        nextSampleWithRewrite = std::numeric_limits<int>::max(); // no more rewrites -> mark all remaining programs as STOP
      } else {
        nextSampleWithRewrite = rewrites[rewriteIdx].first; // program with applicable rewrite in sight
      }
    }

    assert(actionProgs.size() == roundProgs.size());
    return actionProgs;
#undef IF_DEBUG_SAMPLE
  }

#undef IF_DEBUG_MV
};




void
MonteCarloTest() {
  Model model("build/apo_graph.pb", "model.conf");
  auto rules = BuildRules();

// PARAMETERS (TODO factor out)
  // TODO factor out into MCOptimizer
// random program options
  const int numSamples = model.batch_size;
  const int minStubLen = 2; // minimal progrm stub len (excluding params and return)
  const int maxStubLen = 8; // maximal program stub len (excluding params and return)
  const int maxMutations = 3; // max number of program mutations
  const double pExpand = 0.7; // mutator expansion ratio

// mc search options
  const int mcDerivationSteps = 3; // number of derivations
  const int maxExplorationDepth = maxMutations + 1; // best-effort search depth
  const double pRandom = 1.0; // probability of ignoring the model for inference
  const int numOptRounds = 50; // number of optimization retries

// training
  const int batchTrainSteps = 10;

// number of simulation batches
  const int numGames = 10000;

  MonteCarloOptimizer montOpt(rules, model, numOptRounds);

// generate sample programs

  assert(minStubLen > 0 && "can not generate program within constraints");
  RPG rpg(rules, model.num_Params);

// generate randomly mutated programs

  Mutator expMut(rules, pExpand); // expanding rewriter
  std::uniform_int_distribution<int> mutRand(0, maxMutations);
  std::uniform_int_distribution<int> stubRand(minStubLen, maxStubLen);

  const int logInterval = 100;
  const int dotStep = logInterval / 10;

  for (int g = 0; g < numGames; ++g) {
    bool loggedRound = (g % logInterval == 0);
    if (loggedRound) {
      std::cerr << "\nRound " << g << ":\n";
    } else {
      if (g % dotStep == 0) { std::cerr << "."; }
    }

    ProgramVec progVec(numSamples, nullptr);

    // std::cout << "Generating " << numSamples << " programs..\n";
    #pragma omp parallel for
    for (int i = 0; i < numSamples; ++i) {
      int stubLen = stubRand(randGen());
      int mutSteps = mutRand(randGen());

      auto * P = rpg.generate(stubLen);
      assert(P->size() <= model.prog_length);
      expMut.mutate(*P, mutSteps); // mutate at least once
      progVec[i] = std::shared_ptr<Program>(P);

#if 0
      IF_DEBUG {
        std::cerr << "P " << i << ":\n";
        P->dump();
      }
#endif
    }

#if 0
    // sanity check
    auto derVec = montOpt.searchDerivations(progVec, pRandom, maxExplorationDepth);
    IF_DEBUG {
      std::cerr << "Best derivations:\n";
      for (int i = 1; i < derVec.size(); ++i) {
        derVec[i].dump();
        progVec[i]->dump();
        std::cerr << "\n";
      }
    }
#endif

  // explore all actions from current program

    for (int depth = 0; depth < mcDerivationSteps; ++depth) {

    // compute all one-step derivations
      std::vector<std::pair<int, Rewrite>> rewrites;
      ProgramVec nextProgs;
      const int preAllocFactor = 16;
      rewrites.reserve(preAllocFactor * progVec.size());
      nextProgs.reserve(preAllocFactor * progVec.size());

      // #pragma omp parallel for ordered
      for (int t = 0; t < progVec.size(); ++t) {
        for (int r = 0; r < rules.size(); ++r) {
          for (int j = 0; j < 2; ++j) {
            for (int pc = 0; pc + 1 < progVec[t]->size(); ++pc) { // skip return
              bool leftMatch = (bool) j;

              auto * clonedProg = new Program(*progVec[t]);
              if (!expMut.tryApply(*clonedProg, pc, r, leftMatch)) {
                // TODO clone after match (or render into copy)
                delete clonedProg;
                continue;
              }

              // compact list of programs resulting from a single action
              // #pragma omp ordered
              {
                nextProgs.emplace_back(clonedProg);
                rewrites.emplace_back(t, Rewrite{pc, r, leftMatch});
              }
            }
          }
        }
      }

    // best-effort search for optimal program
      auto derVec = montOpt.searchDerivations(nextProgs, pRandom, maxExplorationDepth);

#if 0
      IF_DEBUG {
        std::cerr << "Best derivations:\n";
        for (int i = 0; i < derVec.size(); ++i) {
          derVec[i].dump();
          nextProgs[i]->dump();
          std::cerr << "\n";
        }
      }
#endif

    // decode reference ResultDistVec from detected derivations
      ResultDistVec refResults;
      montOpt.populateRefResults(refResults, derVec, rewrites, nextProgs, progVec);

    // train model
      double loss = model.train_dist(progVec, refResults, batchTrainSteps, logInterval);

      if (loggedRound) {
        std::cerr << "Loss at " << depth << " : " << loss << "\n";
      }

    // pick an action per program and advance
      bool allStop;
      progVec = montOpt.sampleActions(progVec, refResults, rewrites, nextProgs, allStop);
      // FIXME delete unused program objects

      if (allStop) break; // early exit if no progress was made
    }

#if 0
    for (int i = 0; i < progVec.size(); ++i) {
      std::cerr << "Optimized program " << i << ":\n";
      progVec[i]->dump();
    }
#endif
  }
}

int main(int argc, char ** argv) {
  InitRandom();

  MonteCarloTest();
  return 0;

  RunTests();


  // TestGenerators();
  // return 0;
  //
  Model::shutdown();
}
