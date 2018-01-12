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

// optimize the given program @P using @model (or a uniform random rule application using @maxDist)
// maximal derivation length is @maxDist
// will return the sequence to the best-seen program (even if the model decides to go on)
struct MonteCarloOptimizer {
#define IF_DEBUG_MC if (false)

  // some statistics
  struct Stats {
    size_t sampleActionFailures; // failures to sample actions
    size_t invalidModelDists; // # detected invalid rule/dist distributions from model
    size_t derivationFailures; // # failed derivations
    Stats()
    : sampleActionFailures(0)
    , invalidModelDists(0)
    , derivationFailures(0)
    {}

    std::ostream& print(std::ostream& out) const {
      out << "MCOpt::Stats {"
          << "sampleActionFailures " << sampleActionFailures
          << ", invalidModelDists " << invalidModelDists
          << ", derivationFailures " << derivationFailures << "}";
      return out;
    }
  };
  Stats stats;

  // IF_DEBUG
  RuleVec & rules;
  Model & model;

  int maxGenLen;
  Mutator mut;


  MonteCarloOptimizer(RuleVec & _rules, Model & _model)
  : stats()
  , rules(_rules)
  , model(_model)
  , maxGenLen(model.prog_length - model.num_Params - 1)
  , mut(rules, 0.1) // greedy shrinking mutator
  {}


  bool
  tryApplyModel(Program & P, Rewrite & rewrite, ResultDist & res, bool & signalsStop) {
  // sample a random rewrite at a random location (product of rule and target distributions)
    std::uniform_real_distribution<float> pRand(0, 1.0);

    int ruleEnumId = SampleCategoryDistribution(res.ruleDist, pRand(randGen()));

    size_t keepGoing = 10; // 10 sampling attempts
    int targetId;
    bool validPc;
    do {
      // TODO (efficiently!) normalize targetId to actual program length (apply cutoff after problem len)
      targetId = SampleCategoryDistribution(res.targetDist, pRand(randGen()));

      validPc = targetId + 1 < P.size(); // do not rewrite returns
    } while (keepGoing-- > 0 && !validPc);

  // failed to sample a valid rule application -> STOP
    if (!keepGoing) {
      signalsStop = true;
      return true;
    }

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

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  DerivationVec
  searchDerivations(const ProgramVec & progVec, const double pRandom, const int maxDist, const int numOptRounds) {
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
          const size_t failureLimit = 10;
          bool checkedDists = false; // sample distributions have been sanitized
          size_t failureCount = 0;
          while (!signalsStop && !success) {
            bool uniRule;
            uniRule = !useModel || (ruleRand(randGen()) <= pRandom);

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
              // sanitize distributions drawn from model
              if (!checkedDists &&
                  (!IsValidDistribution(modelRewriteDist[t].ruleDist) ||
                  !IsValidDistribution(modelRewriteDist[t].targetDist))) {
                stats.invalidModelDists++;
                signalsStop = true;
                break;
              }
              checkedDists = true; // model returned proper distributions for rule and target

              // use model to apply rule
              success = tryApplyModel(*roundProgs[t], rewrite, modelRewriteDist[t], signalsStop);
              if (!success) failureCount++;

              // avoid infinite loops by failing after @failureLimit
              signalsStop = (failureCount >= failureLimit);
            }
          }

          stats.derivationFailures += failureCount;

        // don't step over STOP
          if (signalsStop) {
            ++frozen;
            continue;
          }

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

      // could not hit -> STOP
      if (!hit) {
        stats.sampleActionFailures++;

        // std::cerr << "---- Could not sample action!!! -----\n";
        // roundProgs[s]->dump();
        // refResults[s].dump();
        // abort(); // this should never happen

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

// compute a scaore for the sample derivations (assuming refDef contains reference derivations)
double
ScoreDerivations(const DerivationVec & refDer, const DerivationVec & sampleDer) {
  size_t numHit = 0;
  for (int i = 0; i < refDer.size(); ++i) {
    if (sampleDer[i] == refDer[i] || sampleDer[i].betterThan(refDer[i])) {
      numHit++;
    }
  }
  return numHit / (double) refDer.size();
}


struct APO {
  Model model;
  RuleVec rules;
  MonteCarloOptimizer montOpt;
  RPG rpg;
  Mutator expMut;

  int minStubLen; //3; // minimal progrm stub len (excluding params and return)
  int maxStubLen; //4; // maximal program stub len (excluding params and return)
  int maxMutations;// 1; // max number of program mutations
  static constexpr double pExpand = 0.7; //0.7; // mutator expansion ratio

// mc search options
  int mcDerivationSteps; //1; // number of derivations
  int maxExplorationDepth; //maxMutations + 1; // best-effort search depth
  double pRandom; //1.0; // probability of ignoring the model for inference
  int numOptRounds; //50; // number of optimization retries

// training
  int numSamples;//
  int batchTrainSteps; // = 4;

// number of simulation batches
  const int numGames = 10000;

  APO(const std::string taskFile)
  : model("build/apo_graph.pb", "model.conf")
  , rules(BuildRules())
  , montOpt(rules, model)
  , rpg(rules, model.num_Params)
  , expMut(rules, pExpand)
  {
    std::cerr << "Loading task file " << taskFile << "\n";
    Parser task(taskFile);
  // random program options
    numSamples = model.max_batch_size;
    minStubLen = task.get_or_fail<int>("minStubLen"); //3; // minimal progrm stub len (excluding params and return)
    maxStubLen = task.get_or_fail<int>("maxStubLen"); //4; // maximal program stub len (excluding params and return)
    maxMutations = task.get_or_fail<int>("maxMutations");// 1; // max number of program mutations

  // mc search options
    mcDerivationSteps = task.get_or_fail<int>("mcDerivationSteps"); //1; // number of derivations
    maxExplorationDepth = task.get_or_fail<int>("maxExplorationDepth"); //maxMutations + 1; // best-effort search depth
    pRandom = task.get_or_fail<double>("pRandom"); //1.0; // probability of ignoring the model for inference
    numOptRounds = task.get_or_fail<int>("numOptRounds"); //50; // number of optimization retries

  // initialize thread safe random number generators
    InitRandom();
  }

  void
  generatePrograms(ProgramVec & progVec) {
    std::uniform_int_distribution<int> mutRand(0, maxMutations);
    std::uniform_int_distribution<int> stubRand(minStubLen, maxStubLen);

    for (int i = 0; i < progVec.size(); ++i) {
      std::shared_ptr<Program> P = nullptr;
      do {
        int stubLen = stubRand(randGen());
        int mutSteps = mutRand(randGen());
        P.reset(rpg.generate(stubLen));

        assert(P->size() <= model.prog_length);
        expMut.mutate(*P, mutSteps); // mutate at least once
      } while(P->size() > model.prog_length);

      progVec[i] = std::shared_ptr<Program>(P);
    }
  }

  void train() {
    const int numSamples = model.max_batch_size;
    const int numEvalSamples = model.max_batch_size * 4;

    std::cerr << "numSamples = " << numSamples << "\n"
              << "numEvalSamples = " << numEvalSamples << "\n"
              << "numGames = " << numGames << "\n";

  // training
    const int batchTrainSteps = 4;

  // number of simulation batches
    const int numGames = 10000;

    assert(minStubLen > 0 && "can not generate program within constraints");

    const int logInterval = 10;
    const int dotStep = logInterval / 10;

    std::cerr << "\n-- Training --";
    for (int g = 0; g < numGames; ++g) {
      bool loggedRound = (g % logInterval == 0);
      if (loggedRound) {
        auto stats = model.query_stats();
        std::cerr << "\n- Round " << g << " ("; stats.print(std::cerr); std::cerr << ") -\n";

      // print MCTS statistics
        montOpt.stats.print(std::cerr) << "\n";

      // evaluating current model
        ProgramVec evalProgs(numEvalSamples, nullptr);
        generatePrograms(evalProgs);

        // random sampling based (uniform sampling, nio model)
        auto refDerVec = montOpt.searchDerivations(evalProgs, 1.0, maxExplorationDepth, numOptRounds);

        // one shot (model based sampling)
        auto modelDerVec = montOpt.searchDerivations(evalProgs, 0.0, maxExplorationDepth, 1);

        double modelScore = ScoreDerivations(refDerVec, modelDerVec);
        std::cerr << "Model score: " << modelScore << "\n";

      } else {
        if (g % dotStep == 0) { std::cerr << "."; }
      }


  // Generating training programs
      ProgramVec progVec(numSamples, nullptr);
      generatePrograms(progVec);


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
        auto derVec = montOpt.searchDerivations(nextProgs, pRandom, maxExplorationDepth, numOptRounds);

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
        Model::Losses L;
        model.train_dist(progVec, refResults, batchTrainSteps, logInterval ? &L : nullptr);

        if (loggedRound) {
          std::cerr << "At " << depth << " : "; L.print(std::cerr) << "\n";
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

};

int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr << argv[0] << " <scenario.task>\n";
    return -1;
  }

  std::string taskFile = argv[1];
  APO apo(taskFile);

  apo.train();

  Model::shutdown();
}
