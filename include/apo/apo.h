#ifndef APO_APO_H
#define APO_APO_H

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <cassert>
#include <set>
#include <initializer_list>
#include <functional>
#include <queue>

#include "apo/shared.h"
#include "apo/ml.h"

#include "apo/parser.h"
#include "apo/extmath.h"

#include "apo/program.h"
#include "apo/rpg.h"
#include "apo/mutator.h"

#include <sstream>

#include <vector>
// #include "tensorflow/cc/client/client_session.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/cc/framework/tensor.h"

namespace apo {

int
GetProgramScore(const Program & P) {
  return P.size();
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

  std::ostream& print(std::ostream & out) const {
    out << "Derivation (bestScore=" << bestScore << ", dist=" << shortestDerivation << ")"; return out;
  }

  void dump() const { print(std::cerr); }

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

    // should we stop?
    if (pRand(randGen()) <= res.stopDist) {
      signalsStop = true;
      return true;
    }

    size_t keepGoing = 10; // 10 sampling attempts
    int targetId, ruleId;
    bool validPc;
    Rewrite rew;
    do {
      // which rule to apply?
      int actionId = SampleCategoryDistribution(res.actionDist, pRand(randGen()));
      rew = model.toRewrite(actionId);

      // translate to internal rule representation
      validPc = rew.pc + 1 < P.size(); // do not rewrite returns
    } while (keepGoing-- > 0 && !validPc);

  // failed to sample a valid rule application -> STOP
    if (!validPc) {
      signalsStop = true;
      return true;
    }

  // Otw, rely on the mutator to do the job
    return mut.tryApply(P, rew.pc, rew.ruleId, rew.leftMatch);
  }

  DerivationVec
  searchDerivations(const ProgramVec & progVec, const double pRandom, const int maxDist, const int numOptRounds) {
    if (pRandom < 1.0) return searchDerivations_ModelDriven(progVec, pRandom, maxDist, numOptRounds);
    else return searchDerivations_Default(progVec, pRandom, maxDist, numOptRounds);
  }

  // optimized version for model-based seaerch
  DerivationVec
  searchDerivations_ModelDriven(const ProgramVec & progVec, const double pRandom, const int maxDist, const int numOptRounds) {
    assert(pRandom < 1.0 && "use _Generic implementation instead");

    const int numSamples = progVec.size();
    std::uniform_real_distribution<float> ruleRand(0, 1);

    // start with STOP derivation
    std::vector<Derivation> states;
    for (int i = 0; i < progVec.size(); ++i) {
      states.emplace_back(*progVec[i]);
    }

#define IF_DEBUG_DER if (false)

    // pre-compute initial program distribution
    ResultDistVec initialProgDist(progVec.size());
    std::thread handle = model.infer_dist(initialProgDist, progVec, 0, progVec.size());
    handle.join();

    // number of derivation walks
    for (int r = 0; r < numOptRounds; ++r) {

      // re-start from initial program
      ProgramVec roundProgs = Clone(progVec);

      ResultDistVec modelRewriteDist = initialProgDist;

      for (int derStep = 0; derStep < maxDist; ++derStep) {

        std::thread inferThread;
        for (int startIdx = 0; startIdx < numSamples; startIdx += model.max_batch_size) {
          int endIdx = std::min<int>(numSamples, startIdx + model.max_batch_size);
          int nextEndIdx = std::min<int>(numSamples, endIdx + model.max_batch_size);

          // use cached probabilities if possible
          if ((derStep > 0)) {
            if (startIdx == 0) {
              // first instance -> run inference for first and second batch
              inferThread = model.infer_dist(modelRewriteDist, roundProgs, startIdx, endIdx); // TODO also pipeline with the derivation loop
            }
            inferThread.join(); // join with last inference thread

            if (endIdx < nextEndIdx) {// there is a batch coming after this one
              // start infering dist for next batch
              assert(!inferThread.joinable());
              inferThread = model.infer_dist(modelRewriteDist, roundProgs, endIdx, nextEndIdx);
            } else { // no more batches for this derivation step
              assert(!inferThread.joinable());
              inferThread = model.infer_dist(modelRewriteDist, roundProgs, endIdx, nextEndIdx);
            }
          }

          int frozen = 0;

          #pragma omp parallel for reduction(+:frozen)
          for (int t = startIdx; t < endIdx; ++t) {
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
              uniRule = (ruleRand(randGen()) <= pRandom);

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
                    !IsValidDistribution(modelRewriteDist[t].actionDist)) {
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

            stats.derivationFailures += (failureCount >= failureLimit);

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

        if (inferThread.joinable()) inferThread.join();
      }
    }

#undef IF_DEBUG_DER
    return states;
  }

  // search for a best derivation (best-reachable program (1.) through rewrites with minimal derivation sequence (2.))
  DerivationVec
  searchDerivations_Default(const ProgramVec & progVec, const double pRandom, const int maxDist, const int numOptRounds) {
    const int numSamples = progVec.size();
    std::uniform_real_distribution<float> ruleRand(0, 1);

    const bool useModel = pRandom != 1.0;

    // start with STOP derivation
    std::vector<Derivation> states;
    for (int i = 0; i < progVec.size(); ++i) {
      states.emplace_back(*progVec[i]);
    }

#define IF_DEBUG_DER if (false)

    // pre-compute initial program distribution
    ResultDistVec initialProgDist(progVec.size());
    if (useModel) { model.infer_dist(initialProgDist, progVec, 0, progVec.size()).join(); }

    // number of derivation walks
    for (int r = 0; r < numOptRounds; ++r) {

      // re-start from initial program
      ProgramVec roundProgs = Clone(progVec);

      ResultDistVec modelRewriteDist = initialProgDist;
      for (int derStep = 0; derStep < maxDist; ++derStep) {

        // use cached probabilities if possible
        if ((derStep > 0) && useModel) { model.infer_dist(modelRewriteDist, roundProgs, 0, roundProgs.size()).join(); }

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
                  !IsValidDistribution(modelRewriteDist[t].actionDist)) {
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

          stats.derivationFailures += (failureCount >= failureLimit);

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
      // assert(rew.pc < refResult.targetDist.size());
      int actionId = model.toActionID(rew);
      refResult.actionDist[actionId] += 1.0;
      // assert(ruleEnumId < refResult.ruleDist.size());

      IF_DEBUG_MC {
        std::cerr << "Prefix to best. pc=" << rew.pc << ", actionId=" << actionId << "\n";
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

  // sample a target based on the reference distributions (discards STOP programs)
  int
  sampleActions(ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, ProgramVec & oProgs) {
#define IF_DEBUG_SAMPLE if (false)
    std::uniform_real_distribution<float> pRand(0, 1.0);

    int numGenerated = 0;

    int rewriteIdx = 0;
    int nextSampleWithRewrite = rewrites.empty() ? std::numeric_limits<int>::max() : rewrites[rewriteIdx].first;
    for (int s = 0; s < refResults.size(); ++s) {
      IF_DEBUG_SAMPLE { std::cerr << "ACTION: " << refResults.size() << "\n"; }
      if (s < nextSampleWithRewrite) {
        // no rewrite available -> STOP
        // actionProgs.push_back(roundProgs[s]);
        continue;
      }

      // Otw, sample an action
      const int numRetries = 100;
      bool hit = false;
      for (int t = 0; !hit && (t < numRetries); ++t) { // FIXME consider a greedy strategy

        // model picks stop?
        bool shouldStop = pRand(randGen()) < refResults[s].stopDist;

        // valid distributions?
        if (!shouldStop && !IsValidDistribution(refResults[s].actionDist)) {
          hit = false;
          break;
        }

        // try to apply the action
        if (!shouldStop) {
          int actionId = SampleCategoryDistribution(refResults[s].actionDist, pRand(randGen()));
          Rewrite randomRew = model.toRewrite(actionId);
          IF_DEBUG_SAMPLE { std::cerr << "PICK: "; randomRew.print(std::cerr) << "\n";}

          // scan through legal actions until hit
          for (int i = rewriteIdx;
              i < rewrites.size() && rewrites[i].first == s;
              ++i)
          {
            if ((rewrites[i].second == randomRew)
            ) {
              assert(i < nextProgs.size());
              // actionProgs.push_back(nextProgs[i]);
              oProgs[numGenerated++] = nextProgs[i];
              hit = true;
              break;
            }
          }

          // proper action
        } else {
          // STOP action
          // actionProgs.push_back(roundProgs[s]);
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

        // actionProgs.push_back(roundProgs[s]); // soft failure
      }

      // advance to next progam with rewrites
      for (;rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s; ++rewriteIdx) {}

      if (rewriteIdx >= rewrites.size()) {
        nextSampleWithRewrite = std::numeric_limits<int>::max(); // no more rewrites -> mark all remaining programs as STOP
      } else {
        nextSampleWithRewrite = rewrites[rewriteIdx].first; // program with applicable rewrite in sight
      }
    }

    // assert(actionProgs.size() == roundProgs.size()); // no longer the case since STOP programs get dropped
#undef IF_DEBUG_SAMPLE
    return numGenerated;
  }

#undef IF_DEBUG_MV
};

// compute a scaore for the sample derivations (assuming refDef contains reference derivations)
struct DerStats {
  double matched; // ref == model
  double longerDer; // ref.score == model.score BUT model.der > ref.der
  double betterScore; // model.score < ref.score
  double shorterDer; // ref.score == model.score AND model.der < ref.der

  double getClearedScore() const { return matched + longerDer + betterScore + shorterDer; }
  double getMisses() const { return 1.0 - getClearedScore(); }

  void print(std::ostream&out) const {
    std::streamsize ss = out.precision();
    const int fpPrec = 4;
    out << std::fixed << std::setprecision(fpPrec)
      << " " << getClearedScore() << "  (matched " << matched << ", longerDer "<< longerDer << ", shorterDer: " << shorterDer << ", betterScore: " << betterScore << ")\n"
      << std::setprecision(ss) << std::defaultfloat; // restore
  }
};

int
CountStops(const DerivationVec & derVec) {
  int a = 0;
  for (const auto & der : derVec) a += (der.shortestDerivation == 0);
  return a;
}

DerStats
ScoreDerivations(const DerivationVec & refDer, const DerivationVec & sampleDer) {
  size_t numMatched = 0;
  size_t numLongerDer = 0;
  size_t numBetterScore = 0;
  size_t numShorterDer = 0;

  for (int i = 0; i < refDer.size(); ++i) {
    // improved over reference result
    if (sampleDer[i].betterThan(refDer[i])) {
      // actual improvements (shorter derivation or better program)
      if (sampleDer[i].bestScore < refDer[i].bestScore) {
        numBetterScore++;
      }  else if (sampleDer[i].shortestDerivation < refDer[i].shortestDerivation) {
        numShorterDer++;
      }
    }

    // number of targets hit
    if (sampleDer[i].bestScore == refDer[i].bestScore) {
      if (sampleDer[i].shortestDerivation > refDer[i].shortestDerivation) {
        numLongerDer++;
      } else {
        numMatched++;
      }
    }
  }
  return DerStats{
    numMatched / (double) refDer.size(),
    numLongerDer / (double) refDer.size(),
    numBetterScore / (double) refDer.size(),
    numShorterDer / (double) refDer.size()
  };
}


static
DerivationVec
FilterBest(DerivationVec A, DerivationVec B) {
  DerivationVec res;
  for (int i = 0; i < A.size(); ++i) {
    if (A[i].betterThan(B[i])) {
      res.push_back(A[i]);
    } else {
      res.push_back(B[i]);
    }
  }
  return res;
}

struct APO {
  RuleVec rules;
  Model model;
  std::string cpPrefix; // checkpoint prefix
  MonteCarloOptimizer montOpt;
  RPG rpg;
  Mutator expMut;

  std::string taskName; // name of task

  int minStubLen; //3; // minimal progrm stub len (excluding params and return)
  int maxStubLen; //4; // maximal program stub len (excluding params and return)
  int minMutations;// 1; // max number of program mutations
  int maxMutations;// 1; // max number of program mutations
  static constexpr double pExpand = 0.7; //0.7; // mutator expansion ratio

// mc search options
  int maxExplorationDepth; //maxMutations + 1; // best-effort search depth
  double pRandom; //1.0; // probability of ignoring the model for inference
  int numOptRounds; //50; // number of optimization retries

// eval round interval
  int logRate;
  size_t numRounds;

  bool saveCheckpoints; // save model checkpoints at @logRate

// training
  int numSamples;//

// number of simulation batches
  APO(const std::string & taskFile, const std::string & _cpPrefix)
  : rules(BuildRules())
  , model("build/rdn", "model.conf", rules.size() * 2)
  , cpPrefix(_cpPrefix)
  , montOpt(rules, model)
  , rpg(rules, model.num_Params)
  , expMut(rules, pExpand)
  {
    std::cerr << "Loading task file " << taskFile << "\n";

    Parser task(taskFile);
  // random program options
    taskName = task.get_or_fail<std::string>("name"); //3; // minimal progrm stub len (excluding params and return)

    numSamples = model.max_batch_size;
    minStubLen = task.get_or_fail<int>("minStubLen"); //3; // minimal progrm stub len (excluding params and return)
    maxStubLen = task.get_or_fail<int>("maxStubLen"); //4; // maximal program stub len (excluding params and return)
    minMutations = task.get_or_fail<int>("minMutations");// 1; // max number of program mutations
    maxMutations = task.get_or_fail<int>("maxMutations");// 1; // max number of program mutations

  // mc search options
    maxExplorationDepth = task.get_or_fail<int>("maxExplorationDepth"); //maxMutations + 1; // best-effort search depth
    pRandom = task.get_or_fail<double>("pRandom"); //1.0; // probability of ignoring the model for inference
    numOptRounds = task.get_or_fail<int>("numOptRounds"); //50; // number of optimization retries

    logRate = task.get_or_fail<int>("logRate"); // 10; // number of round followed by an evaluation
    numRounds = task.get_or_fail<size_t>("numRounds"); // 10; // number of round followed by an evaluation

    saveCheckpoints = task.get_or_fail<int>("saveModel") != 0; // save model checkpoints at @logRate

    if (saveCheckpoints) { std::cerr << "Saving checkpoints to prefix " << cpPrefix << "\n"; }

  // initialize thread safe random number generators
    InitRandom();
  }

  void
  generatePrograms(ProgramVec & progVec, size_t startIdx, size_t endIdx) {
    std::uniform_int_distribution<int> mutRand(minMutations, maxMutations);
    std::uniform_int_distribution<int> stubRand(minStubLen, maxStubLen);

    for (int i = startIdx; i < endIdx; ++i) {
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
    const int numEvalSamples = std::min<int>(4096, model.max_batch_size * 32);


    // hold-out evaluation set
    ProgramVec evalProgs(numEvalSamples, nullptr);
    generatePrograms(evalProgs, 0, evalProgs.size());
    std::cerr << "numSamples = " << numSamples << "\n"
              << "numEvalSamples = " << numEvalSamples << "\n";

  // training
    assert(minStubLen > 0 && "can not generate program within constraints");

    const int dotStep = logRate / 10;

  // Seed program generator
    ProgramVec progVec(numSamples, nullptr);
    generatePrograms(progVec, 0, progVec.size());

    // TESTING
    model.setLearningRate(0.0001);

    std::cerr << "\n-- Training --";
    for (size_t g = 0; g < numRounds; ++g) {
      bool loggedRound = (g % logRate == 0);
      if (loggedRound) {
        auto stats = model.query_stats();
        std::cerr << "\n- Round " << g << " ("; stats.print(std::cerr); std::cerr << ") -\n";

      // print MCTS statistics
        montOpt.stats.print(std::cerr);
        montOpt.stats = MonteCarloOptimizer::Stats();

        // random sampling based (uniform sampling, nio model)
        auto refDerVec = montOpt.searchDerivations(evalProgs, 1.0, maxExplorationDepth, numOptRounds);

        // one shot (model based)
        auto oneShotDerVec = montOpt.searchDerivations(evalProgs, 0.0, maxExplorationDepth, 1);

        // model-guided sampling
        const int guidedSamples = 4;
        auto guidedDerVec = montOpt.searchDerivations(evalProgs, 0.0, maxExplorationDepth, guidedSamples);

        int numStops = CountStops(refDerVec);
        DerStats oneShotStats = ScoreDerivations(refDerVec, oneShotDerVec);
        DerStats guidedStats = ScoreDerivations(refDerVec, guidedDerVec);

        double stopRatio = numStops / (double) numSamples;
        std::cerr << ". Stops  " << stopRatio << "\n";
        std::cerr << "\tOne shot "; oneShotStats.print(std::cerr);
        std::cerr << "\tGuided   "; guidedStats.print(std::cerr);

        // store model
        std::stringstream ss;
        ss << cpPrefix << "/" << taskName << "-" << g << ".cp";
        if (saveCheckpoints) model.saveCheckpoint(ss.str());

      } else {
        if (g % dotStep == 0) { std::cerr << "."; }
      }


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
      auto refDerVec = montOpt.searchDerivations(nextProgs, pRandom, maxExplorationDepth, numOptRounds);

      const int startRacketRound = logRate;
      if (g >= startRacketRound) {
        // model-driven search
        auto guidedDerVec = montOpt.searchDerivations(nextProgs, 0.1, maxExplorationDepth, 10);
        refDerVec = FilterBest(refDerVec, guidedDerVec);
      }

      // decode reference ResultDistVec from detected derivations
      ResultDistVec refResults;
      montOpt.populateRefResults(refResults, refDerVec, rewrites, nextProgs, progVec);

      // train model
      Model::Losses L;
      std::thread trainThread = model.train_dist(progVec, refResults, loggedRound ? &L : nullptr);

      // pick an action per program and drop STOP-ped programs
      int numNextProgs = montOpt.sampleActions(refResults, rewrites, nextProgs, progVec);
      double dropOutRate = 1.0 - numNextProgs / (double) numSamples;

      // fill up with new programs
      generatePrograms(progVec, numNextProgs, numSamples);

      if (loggedRound) {
        trainThread.join();
        std::cerr << "\t"; L.print(std::cerr) << ". Stop drop out=" << dropOutRate << "\n";
      } else {
        trainThread.detach();
      }
    }
  }
};


} // namespace apo

#endif // APO_APO_H
