#include "apo/mcts.h"

#include "apo/ruleBook.h"
#include "apo/ml.h"

#include <iomanip>
#include <iostream>

#define IF_DEBUG_MC if (false)

namespace apo {

int GetProgramScore(const Program &P) { return P.size(); }

std::ostream &Derivation::print(std::ostream &out) const {
  out << "Derivation (bestScore=" << bestScore
      << ", dist=" << shortestDerivation << ")";
  return out;
}

void Derivation::dump() const { print(std::cerr); }

bool Derivation::betterThan(const Derivation &o) const {
  if (bestScore < o.bestScore) {
    return true;
  } else if (bestScore == o.bestScore &&
             (shortestDerivation < o.shortestDerivation)) {
    return true;
  }
  return false;
}

bool Derivation::operator==(const Derivation &o) const {
  return (bestScore == o.bestScore) &&
         (shortestDerivation == o.shortestDerivation);
}
bool Derivation::operator!=(const Derivation &o) const { return !(*this == o); }


// optimize the given program @P using @model (or a uniform random rule
// application using @maxDist) maximal derivation length is @maxDist will return
// the sequence to the best-seen program (even if the model decides to go on)

std::ostream &
MonteCarloOptimizer::Stats::print(std::ostream &out) const {
  out << "MCOpt::Stats   "
      << "sampleActionFailures " << sampleActionFailures
      << ", invalidModelDists " << invalidModelDists << ", derivationFailures "
      << derivationFailures << ", validModelDerivations "
      << validModelDerivations;
  return out;
}


bool
MonteCarloOptimizer::greedyApplyModel(Program &P, Action &rew, ResultDist &res, bool &signalsStop) {
  // should we stop?
  if (res.stopDist > stopThreshold) {
    std::cerr << "STOP: " << res.stopDist << "\n";
    signalsStop = true;
    return true;
  } else {
    std::cerr << "don't STOP: " << res.stopDist << "\n";
  }

  // take random most-likely event
  signalsStop = false;

  // visit actions in descending order
  bool success = false;

  // std::cerr << "BEGIN GREEDY\n";
  VisitDescending(
      res.actionDist, [this, &P, &success, &rew](float pMass, int actionId) {
        if (pMass <= EPS) {
          // noise
          success = false;
          return false;
        }
        // std::cerr << "GREEDY " << pMass << " " << actionId << "\n";

        auto rew = ruleBook.toRewriteAction(actionId);
        if (rew.pc >= P.size() - 1)
          return true; // keep going -> invalid sample

        success = mut.tryApply(P, rew);
        return !success;
      });

  return success;
}

bool MonteCarloOptimizer::tryApplyModel(Program &P, Action &rewrite,
                                        ResultDist &res, bool &signalsStop) {
  // sample a random rewrite at a random location (product of rule and target
  // distributions)
  std::uniform_real_distribution<float> pRand(0, 1.0);

  // should we stop?
  if (pRand(randGen()) <= res.stopDist) {
    signalsStop = true;
    return true;
  }

  size_t keepGoing = 2;
  int targetId, ruleId;
  bool validPc;
  Action rew;
  do {
    // which rule to apply?
    int actionId = SampleCategoryDistribution(res.actionDist, pRand(randGen()));
    rew = ruleBook.toRewriteAction(actionId);

    // translate to internal rule representation
    validPc = rew.pc + 1 < P.size(); // do not rewrite returns
  } while (keepGoing-- > 0 && !validPc);

  // failed to sample a valid rule application -> STOP
  if (!validPc) {
    return false;
  }

  // Otw, rely on the mutator to do the job
  return mut.tryApply(P, rew);
}

// greedyDer - solution that respects STOP (stopDist > stopThreshold)
// bestDer - best een derivation along the path with highest action probability per program (ignores stopDist)
MonteCarloOptimizer::GreedyResult
MonteCarloOptimizer::greedyDerivation(const ProgramVec &origProgVec,
                                      const IntVec & maxDistVec) {
  ProgramVec progVec = Clone(origProgVec);

  DerivationVec stopStates(progVec.size());

  DerivationVec bestStates;
  bestStates.reserve(progVec.size());
  for (int t = 0; t < progVec.size(); ++t) {
    auto startDer = Derivation(*progVec[t]);
    bestStates.push_back(startDer); // do't-move-baseline (in case derivation are awful)
    stopStates.push_back(startDer); // baseline (if maxDistVec[t] is zero, will be overritten once by first @signalsStop
  }

  int frozen = 0; // amount of programs that have stopped derivation
  std::vector<bool> alreadyStopped(origProgVec.size(), false);

  // this loop keeps spinning until all threads have stopped the derivation (++frozen)
  for (int derStep = 0; frozen < progVec.size(); ++derStep) {

    // query action distribution
    ResultDistVec actionDistVec(progVec.size());
    model.infer_dist(actionDistVec, progVec, 0, progVec.size()).join();

#pragma omp parallel for reduction(+ : frozen)
    for (int t = 0; t < progVec.size(); ++t) { // for all programs
      if (alreadyStopped[t]) continue; // do nto proceed on STOP-ped programs

    // act (transform or STOP)
      Action rew;
      bool signalsStop = false;
      greedyApplyModel(*progVec[t], rew, actionDistVec[t], signalsStop);

      // shall we stop after this action?
      bool stopDerivation = (derStep + 1) >= maxDistVec[t] || // last derivation round (time out)
                            (progVec[t]->size() > model.config.prog_length); // in excess of maximal supported program length

      Derivation currState(
          GetProgramScore(*progVec[t]), // program after greedy action/STOP
          derStep + (signalsStop ? 0 : 1) // no step taken if this was a STOP
      );

    // update best seen solution
      if (currState.betterThan(bestStates[t])) {
        bestStates[t] = currState;
      }

    // first STOP signal -> record greedyDer final state
      if (!alreadyStopped[t] &&
          (signalsStop ||  // proper STOP signal
           stopDerivation) // last round (excess of max supported program length or last derivation round)
      ) {
        stopStates[t] = currState;
        alreadyStopped[t] = true;
        stopDerivation = true;
      }

      // stop derivating this program
      if (stopDerivation) {
        alreadyStopped[t] = true;
        ++frozen; continue;
      }
    }
  } // derivation loop

  // DEBUG CHECK
#if 0
  for (int i = 0; i < origProgVec.size(); ++i) {
    if (bestStates[i].betterThan(stopStates[i])) {
      std::cerr << "bestState "; bestStates[i].print(std::cerr) << "\n";
      std::cerr << "stopState "; stopStates[i].print(std::cerr) << "\n";
      abort();
    }
  }
#endif

  return GreedyResult{stopStates, bestStates};
}

// random trajectory based model (or uniform dist) sampling
DerivationVec MonteCarloOptimizer::searchDerivations(const ProgramVec &progVec,
                                                     const double pRandom,
                                                     const IntVec & maxDistVec,
                                                     const int numOptRounds,
                                                     bool allowFallback) {
  if (pRandom < 1.0)
    return searchDerivations_ModelDriven(progVec, pRandom, maxDistVec,
                                         numOptRounds, allowFallback);
  else
    return searchDerivations_Default(progVec, maxDistVec, numOptRounds);
}

// optimized version for model-based seaerch
DerivationVec MonteCarloOptimizer::searchDerivations_ModelDriven(
    const ProgramVec &progVec, const double pRandom, const std::vector<int> & maxDistVec,
    const int numOptRounds, const bool useRandomFallback) {
  assert(pRandom < 1.0 && "use _Debug implementation instead");

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
  Task handle = model.infer_dist(initialProgDist, progVec, 0, progVec.size());
  handle.join();

  // pre-compute maximal derivation distance
  int commonMaxDist = 0;
  for (int d : maxDistVec) commonMaxDist = std::max(commonMaxDist, d);

  // number of derivation walks
  for (int r = 0; r < numOptRounds; ++r) {

    // re-start from initial program
    ProgramVec roundProgs = Clone(progVec);

    ResultDistVec modelRewriteDist = initialProgDist;

    for (int derStep = 0; derStep < commonMaxDist; ++derStep) {

      Task inferThread;
      for (int startIdx = 0; startIdx < numSamples;
           startIdx += model.config.infer_batch_size) {
        int endIdx =
            std::min<int>(numSamples, startIdx + model.config.infer_batch_size);
        int nextEndIdx =
            std::min<int>(numSamples, endIdx + model.config.infer_batch_size);

        // use cached probabilities if possible
        if ((derStep > 0)) {
          if (startIdx == 0) {
            // first instance -> run inference for first and second batch
            inferThread = model.infer_dist(
                modelRewriteDist, roundProgs, startIdx,
                endIdx); // TODO also pipeline with the derivation loop
          }
          inferThread.join(); // join with last inference thread

          if (endIdx < nextEndIdx) { // there is a batch coming after this one
            // start infering dist for next batch
            assert(!inferThread.joinable());
            inferThread = model.infer_dist(modelRewriteDist, roundProgs, endIdx,
                                           nextEndIdx);
          } else { // no more batches for this derivation step
            assert(!inferThread.joinable());
            inferThread = model.infer_dist(modelRewriteDist, roundProgs, endIdx,
                                           nextEndIdx);
          }
        }

        int frozen = 0;

#pragma omp parallel for reduction(+ : frozen)
        for (int t = startIdx; t < endIdx; ++t) {
          // self inflicted timeout
          if (derStep >= maxDistVec[t]) {
            ++frozen;
            continue;
          }
          // freeze if derivation exceeds model
          if (roundProgs[t]->size() >= model.config.prog_length) {
            ++frozen;
            continue;
          }

          IF_DEBUG_DER if (derStep == 0) {
            std::cerr << "Initial prog " << t << ":\n";
            roundProgs[t]->dump();
          }

          // pick & apply a rewrite
          Action rewrite;
          bool success = false;
          bool signalsStop = false;

          // loop until rewrite succeeds (or stop)
          bool uniRule;
          uniRule = (ruleRand(randGen()) <= pRandom);

          // try to apply the model first
          if (!uniRule) {
            bool validDist =
                IsValidDistribution(modelRewriteDist[t].actionDist);
            if (validDist) {
              success = tryApplyModel(*roundProgs[t], rewrite,
                                      modelRewriteDist[t], signalsStop);
            }

            if (!success || !validDist) {
              if (useRandomFallback) {
                uniRule = true; // fall back to uniform application
              } else {
                signalsStop = true; // STOP on derivation failures
              }

              // stats
              if (!validDist) {
                stats.invalidModelDists++;
              } else {
                stats.derivationFailures++;
              }
            } else {
              stats.validModelDerivations++;
            }
          }

          // uniform random mutation
          if (uniRule) {
            rewrite = mut.mutate(*roundProgs[t], 1, pSearchExpand);
            IF_DEBUG_DER {
              std::cerr << "after random rewrite!\n";
              roundProgs[t]->dump();
            }
            success = true; // mutation always succeeeds
            signalsStop = false;
          }

          // don't step over STOP
          if (signalsStop) {
            ++frozen;
            continue;
          }

          // derived program to large for model -> freeze
          if (roundProgs[t]->size() > model.config.prog_length) {
            ++frozen;
            continue;
          }

          // Otw, update incumbent
          // mutated program
          int currScore = GetProgramScore(*roundProgs[t]);
          Derivation thisDer(currScore, derStep + 1);
          if (thisDer.betterThan(states[t])) {
            states[t] = thisDer;
          }
        }

        if (frozen == numSamples) {
          break; // all frozen -> early exit
        }
      }

      if (inferThread.joinable())
        inferThread.join();
    }
  }

#undef IF_DEBUG_DER
  return states;
}

// search for a best derivation (best-reachable program (1.) through rewrites
// with minimal derivation sequence (2.))
DerivationVec MonteCarloOptimizer::searchDerivations_Default(
    const ProgramVec &progVec, const std::vector<int> & maxDistVec, const int numOptRounds) {
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

  // number of derivation walks
  for (int r = 0; r < numOptRounds; ++r) {

    // re-start from initial program
    ProgramVec roundProgs = Clone(progVec);

    // use cached probabilities if possible
#pragma omp parallel for
    for (int t = 0; t < numSamples; ++t) {
      bool keepGoing = true;
      const int maxDist = maxDistVec[t];
      for (int derStep = 0; derStep < maxDist && keepGoing; ++derStep) { // TODO track max distance per program
        // freeze if derivation exceeds model
        if (roundProgs[t]->size() >= model.config.prog_length) {
          keepGoing = false;
          break;
        }

        IF_DEBUG_DER if (derStep == 0) {
          std::cerr << "Initial prog " << t << ":\n";
          roundProgs[t]->dump();
        }

        // pick & apply a rewrite
        Action rewrite;
        bool signalsStop = false;

        // loop until rewrite succeeds (or stop)
        // uniform random rewrite
        rewrite = mut.mutate(*roundProgs[t], 1, pSearchExpand);
        IF_DEBUG_DER {
          std::cerr << "after random rewrite!\n";
          roundProgs[t]->dump();
        }
        signalsStop = false; // HACK to always keep going

        // don't step over STOP
        if (signalsStop) {
          keepGoing = false;
          break;
        }

        // derived program to large for model -> freeze
        if (roundProgs[t]->size() > model.config.prog_length) {
          keepGoing = false;
          break;
        }

        // Otw, update incumbent
        // mutated program
        int currScore = GetProgramScore(*roundProgs[t]);
        Derivation thisDer(currScore, derStep + 1);
        if (thisDer.betterThan(states[t])) {
          states[t] = thisDer;
        }
      }
    }
  }

#undef IF_DEBUG_DER
  return states;
}

// convert detected derivations to refernce distributions
void MonteCarloOptimizer::encodeBestDerivation(ResultDist &refResult, const DerivationVec &derivations, const CompactedRewrites &rewrites, int startIdx, int progIdx) const {
  // find best-possible rewrite
  assert(startIdx < derivations.size());
  bool noBetterDerivation = true;
  Derivation bestDer;
  for (int i = startIdx; i < rewrites.size() && (rewrites[i].first == progIdx);
       ++i) {
    const auto &der = derivations[i];
    if (noBetterDerivation || der.betterThan(bestDer)) {
      noBetterDerivation = false;
      bestDer = der;
    }
  }

  if (noBetterDerivation) {
    // no way to improve over STOP
    refResult = model.createStopResult();
    return;
  }

  IF_DEBUG_MC {
    std::cerr << progIdx << " -> best ";
    bestDer.dump();
    std::cerr << "\n";
  }

  // activate all positions with best rewrites
  bool noBestDerivation = true;
  for (int i = startIdx; i < rewrites.size() && (rewrites[i].first == progIdx);
       ++i) {
    if (derivations[i] != bestDer) {
      continue;
    }
    noBestDerivation = false;
    const auto &rew = rewrites[i].second;
    // assert(rew.pc < refResult.targetDist.size());
    int actionId = ruleBook.toActionID(rew);
    refResult.actionDist[actionId] += 1.0;
    // assert(ruleEnumId < refResult.ruleDist.size());
  }

  assert(!noBestDerivation);
}

void MonteCarloOptimizer::populateRefResults(ResultDistVec &refResults,
                                             const DerivationVec &derivations,
                                             const CompactedRewrites &rewrites,
                                             int numResults) const {
  int rewriteIdx = 0;
  int nextSampleWithRewrite = rewrites[rewriteIdx].first;
  for (int s = 0; s < numResults; ++s) {
    // program without applicable rewrites
    if (s < nextSampleWithRewrite) {
      refResults.push_back(model.createStopResult());
      continue;
    } else {
      refResults.push_back(model.createEmptyResult());
    }

    // convert to a reference distribution
    encodeBestDerivation(refResults[s], derivations, rewrites, rewriteIdx, s);

    // skip to next progam with rewrites
    for (; rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s;
         ++rewriteIdx) {
    }

    if (rewriteIdx >= rewrites.size()) {
      nextSampleWithRewrite =
          std::numeric_limits<int>::max(); // no more rewrites -> mark all
                                           // remaining programs as STOP
    } else {
      nextSampleWithRewrite =
          rewrites[rewriteIdx]
              .first; // program with applicable rewrite in sight
    }
  }

  // normalize distributions
  for (int s = 0; s < numResults; ++s) {
    auto &result = refResults[s];
    result.normalize();

#if 0
    IF_DEBUG_MC {
      std::cerr << "\n Sample " << s << ":\n";
      progVec[s]->dump();
      std::cerr << "Result ";
      result.dump();
    }
#endif
  }
}

// sample a target based on the reference distributions (discards STOP programs)

// int sampleActions(ResultDistVec & refResults, const CompactedRewrites & rewrites, const ProgramVec & nextProgs, const IntVec & nextMaxDerVec, ProgramVec & oProgs, IntVec & oMaxDer);
void MonteCarloOptimizer::sampleActions(const ResultDistVec &refResults,
                                       const CompactedRewrites &rewrites,
                                       const ProgramVec &nextProgs,
                                       ActionCallback actionHandler,
                                       StopCallback stopHandler
  ) {
#define IF_DEBUG_SAMPLE if (false)
  std::uniform_real_distribution<float> pRand(0, 1.0);

  int numGenerated = 0;

  int rewriteIdx = 0;
  int nextSampleWithRewrite = rewrites.empty() ? std::numeric_limits<int>::max()
                                               : rewrites[rewriteIdx].first;
  for (int s = 0; s < refResults.size(); ++s) {
    IF_DEBUG_SAMPLE { std::cerr << "ACTION: " << refResults.size() << "\n"; }
    if (s < nextSampleWithRewrite) {
      bool keepGoing = stopHandler(s, StopReason::NoPossibleAction);
      if (!keepGoing) return;

      // no rewrite available -> STOP
      // actionProgs.push_back(roundProgs[s]);
      continue;
    }

    // Otw, sample an action
    const int numRetries = 100;
    bool hit = false;
    bool checkedDist = false;
    for (int t = 0; !hit && (t < numRetries);
         ++t) { // FIXME consider a greedy strategy

      // model picks stop?
      bool shouldStop = pRand(randGen()) < refResults[s].stopDist;

      if (shouldStop) {
        bool keepGoing = stopHandler(s, StopReason::Choice); // stopped by choice
        if (!keepGoing) return;
        hit = true;
        break;
      }

      // valid distributions?
      if (!checkedDist && !IsValidDistribution(refResults[s].actionDist)) {
        bool keepGoing = stopHandler(s, StopReason::InvalidDist); // stopped by invalid action
        if (!keepGoing) return;
        checkedDist = true;
        hit = false;
        break;
      }

      // try to apply the action
      int actionId = SampleCategoryDistribution(refResults[s].actionDist,
                                                pRand(randGen()));
      Action randomRew = ruleBook.toRewriteAction(actionId);
      IF_DEBUG_SAMPLE {
        std::cerr << "PICK: ";
        randomRew.print(std::cerr) << "\n";
      }

      // scan through legal actions until hit
      for (int i = rewriteIdx; i < rewrites.size() && rewrites[i].first == s;
           ++i) {
        if ((rewrites[i].second == randomRew)) {

          assert(i < nextProgs.size());
          actionHandler(s, i);

          hit = true;
          break;
        }
      }
    }

    // could not hit -> STOP
    if (!hit) {
      bool keepGoing = stopHandler(s, StopReason::DerivationFailure);
      if (!keepGoing) return;
      stats.sampleActionFailures++;
    }

    // advance to next progam with rewrites
    for (; rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s;
         ++rewriteIdx) {
    }

    if (rewriteIdx >= rewrites.size()) {
      nextSampleWithRewrite =
          std::numeric_limits<int>::max(); // no more rewrites -> mark all
                                           // remaining programs as STOP
    } else {
      nextSampleWithRewrite =
          rewrites[rewriteIdx]
              .first; // program with applicable rewrite in sight
    }
  }

    // assert(actionProgs.size() == roundProgs.size()); // no longer the case
    // since STOP programs get dropped
#undef IF_DEBUG_SAMPLE
}

#undef IF_DEBUG_MC





void DerStats::print(std::ostream &out) const {
  std::streamsize ss = out.precision();
  const int fpPrec = 4;
  out << std::fixed << std::setprecision(fpPrec) << " " << getClearedScore()
      << "  (matched " << matched << ", longerDer " << longerDer
      << ", shorterDer: " << shorterDer << ", betterScore: " << betterScore
      << ")\n"
      << std::setprecision(ss) << std::defaultfloat; // restore
}

DerStats ScoreDerivations(const DerivationVec &refDer,
                          const DerivationVec &sampleDer) {
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
      } else if (sampleDer[i].shortestDerivation <
                 refDer[i].shortestDerivation) {
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
  return DerStats{numMatched / (double)refDer.size(),
                  numLongerDer / (double)refDer.size(),
                  numBetterScore / (double)refDer.size(),
                  numShorterDer / (double)refDer.size()};
}

} // namespace apo
