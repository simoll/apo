#include "apo/mcts.h"

#include "apo/ruleBook.h"
#include "apo/ml.h"
#include "apo/timers.h"

#include <iomanip>
#include <iostream>

#define IF_DEBUG_MC if (false)

namespace apo {

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
    signalsStop = true;
    return true;
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

  // should we stop?
  if (drawUnitRand() <= res.stopDist) {
    signalsStop = true;
    return true;
  }

  size_t keepGoing = 2;
  int targetId, ruleId;
  bool validPc;
  Action rew;
  do {
    // which rule to apply?
    int actionId = SampleCategoryDistribution(res.actionDist, drawUnitRand());
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

void
MonteCarloOptimizer::greedyOptimization(ProgramVec & oBestVec, ProgramVec & oStopVec, ProgramVec & progVec, IntVec & reqSteps, const IntVec & maxStepsVec, int startId, int endId, std::string inferTower) {
  const int numJobs = endId - startId;

  if (endId <= startId) return;

  int frozen = 0; // amount of programs that have stopped derivation
  std::vector<char> alreadyStopped(numJobs, false);

  // compute initial score
  std::vector<Derivation> bestDer;
  for (int i = startId; i < endId; ++i) {
    bestDer.push_back(Derivation(*progVec[i]));
    oBestVec[i].reset(new Program(*progVec[i]));
  }

  // this loop keeps spinning until all threads have stopped the derivation (++frozen)
  for (int derStep = 0; frozen < numJobs; ++derStep) {

    // query action distribution
    ResultDistVec actionDistVec(endId + 1); // FIXME pass iterators to infer_dist instead
    model.infer_dist(actionDistVec, progVec, startId, endId, inferTower).join();

    // TODO (fetch CPU lock?)
#pragma omp parallel for \
        reduction(+ : frozen) \
        shared(actionDistVec,alreadyStopped,progVec,maxStepsVec)
    for (int t = 0; t < numJobs; ++t) { // for all programs

      const int progId = startId + t;

    // act (transform or STOP)
      Action rew;
      bool signalsStop = false;
      greedyApplyModel(*progVec[progId], rew, actionDistVec[progId], signalsStop);

      // shall we stop after this action?
      bool stopDerivation = (derStep + 1) >= maxStepsVec[progId] || // last derivation round (time out)
                            (progVec[progId]->size() > model.config.prog_length); // in excess of maximal supported program length

      int curScore = GetProgramScore(*progVec[progId]);

      Derivation curDer(
          curScore, // program after greedy action/STOP
          derStep + (signalsStop ? 0 : 1) // no step taken if this was a STOP
      );

    // update best seen program on this derivation
      if (curDer.betterThan(bestDer[t])) {
        reqSteps[progId] = derStep; // keep track of required step for BESt solution
        oBestVec[progId].reset(new Program(*progVec[progId]));
        bestDer[t] = curDer;
      }

    // signalled STOP (or last reached program)
      if (!alreadyStopped[t] && (stopDerivation || signalsStop)) {
        oStopVec[progId].reset(new Program(*progVec[progId])); // remember stopped program
        alreadyStopped[t] = true;
      }

   // last derivation (freeze state)
      if (stopDerivation) {
        ++frozen;
      }
    }
  } // derivation loop
}

// greedyDer - solution that respects STOP (stopDist > stopThreshold)
// bestDer - best een derivation along the path with highest action probability per program (ignores stopDist)
MonteCarloOptimizer::GreedyResult
MonteCarloOptimizer::greedyDerivation(const ProgramVec &origProgVec,
                                      const IntVec & maxStepsVec, const DeviceVec & devices) {
  ProgramVec progVec = Clone(origProgVec);
  GreedyResult res(progVec.size());

  Distribute(devices, progVec.size(),
      [this, &res, &progVec, &devices, &maxStepsVec](int devId, int startId, int endId) {
        greedyDerivation(res.bestVec, res.greedyVec, progVec, maxStepsVec, startId, endId, devices[devId].tower);
      }
  );

  return res;
}

void
MonteCarloOptimizer::greedyDerivation(DerivationVec & bestStates, DerivationVec & stopStates, ProgramVec & progVec, const IntVec & maxStepsVec, int startId, int endId, std::string inferTower) {
  if (endId <= startId) return;

  const int numJobs = endId - startId;

  for (int i = startId; i < endId; ++i) {
    auto startDer = Derivation(*progVec[i]);
    bestStates[i] = startDer; // do't-move-baseline (in case derivation are awful)
    stopStates[i] = startDer; // baseline (if maxStepsVec[t] is zero, will be overritten once by first @signalsStop
  }

  int frozen = 0; // amount of programs that have stopped derivation
  std::vector<char> alreadyStopped(numJobs, false);
  assert(bestStates.size() == stopStates.size());


  // this loop keeps spinning until all threads have stopped the derivation (++frozen)
  for (int derStep = 0; frozen < numJobs; ++derStep) {

    // query action distribution
    ResultDistVec actionDistVec(progVec.size()); // FIXME pass iterator to infer_dist instead
    model.infer_dist(actionDistVec, progVec, 0, progVec.size(), inferTower).join();

    // std::unique_lock cpuLock(cpuMutex);
#pragma omp parallel for \
        reduction(+ : frozen) \
        shared(actionDistVec,alreadyStopped,progVec,maxStepsVec,bestStates,stopStates) \
        num_threads(6)
    for (int t = 0; t < numJobs; ++t) { // for all programs
      if (alreadyStopped[t]) continue; // do not proceed on STOP-ped programs
      int progId = startId + t;

    // act (transform or STOP)
      Action rew;
      bool signalsStop = false;
      greedyApplyModel(*progVec[progId], rew, actionDistVec[progId], signalsStop);

      // shall we stop after this action?
      bool stopDerivation = (derStep + 1) >= maxStepsVec[progId] || // last derivation round (time out)
                            (progVec[progId]->size() > model.config.prog_length); // in excess of maximal supported program length

      Derivation currState(
          GetProgramScore(*progVec[progId]), // program after greedy action/STOP
          derStep + (signalsStop ? 0 : 1) // no step taken if this was a STOP
      );

    // update best seen solution
      if (currState.betterThan(bestStates[progId])) {
        bestStates[progId] = currState;
      }

    // first STOP signal -> record greedyDer final state
      if (!alreadyStopped[t] &&
          (signalsStop ||  // proper STOP signal
           stopDerivation) // last round (excess of max supported program length or last derivation round)
      ) {
        stopStates[progId] = currState;
      }

    // stop derivating this program
      if (stopDerivation || signalsStop) {
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
}

// random trajectory based model (or uniform dist) sampling
DerivationVec MonteCarloOptimizer::searchDerivations(const ProgramVec &progVec,
                                                     const double pRandom,
                                                     const IntVec & maxStepsVec,
                                                     const int numOptRounds,
                                                     bool allowFallback,
                                                     const DeviceVec & devices,
                                                     SearchPerfStats * oStats)
{
  if (pRandom < 1.0) {
    // model driven queries
    assert(!devices.empty() && "no infer device available!");

    // compute common STOP derivation
    std::vector<Derivation> initialStates;
    initialStates.reserve(progVec.size());
    for (int i = 0; i < progVec.size(); ++i) {
      initialStates.emplace_back(*progVec[i]);
    }

    // distribute load onto devices
    std::vector<DerivationVec> deviceStates(devices.size(), initialStates);

    Distribute(devices, numOptRounds,
         [this, &devices, &deviceStates, &progVec, &maxStepsVec, numOptRounds, pRandom, allowFallback, oStats](int deviceId, int startRounds, int endRounds)
      {
        const auto & dev = devices[deviceId];
        // std::cerr << "Scheduling " << startRounds << " to " << endRounds << " on " << dev.tower << "\n";
        int numRounds = endRounds - startRounds;

        // model-driven search
        searchDerivations_ModelDriven(deviceStates[deviceId], 0, progVec.size(), progVec, pRandom, maxStepsVec, numRounds, allowFallback, dev.tower, oStats);
      }
    );

    if (devices.size() > 1) {
      // reduce(bestDer:FilterBest)
      DerivationVec bestDer;
      for (int i = 0; i < devices.size(); ++i) {
        if (i == 0) bestDer = deviceStates[0];
        else bestDer = FilterBest(bestDer, deviceStates[i]);
      }
      assert(bestDer.size() == progVec.size());
      return bestDer;
    } else {
      return deviceStates[0];
    }

  } else {
    // random search
    return searchDerivations_Default(progVec, maxStepsVec, numOptRounds); // does not support profiling
  }
}

// optimized version for model-based seaerch
void
MonteCarloOptimizer::searchDerivations_ModelDriven(DerivationVec & states, int startSlice, int endSlice,
    const ProgramVec &progVec, const double pRandom, const std::vector<int> & maxDistVec,
    const int numOptRounds, const bool useRandomFallback, std::string inferTower,
    SearchPerfStats * oPerfStats)
{
  assert(pRandom < 1.0 && "use _Debug implementation instead");

  assert(0 <= startSlice && endSlice <= progVec.size());
  const int numSamples = endSlice - startSlice;

  double searchStart = get_wall_time();
  // const int numSamples = progVec.size();

#define IF_DEBUG_DER if (false)


  // re-start from initial program
  ProgramVec roundProgs = ClonePart(progVec, startSlice, endSlice);

  // pre-compute initial program distribution
  ResultDistVec initialProgDist(numSamples);
  Task handle = model.infer_dist(initialProgDist, roundProgs, 0, numSamples, inferTower);
  double joinStart = get_wall_time();
  handle.join();
  double joinEnd = get_wall_time();
  double initialJoinTime = joinEnd - joinStart;

  // pre-compute maximal derivation distance
  int commonMaxDist = 0;
  for (int j = startSlice ; j < endSlice; ++j) commonMaxDist = std::max(commonMaxDist, maxDistVec[j]);

  double inferStallTotal = 0;

  // number of derivation walks
  for (int r = 0;
      r < numOptRounds;
      ++r, ClonePartInto(roundProgs, progVec, startSlice, endSlice)
  ) {
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
                endIdx, inferTower); // TODO also pipeline with the derivation loop
          }
          double joinStart = get_wall_time();
          inferThread.join(); // join with last inference thread
          double joinEnd = get_wall_time();
          inferStallTotal += joinEnd - joinStart;

          if (endIdx < nextEndIdx) { // there is a batch coming after this one
            // start infering dist for next batch
            assert(!inferThread.joinable());
            inferThread = model.infer_dist(modelRewriteDist, roundProgs, endIdx,
                                           nextEndIdx, inferTower);
          } else { // no more batches for this derivation step
            assert(!inferThread.joinable());
            inferThread = model.infer_dist(modelRewriteDist, roundProgs, endIdx,
                                           nextEndIdx, inferTower);
          }
        }

        int frozen = 0;

#pragma omp parallel for \
            reduction(+ : frozen) \
            shared(maxDistVec, roundProgs, modelRewriteDist,states)
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
          uniRule = (drawUnitRand() <= pRandom);

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
#if 0
              // FIXME racy
              if (!validDist) {
                stats.invalidModelDists++;
              } else {
                stats.derivationFailures++;
              }
#endif
            }
#if 0
            // FIXME racy
            else {
              stats.validModelDerivations++;
            }
#endif
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

  double searchEnd = get_wall_time();

  if (oPerfStats) {
    double searchSecs = (searchEnd - searchStart);
    double initialStallSecs = initialJoinTime;
    double inferStallSecs = inferStallTotal;
    *oPerfStats = SearchPerfStats{progVec.size(), searchSecs, initialStallSecs, inferStallSecs};
  }

#undef IF_DEBUG_DER
}

// search for a best derivation (best-reachable program (1.) through rewrites
// with minimal derivation sequence (2.))
DerivationVec MonteCarloOptimizer::searchDerivations_Default(const ProgramVec &progVec, const std::vector<int> & maxDistVec, const int numOptRounds) {
  const int numSamples = progVec.size();

  // std::cerr << "Initial programs\n";
  // start with STOP derivation
  std::vector<Derivation> states;
  for (int i = 0; i < progVec.size(); ++i) {
    // progVec[i]->dump();
    states.emplace_back(*progVec[i]);
  }

#define IF_DEBUG_DER if (false)

  // number of derivation walks
  for (int r = 0; r < numOptRounds; ++r) {
    // std::cerr << "Derivation " << r << "\n";

    // re-start from initial program
    ProgramVec roundProgs = Clone(progVec);

    // use cached probabilities if possible
#pragma omp parallel for \
        shared(maxDistVec, roundProgs, mut, model, states)
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

        // loop until rewrite succeeds (or stop)
        // uniform random rewrite
        rewrite = mut.mutate(*roundProgs[t], 1, pSearchExpand);
        IF_DEBUG_DER {
          std::cerr << "after random rewrite!\n";
          roundProgs[t]->dump();
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
void MonteCarloOptimizer::encodeBestDerivation(ResultDist &refResult, const DerivationVec &derivations, const CompactedRewrites &rewrites, const Derivation stopDer, int startIdx, int progIdx) const {
  // find best-possible rewrite
  assert(startIdx < derivations.size());
  bool noBetterDerivation = true;
  Derivation bestDer;
  int bestIdx = -1;
  for (int i = startIdx; i < rewrites.size() && (rewrites[i].first == progIdx);
       ++i) {
    const auto &  der = derivations[i];
    if (noBetterDerivation || der.betterThan(bestDer)) {
      noBetterDerivation = false;
      bestDer = der;
      bestIdx = i;
    }
  }


  if (noBetterDerivation || // no valid action available
      stopDer.bestScore <= bestDer.bestScore // stil better to STOP than to transform..
  ) {
    IF_DEBUG_MC { std::cerr << "STOP.\n";  }
    // no way to improve over STOP
    refResult = model.createStopResult();
    return;
  }

  IF_DEBUG_MC {
    std::cerr << progIdx << " -> best (rewIdx = " << bestIdx << ") ";
    bestDer.dump();
    std::cerr << "\n";
  }

  // activate all positions with best rewrites
  bool noBestDerivation = true;
  // learn all equivalent moves (in terms of dist)
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
    IF_DEBUG_MC { std::cerr << "\t found at rewIdx=" << i << ", actionId= " << actionId << " : "; rew.print(std::cerr) << "\n"; }
    // assert(ruleEnumId < refResult.ruleDist.size());
  }

  assert(!noBestDerivation);
}

ResultDistVec
MonteCarloOptimizer::populateRefResults(const DerivationVec &derivations,
                                        const CompactedRewrites &rewrites,
                                        const ProgramVec & progVec) const {
  assert(derivations.size() == rewrites.size());

  ResultDistVec refResults;
  refResults.reserve(progVec.size());
  int rewriteIdx = 0;
  int nextSampleWithRewrite = derivations.empty() ? std::numeric_limits<int>::max() : rewrites[rewriteIdx].first; // submit STOP everywhere if no derivations are available
  for (int s = 0; s < progVec.size(); ++s) {
    // program without applicable rewrites
    if (s < nextSampleWithRewrite) {
      refResults.push_back(model.createStopResult());
      continue;
    } else {
      refResults.push_back(model.createEmptyResult());
    }

    assert(refResults.size() == s + 1);

    // convert to a reference distribution
    Derivation stopDer(*progVec[s]);
    IF_DEBUG_MC { std::cerr << "Prog " << s << " "; }
    encodeBestDerivation(refResults[s], derivations, rewrites, stopDer, rewriteIdx, s);

    // skip to next progam with rewrites
    for (; rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s;
         ++rewriteIdx) {
    }

    if (rewriteIdx >= rewrites.size()) {
      nextSampleWithRewrite = std::numeric_limits<int>::max(); // no more rewrites -> mark all
    } else {
      nextSampleWithRewrite = rewrites[rewriteIdx].first; // program with applicable rewrite in sight
    }
  }

  // normalize distributions
  for (int s = 0; s < progVec.size(); ++s) {
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

  assert(refResults.size() == progVec.size());
  return refResults;
}

// sample a target based on the reference distributions (discards STOP programs)

// rewrites are aligned with nextProgs and need to be in ascending order
// refResults[t] is the reference distribution for progVec[t]
// rewrites and nextProgs are compact lists of re-written programs in ascending order (with regards to progVec[t])
// that means the resulting programs from progVec[t] fall compactly in a range [start_t,.. ,end_t] in the rewrites and nextProgs arrays
void MonteCarloOptimizer::sampleActions(const ResultDistVec &refResults,
                                       const CompactedRewrites &rewrites,
                                       const ProgramVec &nextProgs,
                                       ActionCallback &&actionHandler,
                                       StopCallback &&stopHandler
  ) {
#define IF_DEBUG_SAMPLE if (false)
  int rewriteIdx = 0;
  int nextSampleWithRewrite = 0;

  for (int s = 0; s < refResults.size(); ++s) {
    IF_DEBUG_SAMPLE { std::cerr << "ACTION: " << refResults.size() << "\n"; }

    // advance to next block of rewrites
    // nextSampleWithRewrites == s (if any rewites exists). Otw, nextSampleWithRewrites > s.
    for (; rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first < s; ++rewriteIdx) {
      nextSampleWithRewrite = rewrites[rewriteIdx].first;
    }
    if (rewriteIdx == rewrites.size()) {
      nextSampleWithRewrite = std::numeric_limits<int>::max(); // no remaining rewrites -> STOP (choice or forced)
    }

    // model picks stop?
    bool stopByChoice = drawUnitRand() <= refResults[s].stopDist;

    if (stopByChoice) {
      bool keepGoing = stopHandler(s, StopReason::Choice); // stopped by choice
      if (!keepGoing) return;
      continue;
    }

    // no available rewrites for this prog (but did not STOP either)
    if (s < nextSampleWithRewrite) {
      assert(!stopByChoice);
      bool keepGoing = stopHandler(s, StopReason::NoPossibleAction);
      if (!keepGoing) return;
      continue;
    }

    assert((rewrites[rewriteIdx].first == s) && "current rewrite position should point to rewrite block for s");

    // Otw, sample an action
    const int numRetries = 100;
    bool hit = false;
    bool checkedDist = false;
    for (int t = 0; !hit && (t < numRetries);
         ++t) { // FIXME consider a greedy strategy

      // valid distributions?
      if (!checkedDist && !IsValidDistribution(refResults[s].actionDist)) {
        bool keepGoing = stopHandler(s, StopReason::InvalidDist); // stopped by invalid action
        if (!keepGoing) return;
        checkedDist = true;
        hit = false;
        break;
      }

      // try to apply the action
      int actionId = SampleCategoryDistribution(refResults[s].actionDist, drawUnitRand());
      Action randomRew = ruleBook.toRewriteAction(actionId);
      IF_DEBUG_SAMPLE {
        std::cerr << "PICK: ";
        randomRew.print(std::cerr) << "\n";
      }

      // scan through rewrite block until hit
      for (; rewriteIdx < rewrites.size() && rewrites[rewriteIdx].first == s; ++rewriteIdx) {
        if ((rewrites[rewriteIdx].second == randomRew)) {

          assert(rewriteIdx < nextProgs.size());
          // apply action
          actionHandler(s, rewriteIdx);
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
  }

    // assert(actionProgs.size() == roundProgs.size()); // no longer the case
    // since STOP programs get dropped
#undef IF_DEBUG_SAMPLE
}

#undef IF_DEBUG_MC





std::ostream&
DerStats::print(std::ostream &out) const {
  std::streamsize ss = out.precision();
  const int fpPrec = 4;
  out << std::fixed << std::setprecision(fpPrec) << " " << getClearedScore()
      << "  (matched " << matched << ", longerDer " << longerDer
      << ", shorterDer: " << shorterDer << ", betterScore: " << betterScore
      << ")"
      << std::setprecision(ss) << std::defaultfloat; // restore
  return out;
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

    // number of targets hit
    } else if (sampleDer[i].bestScore == refDer[i].bestScore) {
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
