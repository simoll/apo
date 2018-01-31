#include "apo/apo.h"

#include <random>
#include <limits>
#include <thread>
#include <condition_variable>
#include <atomic>

namespace apo {

struct
SampleServer {
  std::uniform_real_distribution<float> dropRand;
  std::uniform_int_distribution<int> elemRand;

  int queueLimit; // maximal training elements to reside in queue

  struct CachedDerivation {
    int numDrawn;               // how often was this sample drawn
    Derivation bestKnownDer;    // best known derivation

    CachedDerivation()
    {}

    CachedDerivation(Derivation _bestDer)
    : numDrawn(0)
    , bestKnownDer(_bestDer)
    {}
  };

  // internal sample map
  std::map<ProgramPtr, CachedDerivation, deref_less<ProgramPtr>> sampleMap;

  struct
  TrainingSample {
    ProgramPtr P;
    ResultDist resultDist;

    TrainingSample(ProgramPtr _P, ResultDist _resultDist)
    : P(_P), resultDist(_resultDist)
    {}
  };

  // training sample queue
  std::queue<TrainingSample> trainingQueue;
  std::condition_variable consumerCV; // waiting
  std::condition_variable producerCV;
  std::mutex queueMutex;

  // statistics
  struct QueueStats {
    int numQueuePush; // number of ::submitResults to queue
    size_t numWaitlessPush; // total time spent waiting (::submtResults)
    size_t totalStallPush; // total time spent waiting (::submtResults)
    int numQueuePull; // number of ::drawSamples from queue
    int numWaitlessPull;
    size_t totalStallPull; // total time spent waiting for data (::drawSamples)

    void addPull(clock_t stallTime) {
      ++numQueuePull;
      numWaitlessPull += (stallTime == 0);
      totalStallPull += stallTime;
    }

    void addPush(clock_t stallTime) {
      ++numQueuePush;
      numWaitlessPush += (stallTime == 0);
      totalStallPush += stallTime;
    }

    double getPushStall() const { return numQueuePush > 0 ? ((totalStallPush / (double) numQueuePush) / CLOCKS_PER_SEC) : 0; }
    double getPullStall() const { return numQueuePull> 0 ? ((totalStallPull / (double) numQueuePull) / CLOCKS_PER_SEC) : 0; }

    double getWaitlessPullRatio() const { return numQueuePull > 0 ? (numWaitlessPull / (double) numQueuePull) : 1.0; }
    double getWaitlessPushRatio() const { return numQueuePush > 0 ? (numWaitlessPush / (double) numQueuePush) : 1.0; }

    std::ostream&
    print(std::ostream & out) const {
      out << "Queue (avgPushStall=" << getPushStall() << " s, fastPushRatio=" <<  getWaitlessPushRatio()
          << ", avgPullStall=" << getPullStall() << "s, fastPullRatio=" << getWaitlessPullRatio() << ")";
      return out;
    }
    QueueStats()
    : numQueuePush(0)
    , numWaitlessPush(0)
    , totalStallPush(0)
    , numQueuePull(0)
    , numWaitlessPull(0)
    , totalStallPull(0)
    {}
  };

  QueueStats queueStats;

  // read out current stats and reset
  QueueStats
  resetQueueStats() {
    std::unique_lock lock(queueMutex);
    auto currStats = queueStats;
    queueStats = QueueStats();
    return currStats;
  }


  SampleServer(const std::string & serverConfig)
  : dropRand(0, 1)
  , elemRand(0, std::numeric_limits<int>::max())
  , sampleMap()
  , queueStats()
  {
    Parser cfg(serverConfig);

    queueLimit = cfg.get_or_fail<int>("queueLimit"); // number of training samples in queue
  }

  // submit a complete action distribution
  void
  submitResults(ProgramVec & progVec, const ResultDistVec & resDistVec) {
    {
      std::unique_lock queueLock(queueMutex);

      // wait until slots in the training queue become available
      clock_t stallTime = 0;
      if (trainingQueue.size() > queueLimit) {
        clock_t startStallPush = clock();
        producerCV.wait(queueLock, [this](){ return trainingQueue.size() < queueLimit; });
        clock_t endStallPush = clock();
        stallTime = (endStallPush - startStallPush);
      }
      queueStats.addPush(stallTime);

      // fill slots
      for (int i = 0; i < progVec.size(); ++i) {
        trainingQueue.emplace(progVec[i], resDistVec[i]);
      }
    }

    consumerCV.notify_all();
  }

  // store a new program result in the cache (return true if the cache was improved by the operation (new program or better derivation))
  bool
  submitDerivation(ProgramPtr P, Derivation sampleDer) {
    return false; // FIXME use the sample cache

    auto itSample = sampleMap.find(P);
    if (itSample == sampleMap.end()) {
      sampleMap[P] = CachedDerivation(sampleDer);
      return true;
    } else {
      auto & cached = itSample->second;
      if (sampleDer.betterThan(cached.bestKnownDer)) {
        cached.bestKnownDer = sampleDer;
        cached.numDrawn = 0;
      }
      return true;
    }
    return false;
  }

  // TODO this blocks until (endIdx - startIdx) many training samples have been made available by the searchThread
  void
  drawSamples(ProgramVec & oProgs, ResultDistVec & oResultDist, int startIdx, int endIdx) {
    {
      std::unique_lock queueLock(queueMutex);

    // wait until samples become available
      const int numNeeded = endIdx - startIdx;
      assert((numNeeded < queueLimit) && "deadlock waiting to happen. number of required samples is below queue limit (TODO)");

      clock_t stallTime = 0;
      if (trainingQueue.size() < numNeeded) {
        clock_t startStallPull = clock();
        consumerCV.wait(queueLock, [this, numNeeded]() { return trainingQueue.size() >= numNeeded; });
        clock_t endStallPull = clock();
        stallTime = (endStallPull - startStallPull);
      }
      queueStats.addPull(stallTime);

    // fetch samples
      for (int i = startIdx; i < endIdx; ++i) {
        auto sample = trainingQueue.front();
        trainingQueue.pop();

        oProgs[i] = sample.P;
        oResultDist[i] = sample.resultDist;

      }
    }

    producerCV.notify_all();
    // TODO draw samples from the training queue
#if 0
    assert((endIdx - startIdx) <= sampleMap.size());

  // ordered sample positions
    // samples stored in a map with linear random access complexity (order sample indices -> scan once through map)
    std::vector<int> indices;
    std::set<int> drawnIndices;
    for (int i = startIdx; i < endIdx; ++i) {
      // draw fresh sample
      int sampleIdx;
      do {
        sampleIdx = elemRand(randGen()) % sampleMap.size();
      } while (!drawnIndices.insert(sampleIdx).second);
    }
    std::sort(indices.begin(), indices.end());

    // scan through sampleMap and drawn samples
    auto it = sampleMap.begin();
    int pos = 0; // iterator offset into map
    int i = 0;
    while (i + startIdx < endIdx) {
      assert(indices[i] >= pos);
      // advance to next sample position
      if (indices[i] > pos) {
        std::advance(it, indices[i] - pos);
        pos = indices[i];
      }

      // take sample
      auto & sample = it->second;
      oProgs[startIdx + i] = it->first;
      oDerVec[startIdx + i] = sample.bestKnownDer;
      sample.numDrawn++;
      ++i;
    }
#endif
  }
};





static DerivationVec FilterBest(DerivationVec A, DerivationVec B) {
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

static int CountStops(const DerivationVec &derVec) {
  int a = 0;
  for (const auto &der : derVec)
    a += (der.shortestDerivation == 0);
  return a;
}

// number of simulation batches
APO::APO(const std::string &taskFile, const std::string &_cpPrefix)
    : modelConfig("model.conf")
    , rewritePairs(BuildRewritePairs())
    , ruleBook(modelConfig, rewritePairs)
    , model("build/rdn", modelConfig, ruleBook.num_Rules())
    , cpPrefix(_cpPrefix), montOpt(ruleBook, model), rpg(ruleBook, modelConfig.num_Params)
    , expMut(ruleBook)
    , numFinished(0)
{
  std::cerr << "Loading task file " << taskFile << "\n";

  Parser task(taskFile);
  // random program options
  taskName = task.get_or_fail<std::string>(
      "name"); // 3; // minimal progrm stub len (excluding params and return)

  numSamples = task.get_or_fail<int>("numSamples"); // number of batch programs
  assert(numSamples > 0);
  minStubLen = task.get_or_fail<int>("minStubLen"); // minimal stub len
  maxStubLen = task.get_or_fail<int>("maxStubLen"); // maximal stub len (exluding params and return)
  minMutations = task.get_or_fail<int>( "minMutations"); // minimal number of mutations
  maxMutations = task.get_or_fail<int>( "maxMutations"); // maximal number of mutations

  // mc search options
  extraExplorationDepth = task.get_or_fail<int>("extraExplorationDepth"); // additional derivation steps beyond known number of mutations
  maxExplorationDepth = task.get_or_fail<int>( "maxExplorationDepth"); // maxMutations + 1; // best-effort search depth
  pRandom = task.get_or_fail<double>( "pRandom"); // 1.0; // probability of ignoring the model for inference
  numOptRounds = task.get_or_fail<int>( "numOptRounds"); // 50; // number of optimization retries
  numEvalOptRounds = task.get_or_fail<int>( "numEvalOptRounds"); // eval opt rounds used in evaluation

  logRate = task.get_or_fail<int>( "logRate"); // 10; // number of round followed by an evaluation
  numRounds = task.get_or_fail<size_t>( "numRounds"); // 10; // number of round followed by an evaluation
  racketStartRound = task.get_or_fail<size_t>( "racketStartRound"); // 10; // number of round followed by an evaluation

  saveCheckpoints = task.get_or_fail<int>("saveModel") != 0; // save model checkpoints at @logRate

  if (saveCheckpoints) {
    std::cerr << "Saving checkpoints to prefix " << cpPrefix << "\n";
  }

  // initialize thread safe random number generators
  InitRandom();
}

inline void
APO::generatePrograms(int numSamples, std::function<void(ProgramPtr P, int numMutations)> handleFunc) {
  std::uniform_int_distribution<int> mutRand(minMutations, maxMutations);
  std::uniform_int_distribution<int> stubRand(minStubLen, maxStubLen);

  for (int i = 0; i < numSamples; ++i) {
    ProgramPtr P = nullptr;
    int mutSteps = 0;
    do {
      int stubLen = stubRand(randGen());
      mutSteps = mutRand(randGen());
      P.reset(rpg.generate(stubLen));

      assert(P->size() <= modelConfig.prog_length);
      expMut.mutate(*P, mutSteps, pGenExpand); // mutate at least once
    } while (P->size() > modelConfig.prog_length);

    handleFunc(P, mutSteps);
  }
}

void APO::generatePrograms(ProgramVec &progVec, IntVec & maxDerVec, int startIdx,
                           int endIdx) {
  std::uniform_int_distribution<int> mutRand(minMutations, maxMutations);
  std::uniform_int_distribution<int> stubRand(minStubLen, maxStubLen);

  int i = startIdx;
  generatePrograms(endIdx - startIdx, [this, &progVec, &maxDerVec, &i](ProgramPtr P, int numMutations) {
    maxDerVec[i] = std::min(numMutations + extraExplorationDepth, maxExplorationDepth);
    progVec[i] = ProgramPtr(P);
    ++i;
  });
}



void APO::train() {
// set-up training server
  SampleServer server("server.conf");

// evaluation dataset
  const int numEvalSamples = std::max<int>(4096, modelConfig.train_batch_size * 32);
  std::cerr << "numEvalSamples = " << numEvalSamples << "\n";

  // hold-out evaluation set
  std::cerr << "-- Buildling evaluation set (" << numEvalSamples << " samples, "
            << numEvalOptRounds << " optRounds) --\n";
  ProgramVec evalProgs(numEvalSamples, nullptr);
  IntVec evalDistVec(numEvalSamples, 0);
  generatePrograms(evalProgs, evalDistVec, 0, evalProgs.size());
  auto refEvalDerVec = montOpt.searchDerivations(evalProgs, 1.0, evalDistVec, numEvalOptRounds, false);

  int numStops = CountStops(refEvalDerVec);
  double stopRatio = numStops / (double)refEvalDerVec.size();
  std::cerr << "Stop ratio  " << stopRatio << ".\n";


  auto bestEvalDerVec = refEvalDerVec;

// training
  assert(minStubLen > 0 && "can not generate program within constraints");

  std::atomic<bool> keepRunning = true;

  std::mutex cpuMutex; // to co-ordinate multi-threaded processing on the GPU (eg searchThread and evaluation rounds on the trainThread)

  // model training - repeatedly draw samples from SampleServer and submit to device for training
  std::thread
  trainThread([this, &keepRunning, &server, &evalProgs, &evalDistVec, &refEvalDerVec, &bestEvalDerVec, &cpuMutex] {
    const int dotStep = logRate / 10;

    // Fetch initial programs
    ProgramVec progVec(numSamples, nullptr);
    ResultDistVec refResults(numSamples);

    Task trainTask;
    while (keepRunning.load()) {
      clock_t roundTotal = 0;
      clock_t derTotal = 0;
      size_t numTimedRounds = 0;

      std::cerr << "\n-- Training --\n";
      for (size_t g = 0; g < numRounds; ++g) {

    // evaluation round logic
        bool loggedRound = (g % logRate == 0);
        if (loggedRound) {
          std::unique_lock lock(cpuMutex); // drop the lock every now and then..

        // dump some statistics
          auto mlStats = model.query_stats();
          std::cerr << "\n- Round " << g << " (";
          mlStats.print(std::cerr);

          std::cerr << ") ";
          auto queueStats = server.resetQueueStats();
          queueStats.print(std::cerr) << " -\n";

          // print MCTS statistics
          montOpt.stats.print(std::cerr) << "\n";
          montOpt.stats = MonteCarloOptimizer::Stats();

          auto greedyDerVecs = montOpt.greedyDerivation(evalProgs, evalDistVec);

          DerStats greedyStats = ScoreDerivations(refEvalDerVec, greedyDerVecs.greedyVec);
          std::cerr << "\tGreedy (STOP) "; greedyStats.print(std::cerr); // apply most-likely action, respect STOP

          DerStats bestGreedyStats = ScoreDerivations(refEvalDerVec, greedyDerVecs.bestVec);
          std::cerr << "\tGreedy (best) "; bestGreedyStats.print(std::cerr); // same as greedy but report best program along trajectory to STOP

#if 0
          // TODO too expensive
          // model-guided sampling (best of randomly sampled trajectory)
          const int guidedSamples = 4; // number of random trajectories to sample
          auto guidedEvalDerVec = montOpt.searchDerivations(evalProgs, 0.0, evalDistVec, guidedSamples, false);

          DerStats guidedStats = ScoreDerivations(refEvalDerVec, guidedEvalDerVec);
          std::cerr << "\tSampled       "; guidedStats.print(std::cerr); // best of 4 random trajectories, ignore STOP
          bestEvalDerVec = FilterBest(bestEvalDerVec, guidedEvalDerVec);
#endif

          // improve best-known solution on the go
          bestEvalDerVec = FilterBest(bestEvalDerVec, greedyDerVecs.greedyVec);
          bestEvalDerVec = FilterBest(bestEvalDerVec, greedyDerVecs.bestVec);
          DerStats bestStats = ScoreDerivations(refEvalDerVec, bestEvalDerVec);
          std::cerr << "\tIncumbent     ";
          bestStats.print(
              std::cerr); // best of all sampling strategies (improving over time)

          // store model
          if (saveCheckpoints) {
            std::stringstream ss;
            ss << cpPrefix << "/" << taskName << "-" << g << ".cp";
            model.saveCheckpoint(ss.str());
          }

        } else {
          if (g % dotStep == 0) {
            std::cerr << ".";
          }
        }

  // actual round logic
        clock_t startRound = clock();

        auto endRound = clock();
        // statistics
        roundTotal += (endRound - startRound);
        numTimedRounds++;

        // fetch a new batch
        server.drawSamples(progVec, refResults, 0, numSamples);

        // train model (progVec, refResults)
        Model::Losses L;
        if (trainTask.joinable()) trainTask.join(); // TODO move this into the ml.cpp
        trainTask = model.train_dist(progVec, refResults, loggedRound ? &L : nullptr);

        if (loggedRound) {
        // compute number of stops in reference result
          int numStops = 0;
          for (auto & resultDist : refResults) {
            numStops += (resultDist.stopDist > 0.5f ? 1 : 0); // 0.0 or 1.0 in reference results
          }
          float dropOutRate = numStops / (double) refResults.size();

          trainTask.join();
          std::cerr << "\t";
          L.print(std::cerr) << ". Stop drop out=" << dropOutRate << "\n";
        }

      } // rounds
    } // while keepGoing
  });


  // MCTS search thread - find shortest derivations to best programs, register findings with SampleServer
  std::thread
  searchThread([this, &keepRunning, &server, &cpuMutex]{

    // compute all one-step derivations
    std::vector<std::pair<int, Action>> rewrites;
    ProgramVec nextProgs;
    IntVec nextMaxDistVec;
    const int preAllocFactor = 16;
    rewrites.reserve(preAllocFactor * numSamples);
    nextProgs.reserve(preAllocFactor * numSamples);
    nextMaxDistVec.reserve(preAllocFactor * numSamples);
    clock_t derTotal = 0;
    size_t numFinished = 0;

    // generate initial programs
    ProgramVec progVec(numSamples, nullptr);
    IntVec maxDistVec(progVec.size(), 0);
    generatePrograms(progVec, maxDistVec, 0, numSamples);

    while (keepRunning.load()) {
    // queue all programs that are reachable by a single move
      // (progVec, maxDistVec) -> (nextProgs, rewrites, nextMaxDistVec)
      // #pragma omp parallel for ordered
      for (int t = 0; t < progVec.size(); ++t) {
        for (int r = 0; r < ruleBook.num_Rules(); ++r) {
          const int progSize= progVec[t]->size();
          for (int pc = 0; pc + 1 < progSize ; ++pc) {
            Action act{pc, r};

            auto *clonedProg = new Program(*progVec[t]);
            if (!expMut.tryApply(*clonedProg, act)) {
              // TODO clone after match (or render into copy)
              delete clonedProg;
              continue;
            }

            // compact list of programs resulting from a single action
            // #pragma omp ordered
            {
              int remainingSteps = std::max(1, maxDistVec[t] - 1);
              nextMaxDistVec.push_back(remainingSteps);
              nextProgs.emplace_back(clonedProg);
              rewrites.emplace_back(t, act);
            }
          }
        }
      }

      clock_t startDer = clock();
      // best-effort search for optimal program
      // nextProgs -> refDerVec
      DerivationVec refDerVec;
      {
        std::unique_lock lock(cpuMutex); // acquire lock for most CPU-heavy task
        refDerVec = montOpt.searchDerivations(nextProgs, pRandom, nextMaxDistVec, numOptRounds, false);
      }
      // NOTE all derivations in @refDerVec are from programs in @nextProgs which are one step closer to the optimum than their source programs in @progVec

#if 0
      // TODO enable model inference
      if (g >= racketStartRound) {
        // model-driven search
        auto guidedDerVec = montOpt.searchDerivations(nextProgs, 0.1, nextMaxDistVec, 4, true);
        refDerVec = FilterBest(refDerVec, guidedDerVec);
      }
#endif
      clock_t endDer = clock();
      derTotal += (endDer - startDer);

      // (rewrites, refDerVec) --> refResults
      // decode reference ResultDistVec from detected derivations
      ResultDistVec refResults;
      montOpt.populateRefResults(refResults, refDerVec, rewrites, progVec);

      // submit results to server
      server.submitResults(progVec, refResults);

    // sample moves from the reference distribution
      // (refResults, rewrites, nextProgs, nextMaxDistVec) -> progVec, maxDistVec
      int numNextProgs = 0;
      montOpt.sampleActions(refResults, rewrites, nextProgs,

      // actionHandler
        // 1. keep programs with applicable action in batch (progVec)
        // 2. tell the server about the detected derivations
        [&numNextProgs, &nextProgs, &progVec, &maxDistVec, &nextMaxDistVec, &refDerVec, &server](int sampleIdx, int rewriteIdx) {
            assert(sampleIdx >= numNextProgs);

            // register derivation info
            auto & startProg = progVec[sampleIdx];
            auto bestDer = refDerVec[rewriteIdx];

            // submit detected derivation
            server.submitDerivation(progVec[sampleIdx], bestDer);

            // insert program after action into queue
            int progIdx = numNextProgs++;
            progVec[progIdx] = nextProgs[rewriteIdx];
            maxDistVec[progIdx] = std::max(1, nextMaxDistVec[rewriteIdx] - 1); // carry on unless there is an explicit STOP

            return true;
        },

      // stopHandler
        // 1. tell the server about the detected stop derivation
        // 2. stop STOP-ped program from batch in any way (implicit)
        [&server, &progVec](int sampleIdx, StopReason reason) {
          if (reason == StopReason::Choice) {
            auto & stopProg = progVec[sampleIdx];
            server.submitDerivation(stopProg, Derivation(*stopProg));
          }

          return true;
        }
      );


      double dropOutRate = 1.0 - numNextProgs / (double) numSamples;
      numFinished += (numSamples - numNextProgs);

      // fill up dropped slots with new programs
      generatePrograms(progVec, maxDistVec, numNextProgs, numSamples);
    }
  });

  searchThread.detach();
  trainThread.join();

  keepRunning.store(false); // shutdown all workers
}


} // namespace apo
