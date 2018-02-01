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

  int queueLimit; // maximal # training samples in queue
  int derCacheLimit; // maximal # derivations in cache

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
  std::unordered_map<ProgramPtr, CachedDerivation, ProgramPtrHasher, ProgramPtrEqual> sampleMap;

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
  struct ServerStats {
  // queue stats
    int numQueuePush; // number of ::submitResults to queue
    size_t numWaitlessPush; // total time spent waiting (::submtResults)
    size_t totalStallPush; // total time spent waiting (::submtResults)
    int numQueuePull; // number of ::drawSamples from queue
    int numWaitlessPull;
    size_t totalStallPull; // total time spent waiting for data (::drawSamples)

  // derivation & cache statistics
    int derCacheSize; // size of derivation cache (at ::resetServerStats)
    int numCacheQueries; // ::getDerivation queries
    int numCacheHits; // cache hits in ::getDerivation
    int numImprovedDer; // amount of improved results in ::submitDerivation
    int numAddedDer; // number of derivation added in this round
    int numSeenBeforeDer; // amount of improved results in ::submitDerivation

  // search thread statistics
    clock_t derTotalTime;
    clock_t sampleTotalTime;
    clock_t replayTotalTime;
    int numSearchRounds;

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

    double getCacheHitRate() const { return numCacheQueries > 0 ?  (numCacheHits / (double) numCacheQueries) : 1; }

    double getAvgDerTime() const { return numSearchRounds > 0 ? ((derTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }
    double getAvgSampleTime() const { return numSearchRounds > 0 ? ((sampleTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }
    double getAvgReplaySampleTime() const { return numSearchRounds > 0 ? ((replayTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }

    std::ostream&
    print(std::ostream & out) const {
      out << "Server stats:\n"
          << "\tQueue  (avgPushStall=" << getPushStall() << " s, fastPushRatio=" <<  getWaitlessPushRatio() << ", avgPullStall=" << getPullStall() << "s, fastPullRatio=" << getWaitlessPullRatio() << ")\n"
          << "\tCache  (derCacheSize=" << derCacheSize << ", hitRate=" << getCacheHitRate() << ", numImproved=" << numImprovedDer << ", numAdded=" << numAddedDer << ", seenBefore=" << numSeenBeforeDer << ")\n"
          << "\tSearch (avgDerTime=" << getAvgDerTime() << "s , avgSampleTime=" << getAvgSampleTime() << "s , avgReplaySampleTime=" << getAvgReplaySampleTime() << ", numSearchRounds=" << numSearchRounds << ")\n";
      return out;
    }

    ServerStats()
    // queue stats
    : numQueuePush(0)
    , numWaitlessPush(0)
    , totalStallPush(0)
    , numQueuePull(0)
    , numWaitlessPull(0)
    , totalStallPull(0)

    // der cache
    , numCacheQueries(0)
    , numCacheHits(0)
    , numImprovedDer(0)
    , numAddedDer(0)
    , numSeenBeforeDer(0)

    // search statistics
    , derTotalTime(0)
    , sampleTotalTime(0)
    , replayTotalTime(0)
    , numSearchRounds(0)
    {}
  };

  mutable ServerStats serverStats;

  // server.addSearchRoundStats(endDer - startDer, endSample - startSample, endReplay - startReplay)
  void
  addSearchRoundStats(clock_t deltaDer, clock_t deltaSampleActions, clock_t deltaReplay) {
    std::unique_lock lock(queueMutex);
    serverStats.numSearchRounds++;
    serverStats.derTotalTime += deltaDer;
    serverStats.sampleTotalTime += deltaSampleActions;
    serverStats.replayTotalTime += deltaReplay;
  }


  // read out current stats and reset
  ServerStats
  resetServerStats() {
    std::unique_lock lock(queueMutex);
    auto currStats = serverStats;
    serverStats = ServerStats();

    currStats.derCacheSize = sampleMap.size();
    return currStats;
  }

  SampleServer(const std::string & serverConfig)
  : dropRand(0, 1)
  , elemRand(0, std::numeric_limits<int>::max())
  , sampleMap()
  , serverStats()
  {
    Parser cfg(serverConfig);

    queueLimit = cfg.get_or_fail<int>("queueLimit"); // number of training samples in queue
    derCacheLimit = cfg.get_or_fail<int>("derivationCacheLimit"); // number of program derivations to store in the derivation cache
  }

  // submit a complete action distribution
  void
  submitResults(ProgramVec & progVec, const ResultDistVec & resDistVec) {
    // TODO append a random fraction to the replay queue
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
      serverStats.addPush(stallTime);

      // fill slots
      for (int i = 0; i < progVec.size(); ++i) {
        trainingQueue.emplace(progVec[i], resDistVec[i]);
      }
    }

    consumerCV.notify_all();
  }

  // derivation cache query
  bool
  getDerivation(ProgramPtr P, Derivation & oDer) const {
    serverStats.numCacheQueries++;

    auto itDer = sampleMap.find(P);
    if (itDer == sampleMap.end()) return false;
    serverStats.numCacheHits++;
    oDer = itDer->second.bestKnownDer;
    return true;
  }

  // store a new program result in the cache (return true if the cache was improved by the operation (new program or better derivation))
  bool
  submitDerivation(ProgramPtr P, Derivation sampleDer) {
    // TODO enfore cache size limit

    auto itSample = sampleMap.find(P);
    if (itSample == sampleMap.end()) {
      // std::cerr << "NEW!!!\n";
      serverStats.numAddedDer++;
      sampleMap[P] = CachedDerivation(sampleDer);
      return true;
    } else {
      // std::cerr << "HIT!!!\n";
      auto & cached = itSample->second;
      if (sampleDer.betterThan(cached.bestKnownDer)) {
        serverStats.numImprovedDer++;
        cached.bestKnownDer = sampleDer;
        cached.numDrawn = 0;
      } else {
        serverStats.numSeenBeforeDer++;
      }
      return true;
    }
    return false;
  }

  int
  drawReplays(ProgramVec & oProgs, IntVec & maxDerVec, int startIdx, int endIdx, int extraDerSteps, int maxDerSteps) {
    assert((endIdx - startIdx) <= sampleMap.size());

    int actualEndIdx = std::min<int>(startIdx + sampleMap.size(), endIdx);

  // ordered sample positions
    // samples stored in a map with linear random access complexity (order sample indices -> scan once through map)
    std::vector<int> indices;
    std::set<int> drawnIndices;
    for (int i = startIdx; i < actualEndIdx; ++i) {
      // draw fresh sample
      int sampleIdx;
      do {
        sampleIdx = elemRand(randGen()) % sampleMap.size();
      } while (!drawnIndices.insert(sampleIdx).second);
      indices.push_back(sampleIdx);
    }
    std::sort(indices.begin(), indices.end());

    // scan through sampleMap and drawn samples
    int pos = 0; // iterator offset into map
    auto it = sampleMap.begin();
    for(int i = 0; i + startIdx < actualEndIdx; ++i) {
      assert(indices[i] >= pos);
      // advance to next sample position
      if (indices[i] > pos) {
        std::advance(it, indices[i] - pos);
        pos = indices[i];
      }

      // take sample
      auto & sample = it->second;
      oProgs[startIdx + i] = it->first;
      maxDerVec[startIdx + i] = std::min(sample.bestKnownDer.shortestDerivation + extraDerSteps, maxDerSteps);
      sample.numDrawn++;
    }

    return actualEndIdx;
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
      serverStats.addPull(stallTime);

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

  replayRate = task.get_or_fail<double>("replayRate"); // 10; // number of round followed by an evaluation

  saveCheckpoints = task.get_or_fail<int>("saveModel") != 0; // save model checkpoints at @logRate

  if (saveCheckpoints) {
    std::cerr << "Saving checkpoints to prefix " << cpPrefix << "\n";
  }

  // initialize thread safe random number generators
  InitRandom();
}

inline void
APO::generatePrograms(int numSamples, std::function<void(ProgramPtr P, int numMutations)> &&handleFunc) {
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

          std::cerr << ") -\n";
          auto serverStats = server.resetServerStats();
          serverStats.print(std::cerr);

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
    using RewriteVec = std::vector<std::pair<int, Action>>;
    clock_t derTotal = 0;
    size_t numFinished = 0;

    // generate initial programs
    ProgramVec progVec(numSamples, nullptr);
    IntVec maxDistVec(progVec.size(), 0);
    generatePrograms(progVec, maxDistVec, 0, numSamples);

    // results to find by mcts search
    RewriteVec mctsRewrites;
    ProgramVec mctsNextProgs;
    IntVec mctsNextMaxDistVec;
    const int preAllocSize = numSamples * ruleBook.num_Rules() *  (modelConfig.prog_length / 2);
    mctsRewrites.reserve(preAllocSize);
    mctsNextProgs.reserve(preAllocSize);
    mctsNextMaxDistVec.reserve(preAllocSize);

    // cached results (known)
    IntVec cachedNextMaxDistVec;
    ProgramVec cachedNextProgs;
    RewriteVec cachedRewrites;
    cachedNextProgs.reserve(preAllocSize);
    cachedRewrites.reserve(preAllocSize);

    while (keepRunning.load()) {
      // (progVec, maxDistVec) -> (nextProgs, rewrites, nextMaxDistVec)
      // #pragma omp parallel for ordered

      DerivationVec cachedDerVec;
      cachedDerVec.reserve(preAllocSize);

    // queue all programs that are reachable by a single move
      for (int t = 0; t < progVec.size(); ++t) {
        // TODO prog itself could be a STOP candidate..
        for (int r = 0; r < ruleBook.num_Rules(); ++r) {
          const int progSize = progVec[t]->size();
          for (int pc = 0; pc + 1 < progSize ; ++pc) {
            Action act{pc, r};

            auto *clonedProg = new Program(*progVec[t]);
            if (!expMut.tryApply(*clonedProg, act)) {
              // TODO clone after match (or render into copy)
              delete clonedProg;
              continue;
            }

            int remainingSteps = std::max(1, maxDistVec[t] - 1);

            ProgramPtr actionProg(std::move(clonedProg)); // must not use @clonedProg after this point
            Derivation cachedDer;
            if (server.getDerivation(actionProg, cachedDer)) {
              // use cached result (TODO run ::searchDer with low sample rate instead to keep improving)
              cachedDerVec.push_back(cachedDer);
              cachedNextMaxDistVec.push_back(remainingSteps);
              cachedNextProgs.push_back(actionProg);
              cachedRewrites.emplace_back(t, act);

            } else {
              // full MCTS derivation path for unseen programs
              // montOpt.searchDerivations will fill in derivations for us
              mctsNextMaxDistVec.push_back(remainingSteps);
              mctsNextProgs.emplace_back(actionProg);
              mctsRewrites.emplace_back(t, act);
            }
          }
        }
      }

      clock_t startDer = clock();
      // best-effort search for optimal program
      // mctsNextProgs -> mctsDerVec
      DerivationVec mctsDerVec;
      {
        std::unique_lock lock(cpuMutex); // acquire lock for most CPU-heavy task
        mctsDerVec = montOpt.searchDerivations(mctsNextProgs, pRandom, mctsNextMaxDistVec, numOptRounds, false);
      }

      assert(mctsDerVec.size() == mctsNextMaxDistVec.size());
      assert(mctsNextMaxDistVec.size() == mctsNextProgs.size());
      assert(mctsNextProgs.size() == mctsRewrites.size());

      // concat mcts-derived results to solution vectors
      const int totalNumNextProgs = mctsNextMaxDistVec.size() + cachedNextMaxDistVec.size();
      IntVec nextMaxDistVec; nextMaxDistVec.reserve(totalNumNextProgs);
      ProgramVec nextProgs; nextProgs.reserve(totalNumNextProgs);
      RewriteVec rewrites; rewrites.reserve(totalNumNextProgs);
      DerivationVec refDerVec; refDerVec.reserve(totalNumNextProgs);

      // re-interleave the results (by their program index) (or sampleAction will fail)
      int mctsIdx = 0;
      int cachedIdx = 0;
      for (int t = 0; t < progVec.size(); ) {
        if (cachedIdx < cachedRewrites.size() && cachedRewrites[cachedIdx].first == t) {
          refDerVec.push_back(cachedDerVec[cachedIdx]);
          nextMaxDistVec.push_back(cachedNextMaxDistVec[cachedIdx]);
          nextProgs.push_back(cachedNextProgs[cachedIdx]);
          rewrites.push_back(cachedRewrites[cachedIdx]);

          cachedIdx++;
        } else if (mctsIdx < mctsRewrites.size() && mctsRewrites[mctsIdx].first == t) {
          refDerVec.push_back(mctsDerVec[mctsIdx]);
          nextMaxDistVec.push_back(mctsNextMaxDistVec[mctsIdx]);
          nextProgs.push_back(mctsNextProgs[mctsIdx]);
          rewrites.push_back(mctsRewrites[mctsIdx]);

          mctsIdx++;
        } else {
          // all results for progVec[t] have been merged, continue
          ++t;
        }
      }
      // NOTE all derivations in @refDerVec are from programs in @nextProgs which are one step closer to the optimum than their source programs in @progVec
      clock_t endDer = clock();

#if 0
      // TODO enable model inference
      if (g >= racketStartRound) {
        // model-driven search
        auto guidedDerVec = montOpt.searchDerivations(nextProgs, 0.1, nextMaxDistVec, 4, true);
        refDerVec = FilterBest(refDerVec, guidedDerVec);
      }
#endif

      // (rewrites, refDerVec) --> refResults
      // decode reference ResultDistVec from detected derivations
      ResultDistVec refResults;
      montOpt.populateRefResults(refResults, refDerVec, rewrites, progVec);

      // submit results to server
      server.submitResults(progVec, refResults);

      clock_t startSample = clock();
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
      clock_t endSample = clock();

      double dropOutRate = 1.0 - numNextProgs / (double) numSamples;
      numFinished += (numSamples - numNextProgs);

      // fill up dropped slots with new programs
      int numRefill = numSamples - numNextProgs;
      int numFreshSamples = (int) floor(numRefill * replayRate);

      // replay samples
      clock_t startReplay = clock();
      int actualEndIdx = server.drawReplays(progVec, maxDistVec, numNextProgs, numNextProgs + numFreshSamples, extraExplorationDepth, maxExplorationDepth);
      clock_t endReplay = clock();

      server.addSearchRoundStats(endDer - startDer, endSample - startSample, endReplay - startReplay);

      // generate new samples for the remainder
      generatePrograms(progVec, maxDistVec, actualEndIdx, numSamples);
    }
  });

  searchThread.detach();
  trainThread.join();

  keepRunning.store(false); // shutdown all workers
}


} // namespace apo
