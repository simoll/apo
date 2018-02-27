#ifndef APO_SAMPLESERVER_H
#define APO_SAMPLESERVER_H

#include <condition_variable>
#include <random>

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
  // std::map<ProgramPtr, CachedDerivation, deref_less<ProgramPtr>> backupMap;

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
    clock_t generateMoveTotalTime;
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

    double getAvgGenerateMoveTime() const { return numSearchRounds > 0 ? ((generateMoveTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }
    double getAvgDerTime() const { return numSearchRounds > 0 ? ((derTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }
    double getAvgSampleTime() const { return numSearchRounds > 0 ? ((sampleTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }
    double getAvgReplaySampleTime() const { return numSearchRounds > 0 ? ((replayTotalTime / (double) numSearchRounds) / CLOCKS_PER_SEC) : 0; }

    std::ostream&
    print(std::ostream & out) const {
      out << "Server: "
          << "Queue  (avgPushStall=" << getPushStall() << "s, fastPushRatio=" <<  getWaitlessPushRatio() << ", avgPullStall=" << getPullStall() << "s, fastPullRatio=" << getWaitlessPullRatio() << ")\n"
          << "\tCache  (derCacheSize=" << derCacheSize << ", hitRate=" << getCacheHitRate() << ", numImproved=" << numImprovedDer << ", numAdded=" << numAddedDer << ", seenBefore=" << numSeenBeforeDer << ")\n"
          << "\tSearch (avgGenerateMoveTime=" << getAvgGenerateMoveTime() << "s , avgDerTime=" << getAvgDerTime() << "s , avgSampleTime=" << getAvgSampleTime() << "s , avgReplaySampleTime=" << getAvgReplaySampleTime() << ", numSearchRounds=" << numSearchRounds << ")\n";
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
    , generateMoveTotalTime(0)
    , derTotalTime(0)
    , sampleTotalTime(0)
    , replayTotalTime(0)
    , numSearchRounds(0)
    {}
  };

  mutable ServerStats serverStats;

  void
  addSearchRoundStats(clock_t deltaGenerateMove, clock_t deltaDer, clock_t deltaSampleActions, clock_t deltaReplay) {
    std::unique_lock<std::mutex> lock(queueMutex);
    serverStats.numSearchRounds++;
    serverStats.generateMoveTotalTime += deltaGenerateMove;
    serverStats.derTotalTime += deltaDer;
    serverStats.sampleTotalTime += deltaSampleActions;
    serverStats.replayTotalTime += deltaReplay;
  }


  // read out current stats and reset
  ServerStats
  resetServerStats() {
    std::unique_lock<std::mutex> lock(queueMutex);
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
      std::unique_lock<std::mutex> queueLock(queueMutex);

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
#ifdef APO_ENABLE_DER_CACHE
    serverStats.numCacheQueries++;

    auto itDer = sampleMap.find(P);
    if (itDer == sampleMap.end()) return false;
    serverStats.numCacheHits++;
    oDer = itDer->second.bestKnownDer;
    return true;
#else
    return false;
#endif
  }

  bool acceptsDerivations() const { return sampleMap.size() < derCacheLimit; }
  // store a new program result in the cache (return true if the cache was improved by the operation (new program or better derivation))
  bool
  submitDerivation(const ProgramPtr & P, const Derivation & sampleDer) {
    if (sampleMap.size() > derCacheLimit) {
      return false;
    }
#ifdef APO_ENABLE_DER_CACHE
    auto itSample = sampleMap.find(P);
    if (itSample == sampleMap.end()) {
      serverStats.numAddedDer++;
      sampleMap[P] = CachedDerivation(sampleDer);
      return true;
    } else {
      Derivation & cachedDer = itSample->second.bestKnownDer;
      if (sampleDer.betterThan(cachedDer)) {
        serverStats.numImprovedDer++;
        cachedDer = sampleDer;
      } else {
        serverStats.numSeenBeforeDer++;
      }
      return true;
    }
    return false;
#endif

    return false;
  }

  int
  drawReplays(ProgramVec & oProgs, IntVec & maxDerVec, int startIdx, int endIdx, int extraDerSteps, int maxDerSteps) {
#ifdef APO_ENABLE_DER_CACHE
    int actualEndIdx = std::min<int>(startIdx + sampleMap.size(), endIdx);

  // ordered sample positions
    // samples stored in a map with linear random access complexity (order sample indices -> scan once through map)
    std::vector<int> indices(actualEndIdx - startIdx, 0);
    std::set<int> drawnIndices;
    for (int i = startIdx; i < actualEndIdx; ++i) {
      // draw fresh sample
      int sampleIdx;
      do {
        sampleIdx = elemRand(randGen()) % sampleMap.size();
      } while (!drawnIndices.insert(sampleIdx).second);
      indices[i - startIdx] = sampleIdx;
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
#else
    return startIdx; // disabled
#endif
  }

  // TODO this blocks until (endIdx - startIdx) many training samples have been made available by the searchThread
  void
  drawSamples(ProgramVec & oProgs, ResultDistVec & oResultDist, int startIdx, int endIdx) {
    {
      std::unique_lock<std::mutex> queueLock(queueMutex);

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

} // namespace apo

#endif // APO_SAMPLESERVER_H

