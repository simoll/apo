#include "apo/apo.h"

#include <random>
#include <limits>

namespace apo {

struct
SampleCache {
  std::uniform_real_distribution<float> dropRand;
  std::uniform_int_distribution<int> elemRand;

  struct CachedSample {
    int numDrawn;               // how often was this sample drawn
    Derivation bestKnownDer;    // best known derivation

    CachedSample()
    {}

    CachedSample(Derivation _bestDer)
    : numDrawn(0)
    , bestKnownDer(_bestDer)
    {}
  };

  std::map<ProgramPtr, CachedSample, deref_less<ProgramPtr>> sampleMap;

  SampleCache()
  : dropRand(0, 1)
  , elemRand(0, std::numeric_limits<int>::max())
  , sampleMap()
  {}

  // drop old samples (dropBase ^ sample.numDrawn)
#if 0
  void
  reviseSampleCache() {
    const float dropBase = 0.5;

    for (int i = 0; i < samples.size(); ) {
      float pDrop = pow(dropBase, samples[i].numDrawn);
      if (dropRand(randGen()) > pDrop) {
        samples.erase(samples.begin() + i);
      } else {
        ++i;
      }
    }
  }
#endif

  // store a new program result in the cache (return true if the cache was improved by the operation (new program or better derivation))
  bool addResult(ProgramPtr P, Derivation sampleDer) {
    auto itSample = sampleMap.find(P);
    if (itSample == sampleMap.end()) {
      sampleMap[P] = CachedSample(sampleDer);
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

  void
  drawSamples(ProgramVec & oProgs, DerivationVec & oDerVec, int startIdx, int endIdx) {
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

  cacheSize = task.get_or_fail<int>("cacheSize"); // number of samples in cache at any given time
  cacheRatio = task.get_or_fail<float>("cacheRatio"); // number of re-used derivations from cache
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
  const int numEvalSamples = std::min<int>(4096, modelConfig.train_batch_size * 32);
  std::cerr << "numEvalSamples = " << numEvalSamples << "\n";

  // hold-out evaluation set
  std::cerr << "-- Buildling eval set (" << numEvalSamples << " samples, "
            << numEvalOptRounds << " optRounds) --\n";
  ProgramVec evalProgs(numEvalSamples, nullptr);
  IntVec evalDistVec(numEvalSamples, 0);
  generatePrograms(evalProgs, evalDistVec, 0, evalProgs.size());
  auto refEvalDerVec = montOpt.searchDerivations(
      evalProgs, 1.0, evalDistVec, numEvalOptRounds, false);

  int numStops = CountStops(refEvalDerVec);
  double stopRatio = numStops / (double)refEvalDerVec.size();
  std::cerr << "Stop ratio  " << stopRatio << ".\n";

  auto bestEvalDerVec = refEvalDerVec;

  // training
  assert(minStubLen > 0 && "can not generate program within constraints");

  const int dotStep = logRate / 10;

  // Seed program generator
  ProgramVec progVec(numSamples, nullptr);
  IntVec maxDistVec(progVec.size(), 0);
  generatePrograms(progVec, maxDistVec, 0, progVec.size());

  // TESTING
  // model.setLearningRate(0.0001); // works well

  // TrainingCache sampleCache(cacheSize);

  clock_t roundTotal = 0;
  clock_t derTotal = 0;
  size_t numTimedRounds = 0;
  std::cerr << "\n-- Training --\n";
  for (size_t g = 0; g < numRounds; ++g) {

// evaluation round logic
    bool loggedRound = (g % logRate == 0);
    if (loggedRound) {
      auto stats = model.query_stats();
      std::cerr << "\n- Round " << g << " (";
      stats.print(std::cerr);
      if (g == 0) {
        std::cerr << ") -\n";
      } else {
        // report round timing statistics
        double avgRoundTime = (roundTotal / (double)numTimedRounds) / CLOCKS_PER_SEC;
        double avgDerTime = (derTotal / (double)numTimedRounds) / CLOCKS_PER_SEC;
        std::cerr << ", avgRoundTime=" << avgRoundTime << " s, avgDerTime=" << avgDerTime << " s, numFinished=" << numFinished << " ) -\n";
        roundTotal = 0;
        derTotal = 0;
        numTimedRounds = 0;
      }

      // print MCTS statistics
      montOpt.stats.print(std::cerr) << "\n";
      montOpt.stats = MonteCarloOptimizer::Stats();

      // one shot (model based)
      // auto oneShotEvalDerVec = montOpt.searchDerivations(evalProgs, 0.0,
      // maxExplorationDepth, 1, false);

      // model-guided sampling
      const int guidedSamples = 4;
      auto guidedEvalDerVec = montOpt.searchDerivations(
          evalProgs, 0.0, evalDistVec, guidedSamples, false);

      // greedy (most likely action)
      auto greedyDerVecs =
          montOpt.greedyDerivation(evalProgs, evalDistVec);

      // DerStats oneShotStats = ScoreDerivations(refEvalDerVec,
      // oneShotEvalDerVec);
      DerStats greedyStats =
          ScoreDerivations(refEvalDerVec, greedyDerVecs.greedyVec);
      std::cerr << "\tGreedy (STOP) ";
      greedyStats.print(std::cerr); // apply most-likely action, respect STOP

      DerStats bestGreedyStats =
          ScoreDerivations(refEvalDerVec, greedyDerVecs.bestVec);
      std::cerr << "\tGreedy (best) ";
      bestGreedyStats.print(std::cerr); // one random trajectory, ignore S

      DerStats guidedStats = ScoreDerivations(refEvalDerVec, guidedEvalDerVec);
      std::cerr << "\tSampled       ";
      guidedStats.print(
          std::cerr); // best of 4 random trajectories, ignore STOP

      // improve best-known solution on the go
      bestEvalDerVec = FilterBest(bestEvalDerVec, guidedEvalDerVec);
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
    // compute all one-step derivations
    std::vector<std::pair<int, Action>> rewrites;
    ProgramVec nextProgs;
    IntVec nextMaxDistVec;
    const int preAllocFactor = 16;
    rewrites.reserve(preAllocFactor * progVec.size());
    nextProgs.reserve(preAllocFactor * progVec.size());
    nextMaxDistVec.reserve(preAllocFactor * progVec.size());

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
    auto refDerVec = montOpt.searchDerivations(
        nextProgs, pRandom, nextMaxDistVec, numOptRounds, false);

    if (g >= racketStartRound) {
      // model-driven search
      auto guidedDerVec = montOpt.searchDerivations(
          nextProgs, 0.1, nextMaxDistVec, 4, true);
      refDerVec = FilterBest(refDerVec, guidedDerVec);
    }
    clock_t endDer = clock();
    derTotal += (endDer - startDer);

    // decode reference ResultDistVec from detected derivations
    ResultDistVec refResults;
    montOpt.populateRefResults(refResults, refDerVec, rewrites, nextProgs, progVec);

    // train model
    Model::Losses L;
    Task trainThread =
        model.train_dist(progVec, refResults, loggedRound ? &L : nullptr);

    // pick an action per program and drop STOP-ped programs
    int numNextProgs =
        montOpt.sampleActions(refResults, rewrites, nextProgs, nextMaxDistVec, progVec, maxDistVec);
    double dropOutRate = 1.0 - numNextProgs / (double) numSamples;

    numFinished += (numSamples - numNextProgs);

    // fill up with new programs
    generatePrograms(progVec, maxDistVec, numNextProgs, numSamples);

    auto endRound = clock();
    // statistics
    roundTotal += (endRound - startRound);
    numTimedRounds++;

    if (loggedRound) {
      trainThread.join();
      std::cerr << "\t";
      L.print(std::cerr) << ". Stop drop out=" << dropOutRate << "\n";
    } else {
      trainThread.detach();
    }
  }
}


} // namespace apo
