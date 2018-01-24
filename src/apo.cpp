#include "apo/apo.h"

namespace apo {

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
    , cpPrefix(_cpPrefix), montOpt(ruleBook, model), rpg(ruleBook, modelConfig.num_Params),
      expMut(rewritePairs, pExpand)
{
  std::cerr << "Loading task file " << taskFile << "\n";

  Parser task(taskFile);
  // random program options
  taskName = task.get_or_fail<std::string>(
      "name"); // 3; // minimal progrm stub len (excluding params and return)

  numSamples = task.get_or_fail<int>("numSamples"); // 3; // minimal progrm stub
                                                    // len (excluding params and
                                                    // return
  minStubLen = task.get_or_fail<int>("minStubLen"); // 3; // minimal progrm stub
                                                    // len (excluding params and
                                                    // return)
  maxStubLen =
      task.get_or_fail<int>("maxStubLen"); // 4; // maximal program
                                           // stub len (excluding params
                                           // and return)
  minMutations = task.get_or_fail<int>(
      "minMutations"); // 1; // max number of program mutations
  maxMutations = task.get_or_fail<int>(
      "maxMutations"); // 1; // max number of program mutations

  // mc search options
  maxExplorationDepth = task.get_or_fail<int>(
      "maxExplorationDepth"); // maxMutations + 1; // best-effort search depth
  pRandom = task.get_or_fail<double>(
      "pRandom"); // 1.0; // probability of ignoring the model for inference
  numOptRounds = task.get_or_fail<int>(
      "numOptRounds"); // 50; // number of optimization retries

  numEvalOptRounds = task.get_or_fail<int>(
      "numEvalOptRounds"); // eval opt rounds used in evaluation

  logRate = task.get_or_fail<int>(
      "logRate"); // 10; // number of round followed by an evaluation
  numRounds = task.get_or_fail<size_t>(
      "numRounds"); // 10; // number of round followed by an evaluation
  racketStartRound = task.get_or_fail<size_t>(
      "racketStartRound"); // 10; // number of round followed by an evaluation

  saveCheckpoints = task.get_or_fail<int>("saveModel") !=
                    0; // save model checkpoints at @logRate

  if (saveCheckpoints) {
    std::cerr << "Saving checkpoints to prefix " << cpPrefix << "\n";
  }

  // initialize thread safe random number generators
  InitRandom();
}

void APO::generatePrograms(ProgramVec &progVec, size_t startIdx,
                           size_t endIdx) {
  std::uniform_int_distribution<int> mutRand(minMutations, maxMutations);
  std::uniform_int_distribution<int> stubRand(minStubLen, maxStubLen);

  for (int i = startIdx; i < endIdx; ++i) {
    std::shared_ptr<Program> P = nullptr;
    do {
      int stubLen = stubRand(randGen());
      int mutSteps = mutRand(randGen());
      P.reset(rpg.generate(stubLen));

      assert(P->size() <= modelConfig.prog_length);
      expMut.mutate(*P, mutSteps); // mutate at least once
    } while (P->size() > modelConfig.prog_length);

    progVec[i] = std::shared_ptr<Program>(P);
  }
}

void APO::train() {
  const int numEvalSamples = std::min<int>(4096, modelConfig.train_batch_size * 32);
  std::cerr << "numEvalSamples = " << numEvalSamples << "\n";

  // hold-out evaluation set
  std::cerr << "-- Buildling eval set (" << numEvalSamples << " samples, "
            << numEvalOptRounds << " optRounds) --\n";
  ProgramVec evalProgs(numEvalSamples, nullptr);
  generatePrograms(evalProgs, 0, evalProgs.size());
  auto refEvalDerVec = montOpt.searchDerivations(
      evalProgs, 1.0, maxExplorationDepth, numEvalOptRounds, false);

  int numStops = CountStops(refEvalDerVec);
  double stopRatio = numStops / (double)refEvalDerVec.size();
  std::cerr << "Stop ratio  " << stopRatio << ".\n";

  auto bestEvalDerVec = refEvalDerVec;

  // training
  assert(minStubLen > 0 && "can not generate program within constraints");

  const int dotStep = logRate / 10;

  // Seed program generator
  ProgramVec progVec(numSamples, nullptr);
  generatePrograms(progVec, 0, progVec.size());

  // TESTING
  // model.setLearningRate(0.0001); // works well

  clock_t roundTotal = 0;
  size_t numTimedRounds = 0;
  std::cerr << "\n-- Training --\n";
  for (size_t g = 0; g < numRounds; ++g) {
    bool loggedRound = (g % logRate == 0);
    if (loggedRound) {
      auto stats = model.query_stats();
      std::cerr << "\n- Round " << g << " (";
      stats.print(std::cerr);
      if (g == 0) {
        std::cerr << ") -\n";
      } else {
        // report round timing statistics
        double avgRoundTime =
            (roundTotal / (double)numTimedRounds) / CLOCKS_PER_SEC;
        std::cerr << ", avgRoundTime=" << avgRoundTime << " s ) -\n";
        roundTotal = 0;
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
          evalProgs, 0.0, maxExplorationDepth, guidedSamples, false);

      // greedy (most likely action)
      auto greedyDerVecs =
          montOpt.greedyDerivation(evalProgs, maxExplorationDepth);

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

    clock_t startRound = clock();

    // compute all one-step derivations
    std::vector<std::pair<int, RewriteAction>> rewrites;
    ProgramVec nextProgs;
    const int preAllocFactor = 16;
    rewrites.reserve(preAllocFactor * progVec.size());
    nextProgs.reserve(preAllocFactor * progVec.size());

    // #pragma omp parallel for ordered
    for (int t = 0; t < progVec.size(); ++t) {
      for (int r = 0; r < ruleBook.num_Rules(); ++r) {
        RewriteRule rewRule = ruleBook.getRewriteRule(r);
        for (int pc = 0; pc + 1 < progVec[t]->size(); ++pc) {
          RewriteAction act(pc, rewRule.pairId, rewRule.leftToRight);

          auto *clonedProg = new Program(*progVec[t]);
          if (!expMut.tryApply(*clonedProg, act)) {
            // TODO clone after match (or render into copy)
            delete clonedProg;
            continue;
          }

          // compact list of programs resulting from a single action
          // #pragma omp ordered
          {
            nextProgs.emplace_back(clonedProg);
            rewrites.emplace_back(t, act);
          }
        }
      }
    }

    // best-effort search for optimal program
    auto refDerVec = montOpt.searchDerivations(
        nextProgs, pRandom, maxExplorationDepth, numOptRounds, false);

    if (g >= racketStartRound) {
      // model-driven search
      auto guidedDerVec = montOpt.searchDerivations(
          nextProgs, 0.1, maxExplorationDepth, 4, true);
      refDerVec = FilterBest(refDerVec, guidedDerVec);
    }

    // decode reference ResultDistVec from detected derivations
    ResultDistVec refResults;
    montOpt.populateRefResults(refResults, refDerVec, rewrites, nextProgs,
                               progVec);

    // train model
    Model::Losses L;
    Task trainThread =
        model.train_dist(progVec, refResults, loggedRound ? &L : nullptr);

    // pick an action per program and drop STOP-ped programs
    int numNextProgs =
        montOpt.sampleActions(refResults, rewrites, nextProgs, progVec);
    double dropOutRate = 1.0 - numNextProgs / (double)numSamples;

    // fill up with new programs
    generatePrograms(progVec, numNextProgs, numSamples);
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
