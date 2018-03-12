#include "apo/apo.h"
#include "apo/sampleserver.h"
#include "apo/timers.h"

#include <random>
#include <limits>
#include <thread>
#include <atomic>

namespace apo {





static int CountStops(const DerivationVec &derVec) {
  int a = 0;
  for (const auto &der : derVec)
    a += (der.shortestDerivation == 0);
  return a;
}


APO::Job::Job(const std::string taskFile, const std::string _cpPrefix)
    : cpPrefix(_cpPrefix)
{
  std::cerr << "Loading training job file " << taskFile << "\n";

  Parser task(taskFile);
  // random program options
  taskName = task.get_or_fail<std::string>(
      "name"); // 3; // minimal progrm stub len (excluding params and return)

  numSamples = task.get_or_fail<int>("numSamples"); // number of batch programs
  if (numSamples <= 0) {
    std::cerr << "Invalid numSamples " << numSamples << "\n";
    abort();
  }
  minStubLen = task.get_or_fail<int>("minStubLen"); // minimal stub len
  maxStubLen = task.get_or_fail<int>("maxStubLen"); // maximal stub len (exluding params and return)
  minMutations = task.get_or_fail<int>( "minMutations"); // minimal number of mutations
  maxMutations = task.get_or_fail<int>( "maxMutations"); // maximal number of mutations
  numShuffle = task.get_or_fail<int>( "numShuffle"); // numbers of random shuffles

  // mc search options
  extraExplorationDepth = task.get_or_fail<int>("extraExplorationDepth"); // additional derivation steps beyond known number of mutations
  maxExplorationDepth = task.get_or_fail<int>( "maxExplorationDepth"); // maxMutations + 1; // best-effort search depth
  pRandom = task.get_or_fail<double>( "pRandom"); // 1.0; // probability of ignoring the model for inference
  numOptRounds = task.get_or_fail<int>( "numOptRounds"); // 50; // number of optimization retries
  numEvalOptRounds = task.get_or_fail<int>( "numEvalOptRounds"); // eval opt rounds used in evaluation

  logRate = task.get_or_fail<int>( "logRate"); // 10; // number of round followed by an evaluation
  numRounds = task.get_or_fail<size_t>( "numRounds"); // 10; // number of round followed by an evaluation

  // reinforcement options
  reinSamples = task.get_or_fail<size_t>( "reinSamples"); // number of reinforcement samples
  reinStartRound = task.get_or_fail<size_t>( "reinStartRound"); // start round for reinforcement learning (uniform random sampling before)
  reinEndRound = task.get_or_fail<size_t>( "reinEndRound"); // round when the final reinforcement ratio shall be used
  reinStartRatio = task.get_or_fail<double>( "reinStartRatio"); // initial probability of using the model distribution
  reinEndRatio = task.get_or_fail<double>( "reinEndRatio"); // initial probability of using the model distribution

  replayRate = task.get_or_fail<double>("replayRate");  // (deprecated) number of rounds to replay

  saveCheckpoints = task.get_or_fail<int>("saveModel") != 0; // save model checkpoints at @logRate

  if (saveCheckpoints) {
    std::cerr << "Saving checkpoints to prefix " << cpPrefix << "\n";
  }
}
// number of simulation batches
APO::APO()
    : modelConfig("model.conf", "train.conf")
    , devices("devices.conf")
    , rewritePairs(BuildRewritePairs())
    , ruleBook(modelConfig, rewritePairs)
    , model("build/rdn", modelConfig, ruleBook)
    , montOpt(ruleBook, model), rpg(ruleBook, modelConfig.num_Params)
    , expMut(ruleBook)
{
  // initialize thread safe random number generators
  InitRandom();
}

inline void
APO::generatePrograms(int numSamples, const Job & task, std::function<void(ProgramPtr P, int numMutations)> &&handleFunc) {
  std::uniform_int_distribution<int> mutRand(task.minMutations, task.maxMutations);
  std::uniform_int_distribution<int> stubRand(task.minStubLen, task.maxStubLen);

  for (int i = 0; i < numSamples; ++i) {
    ProgramPtr P = nullptr;
  // apply random rules
    int mutSteps = 0;
    do {
      int stubLen = stubRand(randGen());
      mutSteps = mutRand(randGen());
      P.reset(rpg.generate_ext(stubLen));

      assert(P->size() <= modelConfig.prog_length);
      for (int j = 0; j < mutSteps; ++j) {
        expMut.mutate(*P, 1, task.pGenExpand); // mutate at least once
        expMut.shuffle(*P, task.numShuffle);
      }
    } while (P->size() > modelConfig.prog_length);

  // shuffle the instructions a little

    handleFunc(P, mutSteps);
  }
}

void APO::generatePrograms(ProgramVec &progVec, IntVec & maxDerVec, const Job & task, int startIdx, int endIdx) {
  int i = startIdx;
  generatePrograms(endIdx - startIdx, task, [this, &task, &progVec, &maxDerVec, &i](ProgramPtr P, int numMutations) {
    maxDerVec[i] = std::min(numMutations + task.extraExplorationDepth, task.maxExplorationDepth);
    progVec[i] = ProgramPtr(P);
    ++i;
  });
}



void
APO::train(const Job & task) {
// set-up training server
  SampleServer server("server.conf");

// validate configuration
  if (task.numSamples % modelConfig.train_batch_size != 0) {
    std::cerr << "Task sample size " << task.numSamples << " does not match model train_batch_size " << modelConfig.train_batch_size << "!\n";
    abort();
  }

  if (devices.getDevices("infer").empty()) {
    std::cerr << "no infer device specified in (devices.conf\n";
    abort();
  }

  if (devices.getDevices("train").empty()) {
    std::cerr << "no train device specified in devices.conf\n";
    abort();
  }

// evaluation dataset
  const int numEvalSamples = 1000; //modelConfig.infer_batch_size; //std::min<int>(1000, modelConfig.train_batch_size * 32);
  std::cerr << "numEvalSamples = " << numEvalSamples << "\n";

  // hold-out evaluation set
  std::cerr << "-- Buildling evaluation set (" << numEvalSamples << " samples, "
            << task.numEvalOptRounds << " optRounds) --\n";
  ProgramVec evalProgs(numEvalSamples, nullptr);
  IntVec evalDistVec(numEvalSamples, 0);
  generatePrograms(evalProgs, evalDistVec, task, 0, evalProgs.size());

#if 0
  // Fuse example
#if 1
  // nope
  // score 4, der 0
  auto * P = new Program(2, {
     Statement(OpCode::Add, -1, -2),
     Statement(OpCode::Add, -1, -2),
     Statement(OpCode::Mul, 0, 1),
     build_ret(2)
  });
#endif


#if 0
  // score 1, der 6
  auto * P = new Program(1, {
     build_pipe(-1),
     Statement(OpCode::Sub, 0, 0),
     build_ret(1)
  });
#endif

#if 0
  // found
  auto * P = new Program(1, {
     build_pipe(-1),
     build_pipe(-1),
     build_const(0),
     Statement(OpCode::Add, 0, 2),
     Statement(OpCode::Sub, 3, 1),
     build_ret(4)
  });
#endif

  const int PipeOpRule = ruleBook.getBuiltinID(BuiltinRules::PipeWrapOps);
  const int FuseRule = ruleBook.getBuiltinID(BuiltinRules::Fuse);

#if 0

  NodeVec holes;

  P->dump();

// 1. STEP (pipe operands)
  bool pipeOk = ruleBook.matchRule(PipeOpRule, *P, 2, holes);
  assert(pipeOk);
  ruleBook.transform(PipeOpRule, *P, 2, holes);
  P->dump();

// 2. STEP (fuse)
  #if 0
    // ???
    auto * P = new Program(2, {
       Statement(OpCode::Add, -1, -2),
       Statement(OpCode::Pipe, 0),
       Statement(OpCode::Add, -1, -2),
       Statement(OpCode::Pipe, 2),
       Statement(OpCode::Mul, 1, 3),
       build_ret(3)
    });
  #endif

  holes.clear();
  bool ok = ruleBook.matchRule(FuseRule, *P, 2, holes);
  assert(ok);
#endif

  evalProgs.emplace_back(P);
  evalDistVec.push_back(4);
#endif
  const DeviceVec & inferDevices = devices.getDevices("infer");

  auto refEvalDerVec = montOpt.searchDerivations(evalProgs, 1.0, evalDistVec, task.numEvalOptRounds, false, inferDevices);

  // std::cerr << "Ref "; refEvalDerVec[refEvalDerVec.size() - 1].dump();

  int numStops = CountStops(refEvalDerVec);
  double stopRatio = numStops / (double)refEvalDerVec.size();
  std::cerr << "Stop ratio  " << stopRatio << ".\n";

  // abort(); // BENCHMARKING
  auto bestEvalDerVec = refEvalDerVec;

// training
  assert(task.minStubLen > 0 && "can not generate program within constraints");

  std::atomic<bool> keepRunning; keepRunning.store(true);

  std::mutex cpuMutex; // to co-ordinate multi-threaded processing on the GPU (eg searchThread and evaluation rounds on the trainThread)

  // model training - repeatedly draw samples from SampleServer and submit to device for training
#if 1
  std::thread
  trainThread([this, &task, &keepRunning, &server, &evalProgs, &evalDistVec, &refEvalDerVec, &bestEvalDerVec, &cpuMutex, &inferDevices] {
    const int dotStep = task.logRate / 10;

    const auto & trainDevices = devices.getDevices("train");
    if (trainDevices.size() != 1) {
    std::cerr << "trainThread: Expected exactly one train device, was " << trainDevices.size() << ".\n";
      keepRunning.store(false);
      exit(-1);
    }
    std::string trainTower = trainDevices[0].tower;

    // Fetch initial programs
    ProgramVec progVec(task.numSamples, nullptr);
    ResultDistVec refResults(task.numSamples);

    Task trainTask;
    {
      std::cerr << "\n-- Training --\n";
      size_t g; // current opt round

      double startTime = get_wall_time();
      double roundStartTime = startTime;

      for (g = 0; g < task.numRounds; ++g) {

      // total time passed between l

    // evaluation round logic
        bool loggedRound = (g % task.logRate == 0);
        if (loggedRound) {
          double roundEndTime = get_wall_time();
          double totalRoundTime = roundEndTime - roundStartTime;

          // std::unique_lock<std::mutex> lock(cpuMutex); // drop the lock every now and then..

        // dump some statistics
          auto mlStats = model.query_stats();
          std::cerr << "\n- Round " << g << " (";
          mlStats.print(std::cerr);

          std::cerr << ") -\n";
          auto serverStats = server.resetServerStats();
          serverStats.avgTrainTime = totalRoundTime / task.logRate;
          serverStats.print(std::cerr);

          // evaluation statistics
          auto greedyDerVecs = montOpt.greedyDerivation(evalProgs, evalDistVec, inferDevices);

          std::cerr << "Eval:   ";
          DerStats greedyStats = ScoreDerivations(refEvalDerVec, greedyDerVecs.greedyVec);
          std::cerr << "Greedy (STOP) "; greedyStats.print(std::cerr); // apply most-likely action, respect STOP

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
          if (task.saveCheckpoints) {
            std::stringstream ss;
            ss << task.cpPrefix << "/" << task.taskName << "-" << g << ".cp";
            model.saveCheckpoint(ss.str());
          }

          roundStartTime = get_wall_time();
        } else {
          if (g % dotStep == 0) {
            std::cerr << ".";
          }
        }

        // fetch a new batch
        server.drawSamples(progVec, refResults, 0, task.numSamples);

        // train model (progVec, refResults)
        // Model::Losses L;
        if (trainTask.joinable()) trainTask.join(); // TODO move this into the ml.cpp
        trainTask = model.train_dist(progVec, refResults, trainTower); //, loggedRound ? &L : nullptr);

        if (loggedRound) {
        // compute number of stops in reference result
          int numStops = 0;
          for (auto & resultDist : refResults) {
            numStops += (resultDist.stopDist > 0.5f ? 1 : 0); // 0.0 or 1.0 in reference results
          }
          float dropOutRate = numStops / (double) refResults.size();

          trainTask.join();
          // std::cerr << "Loss:   "; L.print(std::cerr);
          std::cerr << ". Stop drop out=" << dropOutRate << "\n";
        }

      } // rounds

      // save final model
      std::stringstream ss;
      ss << task.cpPrefix << "/" << task.taskName << "-" << g << "-final.cp";
      model.saveCheckpoint(ss.str());
      std::cerr << "Final model stored to " << ss.str() << ".\n";

      // shutdown server
      keepRunning.store(false);
    } // while keepGoing
  });
#endif


  // MCTS search thread - find shortest derivations to best programs, register findings with SampleServer
  std::vector<std::thread> searchThreads;

  const int numSearchThreads = 3;
  for (int i = 0; i < numSearchThreads; ++i) {
    searchThreads.push_back(
    std::thread([this, &keepRunning, &server, &cpuMutex, &inferDevices, &task]{

      // compute all one-step derivations
      using RewriteVec = std::vector<std::pair<int, Action>>; // progIndex X (pc, ruleId)

      // generate initial programs
      ProgramVec progVec(task.numSamples, nullptr);
      IntVec maxDistVec(progVec.size(), 0);
      generatePrograms(progVec, maxDistVec, task, 0, task.numSamples);

  #ifdef APO_ENABLE_DER_CACHE
      // warm up the cache
      const int warmUpRounds = 100;
  #else
      const int warmUpRounds = 0;
  #endif
      int totalSearchRounds = 0;

      double startRound = get_wall_time();
      while (keepRunning.load()) {
        totalSearchRounds++; // stats

        // (progVec, maxDistVec) -> (nextProgs, rewrites, nextMaxDistVec)
        // #pragma omp parallel for ordered

        assert(progVec.size() == task.numSamples);
        assert(maxDistVec.size() == task.numSamples);

        const int preAllocSize = task.numSamples * ruleBook.num_Rules() *  (modelConfig.prog_length / 2);

        // results to find by mcts search
        RewriteVec rewrites;
        ProgramVec nextProgs;
        IntVec nextMaxDistVec;
        rewrites.reserve(preAllocSize);
        nextProgs.reserve(preAllocSize);
        nextMaxDistVec.reserve(preAllocSize);

      // queue all programs that are reachable by a single move
        double startGenerateMove = get_wall_time();
        for (int t = 0; t < progVec.size(); ++t) {
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

              ProgramPtr actionProg(clonedProg); // must not use @clonedProg after this point

              // full MCTS derivation path for unseen programs
              // montOpt.searchDerivations will fill in derivations for us
              nextMaxDistVec.push_back(remainingSteps);
              assert(actionProg);
              nextProgs.push_back(actionProg);
              rewrites.emplace_back(t, act);
            }
          }
        }
        double endGenerateMove = get_wall_time();

        // DEBUG: std::cerr << "nextProgs = " << nextProgs.size() << "\n";
        double startDer = get_wall_time();


        // reference derivation search (random / model driven)
        DerivationVec refDerVec;
        double pModel = 0.0;
        if (totalSearchRounds >= task.reinStartRound) {
  // #warning "auto consistency reinforcement"
          // reinforcement learning
  #if 0
          // reinforcement ratio
          double reinScale = std::max(0.0, std::min(1.0, (totalSearchRounds - task.reinStartRound) / (double) (task.reinEndRound - task.reinStartRound)));
          pModel = task.reinStartRatio + (task.reinEndRatio - task.reinStartRatio) * reinScale;

          // model-driven search
          // SearchPerfStats searchStats;
          const int modelRounds = task.reinSamples;
          refDerVec = montOpt.searchDerivations(nextProgs, pModel, nextMaxDistVec, modelRounds, true, inferDevices, nullptr); // &searchStats);
  #endif

          // searchStats.dump();
          // greedy results
          auto greedyRes = montOpt.greedyDerivation(nextProgs, nextMaxDistVec, inferDevices);
          refDerVec = greedyRes.bestVec; // aggressive auto-consistency

  #if 0
          // random search
          {
            std::unique_lock<std::mutex> lock(cpuMutex); // acquire lock for most CPU-heavy task
            refDerVec = montOpt.searchDerivations(nextProgs, task.pRandom, nextMaxDistVec, task.numOptRounds, false, inferDevices);
          }

          // assume best solution
          refDerVec = FilterBest(refDerVec, greedyRes.bestVec);
  #endif

        } else {
          // random search
          std::unique_lock<std::mutex> lock(cpuMutex); // acquire lock for most CPU-heavy task
          refDerVec = montOpt.searchDerivations(nextProgs, task.pRandom, nextMaxDistVec, task.numOptRounds, false, inferDevices);
        }


        // NOTE all derivations in @refDerVec are from programs in @nextProgs which are one step closer to the optimum than their source programs in @progVec
        double endDer = get_wall_time();

        // (rewrites, refDerVec) --> refResults
        // decode reference ResultDistVec from detected derivations
        ResultDistVec refResults = montOpt.populateRefResults(refDerVec, rewrites, progVec);

        double endRound = get_wall_time();
        double totalRoundTime = endRound - startRound;

        // submit results for training (-> trainThread)
        server.submitResults(progVec, refResults);
        startRound = get_wall_time();

        double startSample = get_wall_time();
      // sample moves from the reference distribution
        // (refResults, rewrites, nextProgs, nextMaxDistVec) -> progVec, maxDistVec
        int numNextProgs = 0;
        montOpt.sampleActions(refResults, rewrites, nextProgs,

        // actionHandler
          // 1. keep programs with applicable action in batch (progVec)
          // 2. tell the server about the detected derivations
          [&numNextProgs, &nextProgs, &progVec, &maxDistVec, &nextMaxDistVec, &refDerVec, &server](int sampleIdx, int rewriteIdx) {
              assert(sampleIdx >= numNextProgs);

              // insert program after action into queue
              int progIdx = numNextProgs++;
              progVec[progIdx] = nextProgs[rewriteIdx];
              maxDistVec[progIdx] = std::max(1, nextMaxDistVec[rewriteIdx] - 1); // carry on unless there is an explicit STOP

              return true;
          },

        // stopHandler
          // 1. tell the server about the detected stop derivation
          // 2. drop STOP-ped program from batch in any way (implicit)
          [&server, &progVec](int sampleIdx, StopReason reason) {
            assert((reason != StopReason::DerivationFailure) && "sampling can not fail on reference distribution!");
            assert((reason != StopReason::InvalidDist) && "sampling can not fail on reference distribution!");
            return true;
          }
        );
        double endSample = get_wall_time();

        double dropOutRate = 1.0 - numNextProgs / (double) task.numSamples;

        // fill up dropped slots with new programs
        int numRefill = task.numSamples - numNextProgs;
        int numReplayedSamples = (int) floor(numRefill * task.replayRate);

        // replay samples
        double startReplay = get_wall_time();
        int actualEndIdx = server.drawReplays(progVec, maxDistVec, numNextProgs, numNextProgs + numReplayedSamples, task.extraExplorationDepth, task.maxExplorationDepth);
        double endReplay = get_wall_time();

        // fill up with completely new progs
        generatePrograms(progVec, maxDistVec, task, actualEndIdx, task.numSamples);

        // report timing stats
        server.addSearchRoundStats(endGenerateMove - startGenerateMove, endDer - startDer, endSample - startSample, totalRoundTime, nextProgs.size(), pModel);
      }
    }));
  }

  for (auto & t : searchThreads) t.join();
  trainThread.join();

  keepRunning.store(false); // shutdown all workers
}


void
APO::loadCheckpoint(const std::string cpFile) {
  return model.loadCheckpoint(cpFile);
}

void
APO::optimize(ProgramVec & progVec, Strategy optStrat, int stepLimit) {
  const auto & inferDevices = devices.getDevices("infer");

  switch (optStrat) {
  case Strategy::BestGreedy: // best program along greedy derivation trace
  case Strategy::Greedy: { // program for which the net signalled STOP
    IntVec maxDistVec(progVec.size(), stepLimit);
    ProgramVec bestVec(1), stopVec(1);
    montOpt.greedyOptimization(bestVec, stopVec, progVec, maxDistVec, 0, progVec.size(), inferDevices[0].tower);
    if (optStrat == Strategy::BestGreedy) {
      progVec = bestVec;
    } else {
      progVec = stopVec;
    }
  } return;
  case Strategy::Random: {
    // TODO provide actual implementation
    IntVec maxDistVec(progVec.size(), stepLimit);
    const int numOptRounds = 10000;
    DerivationVec derVec = montOpt.searchDerivations(progVec, 1.0, maxDistVec, numOptRounds, false, inferDevices);
    derVec[0].print(std::cerr);
  } return;
  default: abort(); // unsupported strategy
  }
}

} // namespace apo
