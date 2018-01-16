#include "apo/ml.h"

#include "apo/parser.h"
#include "apo/extmath.h"
#include "apo/config.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

#include <limits>
#include <cassert>
#include <string>

#include <thread>

using namespace tensorflow;
// namespace tf = tensorflow;

namespace apo {

bool Model::initialized = false;
tensorflow::Session *  Model::session = nullptr;

int
Model::init_tflow() {
  if (initialized) return 0;
  initialized = true;

  Status gpuStatus = ValidateGPUMachineManager();
  if (gpuStatus.ok()) {
    // Returns the GPU machine manager singleton, creating it and
    // initializing the GPUs on the machine if needed the first time it is
    // called.  Must only be called when there is a valid GPU environment
    // in the process (e.g., ValidateGPUMachineManager() returns OK).
    perftools::gputools::Platform* platform = GPUMachineManager();
    (void) platform;
  } else {
    // running without GPUs
    // std::cout << "CPU_STAT: " << gpuStatus.ToString() << "\n";
    // return 1;
  }


   // Initialize a tensorflow session
  SessionOptions opts;
  opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.95);
  opts.config.mutable_gpu_options()->set_allow_growth(true);

  Status status = NewSession(opts, &session);

  if (!status.ok()) {
    std::cout << "NEW_SESSION: " << status.ToString() << "\n";
    return 1;
  }

  return 0;
}

Model::~Model() {
  // wait until workerThread frees lock
  std::lock_guard<std::mutex> guard(modelMutex);
}

Model::Model(const std::string & saverPrefix, const std::string & configFile, int _numRules)
: num_Rules(_numRules)
{
  init_tflow(); // make sure tensorflow session works as expected

// parse shared configuration
  {
      Parser confParser(configFile);
      prog_length = confParser.get_or_fail<int>("prog_length");
      num_Params = confParser.get_or_fail<int>("num_Params");
      max_batch_size = confParser.get_or_fail<int>("batch_size");
      max_Rules = confParser.get_or_fail<int>("max_Rules");
      batch_train_steps = confParser.get_or_fail<int>("batch_train_steps");
  }

  std::cerr << "Model (apo). prog_length=" << prog_length << ", num_Params=" << num_Params << ", max_batch_size=" << max_batch_size << ", max_Rules=" << max_Rules << ", batch_train_steps=" << batch_train_steps << "\n";

  if (num_Rules > max_Rules) {
    std::cerr << "Model does not support mode than " << max_Rules << " rules at a time! Aborting..\n";
    abort();
  }

// build Graph


#if 1
  // Read in the protobuf graph we exported
  Status status = ReadBinaryProto(Env::Default(), saverPrefix + ".meta", &graph_def);
  if (!status.ok()) {
    std::cerr << "Error reading graph definition from " << saverPrefix << ": " << status.ToString() <<"\n";
    abort();
  }

  // Add the graph to the session
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
    std::cerr << "Error creating graph: " + status.ToString() << "\n";
    abort();
  }
#endif

#if 0
  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  Status status = ReadBinaryProto(Env::Default(), graphFile, &graph_def);
  if (!status.ok()) {
    std::cout << "READ_PROTO: " << status.ToString() << "\n";
    abort();
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << "CREATE_GRAPH: " <<  status.ToString() << "\n";
    abort();
  }
 // graph-def based code
#endif

  // Setup inputs and outputs:
  std::cout << "TF: loaded graph " << saverPrefix << "\n";

  // Initialize our variables
  TF_CHECK_OK(session->Run({}, {}, {"init_op"}, nullptr));
}

void
Model::loadCheckpoint(const std::string & checkPointFile) {

  // Read weights from the saved checkpoint
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkPointFile;

  std::lock_guard<std::mutex> guard(modelMutex);
  Status status = session->Run(
          {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
          {},
          {graph_def.saver_def().restore_op_name()},
          nullptr);
  if (!status.ok()) {
    std::cerr << "Error loading checkpoint from " << checkPointFile << ": " << status.ToString() << "\n";
    abort();
  }
}

void
Model::saveCheckpoint(const std::string & checkPointFile) {

  // Read weights from the saved checkpoint
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkPointFile;

  std::lock_guard<std::mutex> guard(modelMutex);
  Status status = session->Run(
          {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
          {},
          {graph_def.saver_def().save_tensor_name()},
          nullptr);
  if (!status.ok()) {
    std::cerr << "Error saving checkpoint to " << checkPointFile << ": " << status.ToString() << "\n";
    abort();
  }
}

void
Model::shutdown() {
  if (!initialized) return;
  session->Close();
  session = nullptr;
  initialized = false;
}

int
Model::translateOperand(node_t idx) const {
  if (idx < 0) {
    return -idx;
  } else {
    return num_Params + idx;
  }
}

int
Model::encodeOperand(const Statement & stat, int opIdx) const {
  if (opIdx >= stat.num_Operands()) {
    return 0; // zero feed
  } else {
    return translateOperand(stat.getOperand(opIdx));
  }
}

int
Model::encodeOpCode(const Statement & stat) const {
  return (int) stat.oc;
}

using FeedDict = std::vector<std::pair<string, tensorflow::Tensor>>;

struct Batch {
  const Model & model;

  int size() const { return batch_size; }

  int batch_size;
  // program emcoding
  Tensor oc_feed;
  Tensor firstOp_feed;
  Tensor sndOp_feed;
  Tensor length_feed;

  // output encoding
  Tensor target_feed;
  Tensor action_feed;
  Tensor stop_feed;

  Batch(const Model & _model, int _batch_size)
  : model(_model)
  , batch_size(_batch_size)
  , oc_feed(DT_INT32, TensorShape({batch_size, model.prog_length}))
  , firstOp_feed(DT_INT32, TensorShape({batch_size, model.prog_length}))
  , sndOp_feed(DT_INT32,   TensorShape({batch_size, model.prog_length}))
  , length_feed(DT_INT32,  TensorShape({batch_size}))

  , stop_feed(DT_FLOAT,  TensorShape({batch_size}))
  , target_feed(DT_FLOAT,  TensorShape({batch_size, model.prog_length}))
  , action_feed(DT_FLOAT,  TensorShape({batch_size, model.prog_length, model.max_Rules}))
  {}

  void resize(int new_size) {
    if (new_size == batch_size) return;
    batch_size = new_size;

    oc_feed = Tensor(DT_INT32, TensorShape({batch_size, model.prog_length}));
    firstOp_feed = Tensor(DT_INT32, TensorShape({batch_size, model.prog_length}));
    sndOp_feed = Tensor(DT_INT32,   TensorShape({batch_size, model.prog_length}));
    length_feed = Tensor(DT_INT32,  TensorShape({batch_size}));

    stop_feed = Tensor(DT_FLOAT,  TensorShape({batch_size}));
    target_feed = Tensor(DT_FLOAT,  TensorShape({batch_size, model.prog_length}));
    action_feed = Tensor(DT_FLOAT,  TensorShape({batch_size, model.prog_length, model.max_Rules}));
  }

  void encode_Program(int batch_id, const Program & prog) {
    // assert(0 <= batch_id && batch_id < model.batch_size);
    auto oc_Mapped = oc_feed.tensor<int, 2>();
    auto firstOp_Mapped = firstOp_feed.tensor<int, 2>();
    auto sndOp_Mapped = sndOp_feed.tensor<int, 2>();
    auto length_Mapped = length_feed.tensor<int, 1>();

    assert((prog.size() <= model.prog_length) && "program size exceeds model limits");

    for (int t = 0; t < prog.size(); ++t) {
      oc_Mapped(batch_id, t) = model.encodeOpCode(prog.code[t]);
      firstOp_Mapped(batch_id, t) = model.encodeOperand(prog.code[t], 0);
      sndOp_Mapped(batch_id, t) = model.encodeOperand(prog.code[t], 1);
    }
    for (int t = prog.size(); t < model.prog_length; ++t) {
      oc_Mapped(batch_id, t) = 0;
      firstOp_Mapped(batch_id, t) = 0;
      sndOp_Mapped(batch_id, t) = 0;
    }

    length_Mapped(batch_id) = prog.size();
  }

  void encode_Result(int batch_id, const ResultDist & result) {
    stop_feed.tensor<float, 1>()(batch_id) = result.stopDist;

    // std::cerr << "S " << result.stopDist << "\n";
    auto target_Mapped = target_feed.tensor<float, 2>();
    auto action_Mapped = action_feed.tensor<float, 3>();
    for (int t = 0; t < model.prog_length; ++t) {
      double pTarget = 0.0;
      for (int r = 0; r < model.num_Rules; ++r) {
        pTarget += result.actionDist[t * model.num_Rules + r];
      }

      // std::cerr << "T " << batch_id << " " << t << "  :  " << pTarget << "\n";
      target_Mapped(batch_id, t) = pTarget;
      for (int r = 0; r < model.max_Rules; ++r) {
        if (r < model.num_Rules && (pTarget > 0.0)) {
          auto pAction = result.actionDist[t * model.num_Rules + r] / pTarget;
          // std::cerr << "B " << batch_id << " " << t << " " << r << "  :  " << pAction << "\n";
          action_Mapped(batch_id, t, r) = pAction;
        } else {
          action_Mapped(batch_id, t, r) = 0.0;
        }
      }
    }
  }

  FeedDict
  buildFeed() {
    FeedDict dict = {
      {"oc_data", oc_feed},
      {"firstOp_data", firstOp_feed},
      {"sndOp_data", sndOp_feed},
      {"length_data", length_feed},
      {"stop_in", stop_feed},
      {"target_in", target_feed},
      {"action_in", action_feed}
    };
    return dict;
  }
};

// train model on a batch of programs (returns loss)
std::thread
Model::train_dist(const ProgramVec& progs, const ResultDistVec& results, Losses * oLosses) {
  int num_Samples = progs.size();
  assert(results.size() == num_Samples);
  const int batch_size = max_batch_size;
  assert((num_Samples % batch_size == 0) && "TODO implement varying sized training");

  std::vector<Batch> * batchVec = new std::vector<Batch>();

  // encode programs in current thread
  for (int s = 0; s + batch_size - 1 < num_Samples; s += batch_size) {
    Batch batch(*this, max_batch_size);
    for (int i = 0; i < batch_size; ++i) {
      const Program & P = *progs[s + i];
      batch.encode_Program(i, P);
      batch.encode_Result(i, results[s + i]);
    }
    batchVec->push_back(batch);
  }

  // synchronize with pending training session
  std::thread workerThread([this, batchVec, oLosses, num_Samples]{
    std::lock_guard<std::mutex> guard(modelMutex);
    Losses L{0.0, 0.0, 0.0};

    for (Batch & batch : *batchVec) {
      // The session will initialize the outputs
      std::vector<tensorflow::Tensor> outputs;

      // std::cout << " Training on batch " << s << "\n";
      for (int i = 0; i < batch_train_steps; ++i) {
        outputs.clear();
        TF_CHECK_OK( session->Run(batch.buildFeed(), {}, {"train_dist_op"}, &outputs) );
        // summary, _ = sess.run([merged, train_op], feed_dict=feed_dict())
        // writer.add_summary(summary, i)
      }

      if (oLosses) {
        TF_CHECK_OK( session->Run(batch.buildFeed(), {"mean_stop_loss", "mean_target_loss", "mean_action_loss"}, {}, &outputs) );
        float pStopLoss = outputs[0].scalar<float>()(0);
        float pTargetLoss = outputs[1].scalar<float>()(0);
        float pActionLoss = outputs[2].scalar<float>()(0);

        // std::cout << loss_out << "\n";
        L.stopLoss += (double) pStopLoss;
        L.targetLoss += (double) pTargetLoss;
        L.actionLoss += (double) pActionLoss;
      }
    }

    if (oLosses) {
      double numBatches = num_Samples / (double) max_batch_size;
      oLosses->stopLoss = L.stopLoss / numBatches;
      oLosses->targetLoss = L.targetLoss / numBatches;
      oLosses->actionLoss = L.actionLoss / numBatches;
    }

    delete batchVec;
  });

  return workerThread;
}

void
Model::flush() {
  std::lock_guard<std::mutex> guard(modelMutex);
}

#if 0
double
Model::train(const ProgramVec& progs, const std::vector<Result>& results, int num_steps) {
  int num_Samples = progs.size();
  assert(results.size() == num_Samples);

  double avgCorrect = 0.0;
  Batch batch(*this);

  for (int s = 0; s + batch_size - 1 < num_Samples; s += batch_size) {
    for (int i = 0; i < batch_size; ++i) {
      const Program & P = *progs[s + i];
      batch.encode_Program(i, P);
      batch.encode_Result(i, results[s + i]);
    }

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    // std::cout << " Training on batch " << s << "\n";
    for (int i = 0; i < num_steps; ++i) {
      outputs.clear();
      TF_CHECK_OK( session->Run(batch.buildFeed(), {}, {"train_op"}, &outputs) );
      // summary, _ = sess.run([merged, train_op], feed_dict=feed_dict())
      // writer.add_summary(summary, i)
    }

    TF_CHECK_OK( session->Run(batch.buildFeed(), {"pCorrect_op"}, {}, &outputs) );
    // writer.add_summary(summary, i)
    auto pCorrect = outputs[0].scalar<float>()(0);
    // std::cout << loss_out << "\n";
    avgCorrect += (double) pCorrect;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  // std::cout << output_c() << "\n"; // 30

  int numBatches = num_Samples / batch_size;
  return avgCorrect / (double) numBatches;
}
#endif

std::thread
Model::infer_dist(ResultDistVec & oResultDistVec, const ProgramVec& progs, size_t startIdx, size_t endIdx) {
  Program emptyP(num_Params, {}); // the empty program

  auto batchVec = new std::vector<Batch>();
  for (int s = startIdx; s < endIdx; s += max_batch_size) {
    bool allEmpty = true;

    // detect remainder batch
    int batch_size = std::min<int>(max_batch_size, endIdx - s);
    Batch batch(*this, batch_size);

    std::vector<bool> skippedProgVec(progs.size(), false);
    for (int i = 0; i < batch_size; ++i) {
      const Program & P = *progs[s + i];
      if (P.size() > prog_length) {
        batch.encode_Program(i, emptyP);
        skippedProgVec[i] = true;
      } else {
        allEmpty = false;
        batch.encode_Program(i, P);
      }
    }

    if (allEmpty) {
      // early continue
      for (int i = startIdx; i < endIdx; ++i) {
        oResultDistVec[i] = createResultDist();
      }
      continue;
    }

    batchVec->push_back(batch);
  }

  auto workerThread = std::thread([this, batchVec, &oResultDistVec, startIdx]{
    std::lock_guard<std::mutex> guard(modelMutex);

    for (auto & batch : *batchVec) {
      // The session will initialize the outputs
      std::vector<tensorflow::Tensor> outputs;
      TF_CHECK_OK( session->Run(batch.buildFeed(), {"pred_stop_dist", "pred_target_dist", "pred_action_dist"}, {}, &outputs) );

      // writer.add_summary(summary, i)
      auto stopDistTensor = outputs[0];
      auto targetDistTensor = outputs[1];
      auto actionDistTensor = outputs[2];

      auto stopDist_Mapped = stopDistTensor.tensor<float, 1>();
      auto targetDist_Mapped = targetDistTensor.tensor<float, 2>();
      auto actionDist_Mapped = actionDistTensor.tensor<float, 3>();

      // rescale to combined target-rule distribution
      for (int i = 0; i < batch.size(); ++i) {
        ResultDist res = createResultDist();
        res.stopDist = stopDist_Mapped(i);
        for (int t = 0; t < prog_length; ++t) {
          float pTarget = targetDist_Mapped(i, t);

          // std::cerr << i << " " << t << "  :  " << pTarget << "\n";
          for (int r = 0; r < num_Rules; ++r) {
            auto pAction = actionDist_Mapped(i, t, r);
            // std::cerr << i << " " << t << " " << r << "  :  " << pAction << "\n";
            int i = t*num_Rules + r;
            assert(0 <= i && i < res.actionDist.size());
            res.actionDist[i] = pAction * pTarget;
          }
        }

        res.normalize();
        oResultDistVec[startIdx + i] = res;
      }
    }

    delete batchVec;
  });

  return workerThread;
}

// set learning rate
void
Model::setLearningRate(float v) {

  Tensor rateTensor(DT_FLOAT, TensorShape());
  rateTensor.scalar<float>()() = v;

  FeedDict dict = {
      {"new_learning_rate", rateTensor}
  };

  {
    std::lock_guard<std::mutex> guard(modelMutex);
    TF_CHECK_OK( session->Run(dict, {"set_learning_rate"}, {}, nullptr) );
  }
}

Model::Statistics
Model::query_stats() {
  std::vector<tensorflow::Tensor> outputs;
  {
    std::lock_guard<std::mutex> guard(modelMutex);
    TF_CHECK_OK( session->Run({}, {"learning_rate", "global_step"}, {}, &outputs) );
  }
  double learning_rate = outputs[0].scalar<float>()();
  size_t global_step = outputs[1].scalar<int>()();
  return Model::Statistics{global_step, learning_rate};
}


void
Model::Statistics::print(std::ostream & out) const {
  out << "global_step=" << global_step << ", learning_rate=" << learning_rate;
}

ResultDist
Model::createStopResult() const {
  auto dist = createEmptyResult();
  dist.stopDist = 1.0;
  return dist;
}

ResultDist
Model::createEmptyResult() const { return ResultDist(num_Rules, prog_length); }

void
ResultDist::print(std::ostream & out) const {
  out << "Res {stop=" << stopDist << ", actions="; PrintDist(actionDist, out); out << "}\n";
}

void
ResultDist::dump() const { print(std::cerr); }

void
ResultDist::normalize() {
  stopDist = Clamp(stopDist, 0.0, 1.0);
  Normalize(actionDist);
}


bool
ResultDist::isStop() const {
  return stopDist == 1.0;
}


std::ostream&
Model::Losses::print(std::ostream & out) const {
  out << "actionLoss=" << actionLoss << ", targetLoss=" << targetLoss << ", stopLoss=" << stopLoss;
  return out;
}


} // namespace apo
