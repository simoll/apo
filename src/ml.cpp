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


Model::Model(const std::string & graphFile, const std::string & configFile) {
  init_tflow(); // make sure tensorflow session works as expected

// parse shared configuration
  {
      Parser confParser(configFile);
      max_Time = confParser.get<int>("max_Time");
      num_Params = confParser.get<int>("num_Params");
      batch_size = confParser.get<int>("batch_size");
      num_Rules = confParser.get<int>("num_Rules");
  }

  std::cerr << "Model (apo). max_Time=" << max_Time << ", num_Params=" << num_Params << ", batch_size=" << batch_size << ", num_Rules=" << num_Rules << "\n";

// build Graph
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

  // Setup inputs and outputs:
  std::cout << "TF: loaded graph " << graphFile << "\n";

  // Initialize our variables
  TF_CHECK_OK(session->Run({}, {}, {"init_op"}, nullptr));
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
  // program emcoding
  Tensor oc_feed;
  Tensor firstOp_feed;
  Tensor sndOp_feed;
  Tensor length_feed;

  // output encoding
  Tensor rule_feed;
  Tensor target_feed;

  Batch(const Model & _model)
  : model(_model)
  , oc_feed(DT_INT32, TensorShape({model.batch_size, model.max_Time}))
  , firstOp_feed(DT_INT32, TensorShape({model.batch_size, model.max_Time}))
  , sndOp_feed(DT_INT32,   TensorShape({model.batch_size, model.max_Time}))
  , length_feed(DT_INT32,  TensorShape({model.batch_size}))
  , rule_feed(DT_INT32,  TensorShape({model.batch_size}))
  , target_feed(DT_INT32,  TensorShape({model.batch_size}))
  {}

  void encode_Program(int batch_id, const Program & prog) {
    auto oc_Mapped = oc_feed.tensor<int, 2>();
    auto firstOp_Mapped = firstOp_feed.tensor<int, 2>();
    auto sndOp_Mapped = sndOp_feed.tensor<int, 2>();
    auto length_Mapped = length_feed.tensor<int, 1>();

    assert((prog.size() <= model.max_Time) && "program size exceeds model limits");

    for (int t = 0; t < prog.size(); ++t) {
      oc_Mapped(batch_id, t) = model.encodeOpCode(prog.code[t]);
      firstOp_Mapped(batch_id, t) = model.encodeOperand(prog.code[t], 0);
      sndOp_Mapped(batch_id, t) = model.encodeOperand(prog.code[t], 1);
    }
    for (int t = prog.size(); t < model.max_Time; ++t) {
      oc_Mapped(batch_id, t) = 0;
      firstOp_Mapped(batch_id, t) = 0;
      sndOp_Mapped(batch_id, t) = 0;
    }

    length_Mapped(batch_id) = prog.size();
  }

  void encode_Result(int batch_id, const Result & result) {
    auto rule_Mapped = rule_feed.tensor<int, 1>();
    auto target_Mapped = target_feed.tensor<int, 1>();
    rule_Mapped(batch_id) = result.rule;
    target_Mapped(batch_id) = result.target;
  }

  FeedDict
  buildFeed() {
    FeedDict dict = {
      {"oc_data", oc_feed},
      {"firstOp_data", firstOp_feed},
      {"sndOp_data", sndOp_feed},
      {"rule_in", rule_feed},
      {"target_in", target_feed},
      {"length_data", length_feed}
    };
    return dict;
  }
};

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

ResultVec
Model::infer_likely(const ProgramVec& progs) {
  int num_Samples = progs.size();
  assert(num_Samples % batch_size == 0);

  ResultVec results;

  Batch batch(*this);
  for (int s = 0; s + batch_size - 1 < num_Samples; s += batch_size) {
    for (int i = 0; i < batch_size; ++i) {
      const Program & P = *progs[s + i];
      batch.encode_Program(i, P);
    }

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    TF_CHECK_OK( session->Run(batch.buildFeed(), {"pred_rule", "pred_target"}, {}, &outputs) );
    // writer.add_summary(summary, i)
    auto ruleTensor = outputs[0];
    auto targetTensor = outputs[1];

    auto rule_Mapped = ruleTensor.tensor<int, 1>();
    auto target_Mapped = targetTensor.tensor<int, 1>();
    assert(num_Rules == ruleTensor.dim_size(1));

    for (int i = 0; i < batch_size; ++i) {
      results.push_back(Result{rule_Mapped(i), target_Mapped(i)});
    }
  }

  return results;
}

ResultDistVec
Model::infer_dist(const ProgramVec& progs, bool failSilently) {
  int num_Samples = progs.size();
  assert(num_Samples % batch_size == 0);

  ResultDistVec results;

  Program emptyP(num_Params, {}); // the empty program

  Batch batch(*this);
  for (int s = 0; s + batch_size - 1 < num_Samples; s += batch_size) {
    bool allEmpty = true;
    for (int i = 0; i < batch_size; ++i) {
      const Program & P = *progs[s + i];
      if (P.size() > max_Time) {
        batch.encode_Program(i, emptyP);
      } else {
        allEmpty = false;
        batch.encode_Program(i, P);
      }
    }

    if (allEmpty) {
      // early continue
      for (int i = 0; i < batch_size; ++i) {
        results.push_back(createResultDist());
      }
      continue;
    }

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    TF_CHECK_OK( session->Run(batch.buildFeed(), {"pred_rule_dist", "pred_target_dist"}, {}, &outputs) );
    // writer.add_summary(summary, i)
    auto ruleDistTensor = outputs[0];
    auto targetDistTensor = outputs[1];

    auto ruleDist_Mapped = ruleDistTensor.tensor<float, 2>();
    auto targetDist_Mapped = targetDistTensor.tensor<float, 2>();

    for (int i = 0; i < batch_size; ++i) {
      ResultDist res = createResultDist();
      for (int j = 0; j < num_Rules; ++j) {
        res.ruleDist[j] = ruleDist_Mapped(i, j);
      }

      for (int j = 0; j < progs[s+i]->size(); ++j) {
        res.targetDist[j] = targetDist_Mapped(i, j);
      }

      Normalize(res.ruleDist);
      Normalize(res.targetDist);

      results.push_back(res);
    }
  }

  return results;
}

void
ResultDist::print(std::ostream & out) const {
  out << "Res {rule="; PrintDist(ruleDist, out); out << ", target="; PrintDist(targetDist, out); out << "}\n";
}

void
ResultDist::dump() const { print(std::cerr); }

} // namespace apo
