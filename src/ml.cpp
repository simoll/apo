#include "ml.h"

#include "config.h"
#include <cassert>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

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
  opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
  opts.config.mutable_gpu_options()->set_allow_growth(true);

  Status status = NewSession(opts, &session);

  if (!status.ok()) {
    std::cout << "NEW_SESSION: " << status.ToString() << "\n";
    return 1;
  }

  return 0;
}


Model::Model(const std::string & graphFile) {
  init_tflow(); // make sure tensorflow session works as expected

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

void
Model::train(const ProgramVec& progs, const std::vector<Result>& results) {
  // program encoding
  struct Batch {
    const Model & model;
    Tensor oc_feed;
    Tensor firstOp_feed;
    Tensor sndOp_feed;
    Tensor length_feed;
    Tensor result_feed;

    Batch(const Model & _model)
    : model(_model)
    , oc_feed(DT_INT32, TensorShape({model.batch_size, model.max_Time}))
    , firstOp_feed(DT_INT32, TensorShape({model.batch_size, model.max_Time}))
    , sndOp_feed(DT_INT32,   TensorShape({model.batch_size, model.max_Time}))
    , length_feed(DT_INT32,  TensorShape({model.batch_size}))
    , result_feed(DT_INT32,  TensorShape({model.batch_size}))
    {}

    void encode(int batch_id, const Program & prog, const Result & result) {
      auto oc_Mapped = oc_feed.tensor<int, 2>();
      auto firstOp_Mapped = oc_feed.tensor<int, 2>();
      auto sndOp_Mapped = oc_feed.tensor<int, 2>();
      auto length_Mapped = length_feed.tensor<int, 1>();
      auto result_Mapped = result_feed.tensor<int, 1>();

      assert((prog.size() <= model.max_Time) && "program size exceeds model limits");

      for (int t = 0; t < prog.size(); ++t) {
        oc_Mapped(batch_id, t) = model.encodeOpCode(prog.code[t]);
        firstOp_Mapped(batch_id, t) = model.encodeOperand(prog.code[t], 0);
        sndOp_Mapped(batch_id, t) = model.encodeOperand(prog.code[t], 1);
      }
      length_Mapped(batch_id) = prog.size();
      result_Mapped(batch_id) = result.numAdds;
    }

    FeedDict
    buildFeed() {
      FeedDict dict = {
        {"oc_data", oc_feed},
        {"firstOp_data", firstOp_feed},
        {"sndOp_data", sndOp_feed},
        {"rule_in", result_feed},
        {"length_data", length_feed}
      };
      return dict;
    }
  };

  // TODO build batch from programs
  assert(results.size() == progs.size());
  assert(results.size() == batch_size);
  Batch batch(*this);
  for (int i = 0; i < batch_size; ++i) {
    const Program & P = *progs[i];
    batch.encode(i, P, results[i]);
  }


  // a.scalar<float>()() = 3.0;

  // Tensor b(DT_FLOAT, TensorShape());
  // b.scalar<float>()() = 2.0;

  // std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
  //   { "a", a },
  //   { "b", b },
  // };

  // The session will initialize the outputs
  FeedDict inputs = batch.buildFeed();
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  Status status = session->Run(inputs, {"loss"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << "TF: error in session::run : " << status.ToString() << "\n";
    abort();
  }

  // print something interesting
  std::cout << "Outputs: " << outputs.size() << "\n";
  std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  auto loss_out = outputs[0].scalar<float>();
  const int step = 0; // TODO add training loop
  std::cout << "Loss at step " << step << ": " << loss_out << "\n";

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  // std::cout << output_c() << "\n"; // 30
}

} // namespace apo
