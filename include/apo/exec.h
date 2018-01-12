#ifndef APO_EXEC_H
#define APO_EXEC_H

#include "apo/program.h"

namespace apo {


static data_t
Evaluate(const Program & prog, const std::vector<data_t> & params) {
  data_t state[prog.size()];

  assert(prog.code[prog.size() - 1].oc == OpCode::Return);
  for (int pc = 0; pc <= prog.getReturnIndex(); ++pc) {
    data_t result = 0; // no undefined behavior...
    const Statement & stat = prog.code[pc];
    if (stat.oc == OpCode::Constant) {
      // constant
      result = stat.getValue();

    } else if (stat.oc == OpCode::Nop) {
      // pass

    } else if (stat.oc == OpCode::Pipe) {
      // wrapper instructions
      int32_t first = stat.getOperand(0);
      result = first < 0 ? params[GetHoleIndex(first)] : state[first];

    } else {
      // binary operators
      int32_t first = stat.getOperand(0);
      int32_t second = stat.getOperand(1);
      data_t A = first < 0 ? params[GetHoleIndex(first)] : state[first];
      data_t B = second < 0 ? params[GetHoleIndex(second)] : state[second];

      switch (stat.oc) {
      case OpCode::Add: result = A + B; break;
      case OpCode::Sub: result = A - B; break;
      case OpCode::Mul: result = A * B; break;
      case OpCode::And: result = A & B; break;
      case OpCode::Or:  result = A | B; break;
      case OpCode::Xor: result = A ^ B; break;

      default:
        abort(); // not implemented
      }
    }

    state[pc] = result;
  }

  return state[prog.getReturnIndex()];
}

struct RandExecutor {
  std::vector<DataVec> paramSets;

  RandExecutor(int numParams, int numSets)
  : paramSets()
  {
    std::uniform_int_distribution<data_t> argRand(0, std::numeric_limits<data_t>::max());

    for (int s = 0; s < numSets; ++s) {
      DataVec params;
      for (int i = 0; i< numParams; ++i) {
        params.push_back(argRand(randGen()));
      }
      paramSets.push_back(params);
    }
  }

  DataVec run(const Program & P) {
    DataVec results;
    for (const auto & params : paramSets) {
      results.push_back(Evaluate(P, params));
    }
    return results;
 }
};

} // namespace apo

#endif // APO_EXEC_H
