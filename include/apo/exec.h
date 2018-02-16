#ifndef APO_EXEC_H
#define APO_EXEC_H

#include "apo/program.h"
#include "apo/ADT/SmallVector.h"
#include "apo/shared.h"

namespace apo {

using DataVec = llvm::SmallVector<data_t, 2>;

static data_t
Evaluate(OpCode oc, const DataVec dataVec) {
  data_t A = dataVec[0];

  switch (Num_Operands(oc)) {
    case 1: {
      // unary operators
      assert(oc == OpCode::Pipe);
      return A;
    }

    case 2: {
      // binary operators
      data_t B = dataVec[1];
      data_t result;
      switch (oc) {
        case OpCode::Add: result = A + B; break;
        case OpCode::Sub: result = A - B; break;
        case OpCode::Mul: result = A * B; break;
        case OpCode::And: result = A & B; break;
        case OpCode::Or:  result = A | B; break;
        case OpCode::Xor: result = A ^ B; break;
      default:
        abort(); // unknown operator
      }
      return result;
    }

    default: {
      abort(); // unsupported operator
    }
  }
}

static data_t
Evaluate(const Program & prog, const DataVec params) {
  data_t state[prog.size()];

  assert(prog.code[prog.size() - 1].oc == OpCode::Return);
  for (int pc = 0; pc <= prog.getReturnIndex(); ++pc) {
    data_t result = 0; // no undefined behavior...
    const Statement & stat = prog.code[pc];
    if (stat.isConstant()) {
      // constant
      result = stat.getValue();
    } else if (stat.oc == OpCode::Nop) {
      // no-op

    } else if (stat.isOperator()) {
      DataVec dataVec;
      #pragma unroll
      for (int o = 0; o < stat.num_Operands(); ++o) {
        node_t firstIdx = stat.getOperand(0);
        data_t argVal = IsArgument(firstIdx) ? params[GetHoleIndex(firstIdx)] : state[firstIdx];
        dataVec.push_back(argVal);
      }

      result = Evaluate(stat.oc, dataVec);
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
