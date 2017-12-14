#include <stdint.h>
#include <iostream>

enum class OpCode : int16_t {
  Constant = 0,

  // arithmetic
  Add,
  Sub,
  Mul,

  // bitwise logic
  And,
  Or,
  Xor
};
static void
PrintOpCode(OpCode oc, std::ostream & out) {
  switch (oc) {
    case OpCode::Add: { out << "add"; } break;
    case OpCode::Sub: { out << "sub"; } break;
    case OpCode::Mul: { out << "mul"; } break;
    case OpCode::And: { out << "and"; } break;
    case OpCode::Or: { out << "or"; } break;
    case OpCode::Xor: { out << "xor"; } break;
    default: abort();
  }
}

static void
PrintIndex(int32_t idx, std::ostream & out) {
  if (idx < 0) {
    char c = (char) ('a'-1 - idx);
    out << '%' << c;
  } else {
    out << '%' << idx;
  }
}

using Data = uint64_t;

struct Statement {
  OpCode oc;

  union {
    struct {
      int32_t first;
      int32_t second;
    } indices;
    Data value;
  } elements;

  // operand indices
  int32_t getOperand(int i) const {
    if (i == 0) return elements.indices.first;
    if (i == 1) return elements.indices.second;
  }

  // only for constants
  Data getValue() const { return elements.value; }

  void print(std::ostream & out) const {
    if (oc == OpCode::Constant) {
      out << ' ' << elements.value << "\n";
      return;
    }
    PrintOpCode(oc, out);
    out << " "; PrintIndex(getOperand(0), out);
    out << " "; PrintIndex(getOperand(1), out);
    out << "\n";
  }
};

const size_t MaxLength = 16;

struct Program {
  // number of parameters
  int numParams;

  // number of instructions
  int codeLen;

  // instruction listing
  Statement code[MaxLength];

  void print(std::ostream & out) const {
    for (int i = 0; i < codeLen; ++i) {
      out << '%' << i << " = ";
      code[i].print(out);
    }
  }

  void dump() const { print(std::cerr); }

  Program()
  : numParams(0)
  , codeLen(0)
  {}
};

static Data
evaluate(const Program & prog, Data * params) {
  Data state[prog.codeLen];

  Data result = 0; // no undefined behavior...
  for (int pc = 0; pc < prog.codeLen; ++pc) {
    const Statement & stat = prog.code[pc];
    if (stat.oc == OpCode::Constant) {
      result = stat.getValue();
    } else {
      int32_t first = stat.getOperand(0);
      int32_t second = stat.getOperand(1);
      Data A = first < 0 ? params[(-first) - 1] : state[first];
      Data B = second < 0 ? params[(-second) - 1] : state[second];

      switch (stat.oc) {
      case OpCode::Add:
        result = A + B;
      case OpCode::Sub:
        result = A - B;
      case OpCode::Mul:
        result = A * B;
      case OpCode::And:
        result = A & B;
      case OpCode::Or:
        result = A | B;
      case OpCode::Xor:
        result = A ^ B;
      default:
        abort(); // not implemented
      }
    }

    state[pc] = result;
  }

  return result;
}

int main(int argc, char ** argv) {
}

