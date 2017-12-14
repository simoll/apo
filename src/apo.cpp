#include <stdint.h>
#include <iostream>
#include <map>

enum class OpCode : int16_t {
  Nop = 0,
  // @Data typed literal
  Constant,

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
    case OpCode::Nop: { out << "nop"; } break;
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
    int32_t indices[2];
    Data value;
  } elements;

  // operand indices
  int32_t getOperand(int i) const { return elements.indices[i]; }
  void setOperand(int i, int32_t op) { elements.indices[i] = op; }

  // only for constants
  Data getValue() const { return elements.value; }

  void print(std::ostream & out, int i) const {
    if (oc == OpCode::Nop) return;

    out << '%' << i << " = ";

    // constant
    if (oc == OpCode::Constant) {
      out << ' ' << elements.value << "\n";
      return;
    }

    // regular binary operator
    PrintOpCode(oc, out);
    out << " "; PrintIndex(getOperand(0), out);
    out << " "; PrintIndex(getOperand(1), out);
    out << "\n";
  }

  Statement()
  : oc(OpCode::Nop)
  {}

  Statement(OpCode _oc, int32_t firstOp, int32_t secondOp)
  : oc(_oc)
  {
    elements.indices[0] = firstOp;
    elements.indices[1] = secondOp;
  }

  Statement(Data constVal)
  : oc(OpCode::Constant)
  {
    elements.value = constVal;
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

  // compress all no-op (OpCode::Nop) nodes
  void compress() {
    int j = 0;
    std::map<int32_t, int32_t> operandMap;

    for (int i = 0; i < MaxLength; ++i) {
      if (code[i].oc == OpCode::Nop) continue;

      // compress
      operandMap[i] = j;
      int slot = j++;
      code[slot] = code[i];
      if (i != slot) code[i].oc = OpCode::Nop;

      if (code[slot].oc == OpCode::Constant) continue;

      // remap operands
      int oldFirst = code[slot].getOperand(0);
      if (oldFirst >= 0) code[slot].setOperand(0, operandMap[oldFirst]);

      int oldSecond = code[slot].getOperand(1);
      if (oldSecond >= 1) code[slot].setOperand(1, operandMap[oldSecond]);
    }

    codeLen = j;
  }

  void print(std::ostream & out) const {
    out << "Program {\n";
    for (int i = 0; i < codeLen; ++i) {
      code[i].print(out, i);
    }
    out << "}\n";
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

      case OpCode::Nop: break;
      default:
        abort(); // not implemented
      }
    }

    state[pc] = result;
  }

  return result;
}
// matches program @pattern aligning the two at the statement at @pc of @prog
static bool
MatchPattern(const Program & prog, int pc, const Program & pattern) {
  // TODO
}

struct Rule {
  Program lhs;
  Program rhs;
};



int main(int argc, char ** argv) {
  Program prog;
  prog.codeLen = 3;
  prog.numParams = 2;
  prog.code[0] = Statement(OpCode::Add, -1, -2);
  prog.code[1] = Statement(OpCode::Nop, 0, 0);
  prog.code[2] = Statement(OpCode::Sub, 0, -1);
  prog.dump();
  prog.compress();
  prog.dump();
}

