#ifndef APO_PROGRAM_H
#define APO_PROGRAM_H

#include "config.h"

#include <cassert>
#include <map>
#include <functional>
#include <iostream>

#include <vector>
#include <memory>

namespace apo {

enum class OpCode : int16_t {
  Begin_OpCode = 0,
  Nop = 0,
  Pipe = 1, // fake value use (e.g. replication)

  // @data_t typed literal
  Constant = 2,

  // single operand (for now)
  Return = 3,

  // arithmetic
  Begin_Binary = 4,
  Add = Begin_Binary,
  Sub = 5,
  Mul = 6,

  // bitwise logic
  And = 7,
  Or = 8,
  Xor = 9,
  End_Binary = Xor,
  End_OpCode = Xor
};

// arithmetic data type
using data_t = uint64_t;
using DataVec = std::vector<data_t>;

// node index data type
using node_t = int32_t;
using NodeVec = std::vector<node_t>;

static bool
IsCommutative(OpCode oc) {
  return (oc == OpCode::Add || oc == OpCode::Mul || oc == OpCode::And || oc == OpCode::Or || oc == OpCode::Xor);
}

static bool
IsAssociative(OpCode oc) {
  return (oc == OpCode::Add || oc == OpCode::Mul || oc == OpCode::And || oc == OpCode::Or);
}

static bool
HasNeutralRHS(OpCode oc) {
  return (oc == OpCode::Add || oc == OpCode::Mul || oc == OpCode::Sub || oc == OpCode::And || oc == OpCode::Or || oc == OpCode::Xor);
}

static data_t
GetNeutralRHS(OpCode oc) {
  assert(HasNeutralRHS(oc));
  switch (oc) {
  case OpCode::Add:
  case OpCode::Or:
  case OpCode::Xor:
  case OpCode::Sub:
    return (data_t) 0;

  case OpCode::Mul:
    return (data_t) 1;

  case OpCode::And:
    return (data_t) -1;

  default: abort(); // not mapped
  }
}

static
void
for_such(std::function<bool(OpCode oc)> filterFunc, std::function<void(OpCode oc)> userFunc) {
  for (int16_t oc = (int16_t) OpCode::Begin_OpCode; oc <= (int16_t) OpCode::End_OpCode; ++oc) {
    if (filterFunc((OpCode) oc)) userFunc((OpCode) oc);
  }
}

static void
PrintOpCode(OpCode oc, std::ostream & out) {
  switch (oc) {
    case OpCode::Nop: { out << "nop"; } break;
    case OpCode::Pipe: {out << "#"; } break;
    case OpCode::Return: { out << "ret"; } break;
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
PrintIndex(node_t idx, std::ostream & out) {
  if (idx < 0) {
    char c = (char) ('a'-1 - idx);
    out << '%' << c;
  } else {
    out << '%' << idx;
  }
}


const int operandLimit = 2;
struct Statement {
  OpCode oc;

  union {
    node_t indices[operandLimit];
    data_t value;
  } elements;

  // operand indices
  int32_t getOperand(int i) const { return elements.indices[i]; }
  void setOperand(int i, int32_t op) { elements.indices[i] = op; }

  // only for constants
  data_t getValue() const { return elements.value; }

  bool isOperator() const { return oc != OpCode::Nop && oc != OpCode::Constant; }

  bool isConstant() const { return oc == OpCode::Constant; }

  int num_Operands() const {
    if (oc == OpCode::Nop || oc == OpCode::Constant) return 0;
    else if (oc == OpCode::Return || oc == OpCode::Pipe) return 1;
    return 2;
  }

  void print(std::ostream & out, int i) const {
    if (oc == OpCode::Nop) { out << "\n"; return; }
    if (oc == OpCode::Return) {
      out << "ret "; PrintIndex(getOperand(0), out);
      out << "\n";
      return;
    }

    out << '%' << i << " = ";

    // constant
    if (oc == OpCode::Constant) {
      out << ' ' << elements.value << "\n";
      return;
    }

    // regular binary operator
    PrintOpCode(oc, out);
    for (int o = 0; o < num_Operands(); ++o) {
      out << " "; PrintIndex(getOperand(o), out);
    }
    out << "\n";
  }

  Statement()
  : oc(OpCode::Nop)
  {}

  // binary op - ctor
  Statement(OpCode _oc, int32_t firstOp, int32_t secondOp)
  : oc(_oc)
  {
    elements.indices[0] = firstOp;
    elements.indices[1] = secondOp;
  }

  // single op - ctor
  Statement(OpCode _oc, int32_t op)
  : oc(_oc)
  {
    elements.indices[0] = op;
  }

  // constant - ctor
  Statement(data_t constVal)
  : oc(OpCode::Constant)
  {
    elements.value = constVal;
  }

  inline uint64_t hash() const noexcept {
    if (isConstant()) return (uint64_t) elements.value;
    else {
      uint64_t accu = (uint64_t) oc;

      for (int i = 0; i < num_Operands(); ++i) {
        accu = (accu * 97) ^ (uint64_t) getOperand(i);
      }
      return accu;
    }
  }

  bool operator==(const Statement & O) const noexcept {
    if (O.oc != oc) return false;
    if (oc == OpCode::Constant) { return elements.value == O.elements.value; }

    for (int i = 0; i < num_Operands(); ++i) {
      if (getOperand(i) != O.getOperand(i)) return false;
    }

    // equal oc, equal parameters
    return true;
  }

  bool operator<(const Statement & O) const noexcept {
    if (oc < O.oc) return true;
    else if (O.oc < oc) return false;

    // same opCode (compare oc parameters)
    if (oc == OpCode::Constant) { return elements.value < O.elements.value; }

    for (int i = 0; i < num_Operands(); ++i) {
      if (getOperand(i) < O.getOperand(i)) return true;
    }

    // equal oc, equal parameters
    return false;
  }
};

static Statement build_pipe(int32_t handle) { return Statement(OpCode::Pipe, handle); }
static Statement build_nop() { return Statement(); }
static Statement build_ret(int32_t handle) { return Statement(OpCode::Return, handle); }
static Statement build_const(data_t val) { return Statement(val); }


static bool
IsStatement(int32_t idx) {
  return idx >= 0;
}

static bool
IsArgument(int32_t idx) {
  return idx < 0;
}

static int
GetHoleIndex(int32_t valueIdx) {
  assert(valueIdx < 0); // range -1, . , -numHoles
  return -(valueIdx + 1);
}

using ReMap = std::map<int32_t, int32_t>;

struct Program {
  // number of parameters
  int numParams;

  // instruction listing
  std::vector<Statement> code;

  int size() const { return code.size(); }
  int num_Params() const { return numParams; }

  Program(int _numParams, std::initializer_list<Statement> stats)
  : numParams(_numParams)
  , code(stats)
  {}

  Program()
  : numParams(0)
  , code()
  { code.reserve(4); }

  // index of returned value
  int
  getReturnIndex() const {
    assert(code[size() - 1].oc == OpCode::Return);
    return code[size() - 1].getOperand(0);
  }

  bool verify() {
    auto handler = [](int pc, std::string msg) {
      std::cerr << "ERROR at " << pc << ": " << msg << "\n"; return true;
    };
    return verify(handler);
  }

  bool
  verify(std::function<bool(int pc, std::string msg)> handler) const {
    for (int pc = 0; pc < size(); ++pc) {
      const auto & stat = code[pc];
      for (int o = 0; o < stat.num_Operands(); ++o) {
        if (stat.getOperand(o) >= pc) {
          if (handler(pc, "use before definition!")) return false;
        }

        if (stat.getOperand(o) <= -num_Params() - 1) {
          if (handler(pc, "uses out-of-bounds parameter!")) return false;
        }
      }

      if (code[size() - 1].oc != OpCode::Return) {
        if (handler(size() - 1, "does not terminate in a return statement!")) return false;
      }
    }

    return true;
  }

  void push(Statement&& s) {
    code.emplace_back(s);
  }

  // create @size many Nop slots before @endPc
  int make_space(int endPc, int allocSize, ReMap & reMap) {
    IF_VERBOSE { std::cerr << "make_space( endPc " << endPc << ", size " << allocSize << ")\n"; }
    // compact until @endPc
    int j = 0;

    for (int i = 0; i < endPc; ++i) {
      if (code[i].oc == OpCode::Nop) continue;

      // compact
      int slot = j++;
      reMap[i] = slot;
      code[slot] = code[i];
      if (i != slot) code[i].oc = OpCode::Nop;

      // remap operands
      for (int o = 0; o < code[slot].num_Operands(); ++o) {
        int old = code[slot].getOperand(o);
        if (old >= 0) code[slot].setOperand(o, reMap[old]);
      }
    }

    // move all other instructions back by the remaining gap
    int gain = endPc - (j - 1); // compaction gain
    // if (gain >= allocSize) return endPc; // user unmodified


    // Otw, shift everything back (remapping operands)
    int extraSpace = allocSize - gain;
    int shift = std::max<int>(0, extraSpace);

    code.resize(size() + shift, build_nop());

    assert(shift >= 0);
    for (int i = size() - 1; i >= endPc + shift; --i) {
      if (shift != 0) code[i] = code[i - shift];
      for (int o = 0; o < code[i].num_Operands(); ++o) {
        int old = code[i].getOperand(o);
        if (!IsStatement(old)) continue; // only remap statements

        if (old >= endPc) {
          // shifted operand
          code[i].setOperand(o, old + shift);
          reMap[i] = i + shift;
        } else {
          int modifiedOp = 0;
          if (reMap.count(old)) {
            modifiedOp = reMap[old];
          } else {
            modifiedOp = old;
          }
          // remapping before endPc
          code[i].setOperand(o, modifiedOp);
        }
      }
    }

    // fill in no-ops in shifted region
    for (int i = endPc; i < endPc + shift; ++i) {
      code[i].oc = OpCode::Nop;
    }

    return endPc + shift;
  }

  // compact all no-op (OpCode::Nop) nodes
  void compact() {
    int j = 0;
    ReMap operandMap;

    for (int i = 0; i < code.size(); ++i) {
      if (code[i].oc == OpCode::Nop) continue;

      // compact
      operandMap[i] = j;
      int slot = j++;
      code[slot] = code[i];
      if (i != slot) code[i].oc = OpCode::Nop;

      // remap operands
      for (int o = 0; o < code[slot].num_Operands(); ++o) {
        int old = code[slot].getOperand(o);
        if (old >= 0) code[slot].setOperand(o, operandMap[old]);
      }
    }

    code.resize(j);
  }

  // paste this program into dest (starting at @startPc)
  // returns the linked return value (instead of linking it in)
  int link(Program & dest, int startPc, NodeVec & holes) const {
    for (int i = 0; i < size() - 1; ++i) {
      int destPc = startPc + i;
      dest.code[destPc] = code[i];
      for (int o = 0; o < code[i].num_Operands(); ++o) {
        int opIdx = code[i].getOperand(o);
        if (IsArgument(opIdx)) {
          // operand is a hole in the pattern -> replace with index in match hole
          int holeIdx = GetHoleIndex(opIdx);
          dest.code[destPc].setOperand(o, holes[holeIdx]);
        } else {
          // operand is a proper statement -> shift by startPc
          dest.code[destPc].setOperand(o, startPc + opIdx); // apply startPc
        }
      }
    }

    // fetch the return value
    const auto & ret = code[size() - 1];
    assert(ret.oc == OpCode::Return); // expected a return statement
    if (IsArgument(ret.getOperand(0))) {
      int holeIdx = GetHoleIndex(ret.getOperand(0));
      return holes[holeIdx]; // matched hole
    } else {
      return startPc + ret.getOperand(0); // shifted
    }
  }

  Statement & operator()(int pc) { return code[pc]; }
  const Statement & operator()(int pc) const { return code[pc]; }

  void print(std::ostream & out) const {
    out << "Program (" << num_Params() << ") {\n";
    for (int i = 0; i < size(); ++i) {
      if (code[i].oc != OpCode::Nop) out << i << ": "; code[i].print(out, i);
    }
    out << "}\n";
  }

  void dump() const { print(std::cerr); }

  bool operator==(const Program & O) const {
    if (num_Params() != O.num_Params()) return false;
    if (size() != O.size()) return false;
    for (int pc = 0; pc < size(); ++pc) {
      if (!(code[pc] == O.code[pc])) return false;
    }
    return true;
  }

  bool operator<(const Program & O) const {
    if (num_Params() < O.num_Params()) return true;
    if (size() < O.size()) return true;
    if (size() > O.size()) return false;
    // equally sized
    for (int pc = 0; pc < size(); ++pc) {
      if (code[pc] < O.code[pc]) return true;
    }
    return false;
  }
};

// dereferencing less operator
template <class T> struct deref_less {
  bool operator() (const T& x, const T& y) const {
    if (x.get() == nullptr) return y.get() != nullptr;
    else return *x < *y;
  }
};

using ProgramPtr = std::shared_ptr<Program>;
using ProgramVec = std::vector<ProgramPtr>;

struct ProgramHasher {
  size_t operator()(const Program & P) const noexcept
  {
    uint64_t hash = P.num_Params();

    for (uint64_t i = 0; i < P.size(); ++i) {
      hash = 0x8001 * hash + (P.code[i].hash() ^ i);
    }
    return reinterpret_cast<size_t>(hash);
  }
};


struct ProgramPtrHasher {
  size_t operator()(const ProgramPtr & Pptr) const noexcept
  {
    if (Pptr == nullptr) return 0;
    const Program & P = *Pptr;
    uint64_t hash = P.num_Params();

    for (uint64_t i = 0; i < P.size(); ++i) {
      hash = 0x8001 * hash + (P.code[i].hash() ^ i);
    }
    return reinterpret_cast<size_t>(hash);
  }
};

struct ProgramPtrEqual {
  bool operator()( const ProgramPtr & A, const ProgramPtr & B) const noexcept {
    if (A.get() == B.get()) return true;
    if ((A == nullptr) || (B == nullptr)) return false;
    return *A == *B;
  }
};



static
ProgramVec
Clone(const ProgramVec & progVec) {
  ProgramVec cloned;
  cloned.reserve(progVec.size());
  for (const auto & P : progVec) {
    cloned.emplace_back(new Program(*P));
  }
  return cloned;
}


} // namespace apo


#endif // APO_PROGRAM_H
