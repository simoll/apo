#include <stdint.h>
#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <set>
#include <initializer_list>
#include <functional>
#include <queue>

#include <random>

std::mt19937 randGen(42);

// #include "tensorflow/cc/client/client_session.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/cc/framework/tensor.h"

const bool Verbose = true;

#define IF_VERBOSE if (Verbose)

// expensive consistency checks
#define IF_DEBUG if (true)

namespace apo {

enum class OpCode : int16_t {
  Begin_OpCode = 0,
  Nop = 0,
  Pipe, // fake value use (e.g. replication)

  // @Data typed literal
  Constant,

  // single operand (for now)
  Return,

  // arithmetic
  Begin_Binary,
  Add = Begin_Binary,
  Sub,
  Mul,

  // bitwise logic
  And,
  Or,
  Xor,
  End_Binary = Xor,
  End_OpCode = (int32_t) Xor + 1
};

// arithmetic data type
using Data = uint64_t;

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

static Data
GetNeutralRHS(OpCode oc) {
  assert(HasNeutralRHS(oc));
  switch (oc) {
  case OpCode::Add:
  case OpCode::Or:
  case OpCode::Xor:
  case OpCode::Sub:
    return (Data) 0;

  case OpCode::Mul:
    return (Data) 1;

  case OpCode::And:
    return (Data) -1;

  default: abort(); // not mapped
  }
}

static
void
for_such(std::function<bool(OpCode oc)> filterFunc, std::function<void(OpCode oc)> userFunc) {
  for (int16_t oc = (int16_t) OpCode::Begin_OpCode; oc < (int16_t) OpCode::End_OpCode; ++oc) {
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
PrintIndex(int32_t idx, std::ostream & out) {
  if (idx < 0) {
    char c = (char) ('a'-1 - idx);
    out << '%' << c;
  } else {
    out << '%' << idx;
  }
}


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
  Statement(Data constVal)
  : oc(OpCode::Constant)
  {
    elements.value = constVal;
  }
};

static Statement build_pipe(int32_t handle) { return Statement(OpCode::Pipe, handle); }
static Statement build_nop() { return Statement(); }
static Statement build_ret(int32_t handle) { return Statement(OpCode::Return, handle); }
static Statement build_const(Data val) { return Statement(val); }


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
  assert(valueIdx < 0);
  return -valueIdx - 1;
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
        handler(size() - 1, "does not terminate in a return statement!");
      }
    }

    return true;
  }

  void push(Statement&& s) {
    code.emplace_back(s);
  }

  // create @size many Nop slots before @endPc
  int make_space(int endPc, int allocSize, ReMap & reMap) {
    // compact until @endPc
    int j = 0;

    for (int i = 0; i < endPc; ++i) {
      if (code[i].oc == OpCode::Nop) continue;

      // compact
      reMap[i] = j;
      int slot = j++;
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
    if (gain >= allocSize) return endPc; // user unmodified


    // Otw, shift everything back
    int shift = allocSize - gain;
    code.resize(size() + shift, build_nop());

    for (int i = size() + shift - 1; i >= endPc + shift; --i) {
      code[i] = code[i - shift];
      for (int o = 0; o < code[i].num_Operands(); ++o) {
        int old = code[i].getOperand(o);
        if (!IsStatement(old)) continue; // only remap statements

        if (old >= endPc) {
          // shifted operand
          code[i].setOperand(o, i + shift);
          reMap[i] = i + shift;
        } else {
          // remapping before endPc
          code[i].setOperand(o, reMap[old]);
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
  int link(Program & dest, int startPc, int32_t * holes) const {
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

  void print(std::ostream & out) const {
    out << "Program (" << num_Params() << ") {\n";
    for (int i = 0; i < size(); ++i) {
      if (code[i].oc != OpCode::Nop) out << i << ": "; code[i].print(out, i);
    }
    out << "}\n";
  }

  void dump() const { print(std::cerr); }

};

static Data
evaluate(const Program & prog, Data * params) {
  Data state[prog.size()];

  Data result = 0; // no undefined behavior...
  for (int pc = 0; pc < prog.size(); ++pc) {
    const Statement & stat = prog.code[pc];
    if (stat.oc == OpCode::Constant) {
      // constant
      result = stat.getValue();

    } else if (stat.oc == OpCode::Nop) {
      // no-op
      continue;

    } else if (stat.oc == OpCode::Pipe) {
      // wrapper instructions
      int32_t first = stat.getOperand(0);
      result = first < 0 ? params[(-first) - 1] : state[first];

    } else {
      // binary operators
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

using NodeSet = std::set<int32_t>;

static bool
rec_MatchPattern(const Program & prog, int pc, const Program & pattern, int patternPc, int32_t * holes, std::vector<bool> & defined, NodeSet & nodes) {
  // matching a pattern hole
  if (patternPc < 0) {
    int holeIdx = -patternPc - 1;
    if (defined[holeIdx]) {
      return holes[holeIdx] == pc; // matched a defined hole
    } else {
      // add a new definition
      holes[holeIdx] = pc;
      return true;
    }
  }

  assert (patternPc >= 0);
  if (pc < 0) {
    // argument matching
    return false; // can only match parameters with placeholders

  } else {
    nodes.insert(pc); // keep track of matched nodes

    OpCode oc = prog.code[pc].oc;
    if (oc != pattern.code[patternPc].oc) return false;
    if (oc == OpCode::Constant) {
      // constant matching
      return pattern.code[patternPc].getValue() == prog.code[pc].getValue(); // matching constant

    } else {
      // generic statement matching
      for (int i = 0; i < 2; ++i) {
        if (!rec_MatchPattern(prog, prog.code[pc].getOperand(i), pattern, pattern.code[patternPc].getOperand(i), holes, defined, nodes)) return false;
      }
      return true;
    }
  }
}

// matches program @pattern aligning the two at the statement at @pc of @prog (stores the operand index at "holes" (parameters) in @holes
static bool
MatchPattern(const Program & prog, int pc, const Program & pattern, int32_t * holes) {
  assert (pattern.size() > 0);

  std::vector<bool> defined(false, pattern.numParams);
  NodeSet nodes;
  // match the pattern
  int retIndex = pattern.code[pattern.size() - 1].getOperand(0);
  bool ok = rec_MatchPattern(prog, pc, pattern, retIndex, holes, defined, nodes);
  if (Verbose) {
    std::cerr << "op match: " << ok << "\n";
  }

  // verify that there is no user that is not covered by the pattern (except for the match root)
  for (int i = 0; ok && (i < prog.size()); ++i) {
    if (!prog.code[i].isOperator()) continue;

    for (int o = 0; ok && (o < prog.code[i].num_Operands()); ++o) {
      int opIdx = prog.code[i].getOperand(o);
      if (opIdx < 0 || opIdx == pc) continue; // operand is parameter or the match root (don't care)
      if (nodes.count(opIdx)) { // uses part of the match
        if (!nodes.count(i)) { // .. only possible if this instruction is also part of th match
          ok = false;
          break;
        }
      }
    }
  }

  if (Verbose) {
    if (ok) {
      std::cerr << "Match. Holes:";
      for (int i = 0; i < pattern.numParams; ++i) {
        std::cerr << " "; PrintIndex(holes[i], std::cerr);
      }
      std::cerr << "\n";
    }
    else {
      std::cerr << "Mismatch.\n";
    }
  }
  return ok;
}

static void
rec_ErasePattern(Program & prog, int pc, const Program & pattern, int patternPc, NodeSet & erased) {
  if (!IsStatement(pc)) return; // do not erase by argument indices
  if (!erased.insert(pc).second) return;

  auto & stat = prog.code[pc];
  auto & patStat = pattern.code[patternPc];

  if (patStat.isOperator()) {
    for (int i = 0; i < stat.num_Operands(); ++i) {
      int nextPatIdx = patStat.getOperand(i);
      int nextProgIdx = stat.getOperand(i);
      if (IsStatement(nextPatIdx)) {
        rec_ErasePattern(prog, nextProgIdx, pattern, nextPatIdx, erased);
      }
    }
  }

  stat.oc = OpCode::Nop;
}

static void
ErasePattern(Program & prog, int pc, const Program & pattern) {
  if (pattern.size() <= 1) return; // nothing to erase
  NodeSet erased;
  rec_ErasePattern(prog, pc, pattern, pattern.getReturnIndex(), erased);
}


// rewrite rule lhs -> rhs
struct Rule {
  Program lhs;
  Program rhs;

  bool removesHoles(bool leftMatch) const {
    return getMatchProg(leftMatch).num_Params() > getRewriteProg(leftMatch).num_Params();
  }

  Program & getMatchProg(bool matchLeft) {
    if (matchLeft) return lhs; else return rhs;
  }
  const Program & getMatchProg(bool matchLeft) const {
    if (matchLeft) return lhs; else return rhs;
  }

  Program & getRewriteProg(bool matchLeft) {
    if (matchLeft) return rhs; else return lhs;
  }
  const Program & getRewriteProg(bool matchLeft) const {
    if (matchLeft) return rhs; else return lhs;
  }

  Rule(Program _lhs, Program _rhs)
  : lhs(_lhs)
  , rhs(_rhs)
  {}

  bool match(bool matchLeft, const Program & prog, int pc, int32_t * holes) const {
    return MatchPattern(prog, pc, getMatchProg(matchLeft), holes);
  }

  void print(std::ostream & out) const {
    out << "Rule [[ lhs = "; lhs.print(out);
    out << "rhs = ";
    rhs.print(out);
    out << "]]\n";
  }

  void dump() const { print(std::cerr); }

  // applies this rule (after a match)
  void rewrite(bool matchLeft, Program & prog, int rootPc, int32_t * holes) const {
    IF_VERBOSE { std::cerr << "-- rewrite at " << rootPc << " --\n"; prog.dump(); }

    // erase @lhs
    ErasePattern(prog, rootPc, getMatchProg(matchLeft));

    IF_VERBOSE { std::cerr << "-- after erase at " << rootPc << " --\n"; prog.dump(); }

    // make space for rewrite rule
    ReMap reMap;
    int afterInsertPc = prog.make_space(rootPc + 1, getRewriteProg(matchLeft).size(), reMap);
    // updated match root
    int mappedRootPc = reMap.count(rootPc) ? reMap[rootPc] : rootPc;

    IF_VERBOSE { std::cerr << "-- after make_space --\n"; prog.dump(); }

    // insert @rhs
    // patch holes (some positions were remapped)
    IF_VERBOSE { std::cerr << "Adjust remap\n"; }
    for (int i = 0; i < getMatchProg(matchLeft).num_Params(); ++i) {
      IF_VERBOSE { std::cerr << i << " -> " << holes[i]; }
      if (reMap.count(holes[i])) { // re-map hole indices for moved statements
        holes[i] = reMap[holes[i]];
      }
      IF_VERBOSE { std::cerr << "  " << holes[i] << "\n"; }
    }

    // rmappedRootPc replacement after the insertion of this pattern
    int rootSubstPc = getRewriteProg(matchLeft).link(prog, afterInsertPc - (getRewriteProg(matchLeft).size() - 1), holes);

    IF_VERBOSE {
      std::cerr << "-- after linking (root subst " << rootSubstPc << ") -- \n";
      prog.dump();
    }

    // prog.applyAfter(afterInsertPc, lambda[=](Statement & stat) { .. }
    for (int i = rootSubstPc + 1; i < prog.size(); ++i) {
      for (int o = 0; o < prog.code[i].num_Operands(); ++o) {
        if (prog.code[i].getOperand(o) == mappedRootPc) { prog.code[i].setOperand(o, rootSubstPc); }
      }
    }

    IF_VERBOSE {
      std::cerr << "-- after insert -- \n";
      prog.dump();
    }
    prog.compact();
  }
};


using RuleVec = std::vector<Rule>;
// create a basic rule set
static
RuleVec
BuildRules() {
  RuleVec rules;

  // (%b + %a) - %b --> %a
  {
    // try some matching
    Program lhs (2, {Statement(OpCode::Add, -2, -1), Statement(OpCode::Sub, 0, -2), build_ret(1) });
    Program rhs (1, {build_ret(-1)});
    rules.emplace_back(lhs, rhs);
  }

  // commutative rules (oc %a %b --> oc %b %a)
  for_such(IsCommutative, [&](OpCode oc){
    rules.emplace_back(
      Program(2, {Statement(oc, -1, -2), build_ret(0)}),
      Program(2, {Statement(oc, -2, -1), build_ret(0)})
    );
  });

  // associative rules (oc (%a %b) %c --> %a (%b %c)
  for_such(IsAssociative, [&](OpCode oc){
    rules.emplace_back(
      Program(3, {Statement(oc, -1, -2),
                  Statement(oc, 0, -3),
                  build_ret(1)
                  }),
      Program(3, {Statement(oc, -2, -3),
                  Statement(oc, -1, 0),
                  build_ret(1)
                  })
    );
  });

  // %a * (%b + %c) --> (%a * %b) + (%a * %c)
  rules.emplace_back(
      Program(3, {Statement(OpCode::Add, -2, -3),
                  Statement(OpCode::Mul, -1, 0),
                  build_ret(1)
                  }),
      Program(3, {Statement(OpCode::Mul, -1, -2),
                  Statement(OpCode::Mul, -2, -3),
                  Statement(OpCode::Add, 0, 1),
                  build_ret(2)
                  })
  );

  // (%a <oc> neutral) --> %a
  for_such(HasNeutralRHS, [&](OpCode oc){
    Data neutral = GetNeutralRHS(oc);
    rules.emplace_back(
      Program(1, {build_const(neutral),
                  Statement(oc, -1, 0),
                  build_ret(1)
                  }),
      Program(1, {build_ret(-1)
                  })
    );
  });

  //  paranoid rule consistency checking
  for (int i = 0; i < rules.size(); ++i) {
    const auto & rule = rules[i];
    bool lhs = true;
    auto handler = [&rule,&lhs](int pc, std::string msg) -> bool {
      std::cerr << "Error at " << pc << " : " <<msg << "\n";
      return true;
    };

    lhs = true; if (!rule.lhs.verify(handler)) { std::cerr << "in rule lhs: " << i << "\n"; exit(-1); }
    lhs = false; if (!rule.rhs.verify(handler))  { std::cerr << "in rule rhs: " << i << "\n"; exit(-1); }
  }

  // arithmetic simplifaction rules
  return rules;
}

struct RPG {
  int numParams;

  struct Elem {
    int numUses;
    int valIdx; // instruction or arg

    /// Elem()
    /// : numUses(0)
    /// , valIdx(0)
    /// {}

    Elem(int _numUses, int _valIdx) : numUses(_numUses), valIdx(_valIdx) {}
    bool operator< (const Elem & right) const {
      return numUses > right.numUses; // prioritize unused elements
    }
  };

  std::uniform_real_distribution<float> constantRand;

  struct Sampler {
    std::vector<int> unused;
    std::priority_queue<Elem, std::vector<Elem>> opQueue;

    // value may be used but does not have to (arguments)
    void addOptionalUseable(int valIdx) {
      // std::cerr << "OPT " << valIdx << "\n";
      opQueue.emplace(0, valIdx);
    }

    // value has to be used (instruction)
    void addUseable(int valIdx) {
      // std::cerr << "MUST USE: " << valIdx << "\n";
      unused.push_back(valIdx);
    }

    // fetch a random operand and increase its use count
    int acquireOperand(int distToLimit) {

      // check whether its safe to peek to already used operands at this point
      if (unused.size() > 0) {
        bool allowPeek = !opQueue.empty() && (distToLimit > unused.size());
        std::uniform_int_distribution<int> opRand(0, unused.size() - 1 + allowPeek);

        // pick element from unused vector
        int idx = opRand(randGen);
        // std::cerr << "UNUED ID " << idx << "\n";
        assert(idx >= 0);
        if (idx < unused.size()) {
          int valIdx = unused[idx];
          unused.erase(unused.begin() + idx);
          opQueue.emplace(1, valIdx); // add to used-tracked queue for potential re-use
          return valIdx;
        }
      }

      // Otw, take element from queue
      Elem elem = opQueue.top();
      opQueue.pop(); // TODO add randomness
      int valIdx = elem.valIdx;
      elem.numUses++;
      opQueue.push(elem); // re-insert element

      return valIdx;
    }

    bool empty() const { return unused.empty() && opQueue.empty(); }
  };

  std::vector<Data> constVec; // recognized constants in the match rules
  const double pConstant = 0.10;

  void collectConstants(const Program & prog, std::set<Data> & seen) {
    for (auto & stat : prog.code) {
      if (stat.isConstant()) {
        if (seen.insert(stat.getValue()).second) {
          constVec.push_back(stat.getValue());
        }
      }
    }
  }

  RPG(const RuleVec & rules, int _numParams)
  : numParams(_numParams)
  {
    std::set<Data> seen;
    for (auto & rule : rules) {
      collectConstants(rule.lhs, seen);
      collectConstants(rule.rhs, seen);
    }
    std::cerr << "found " << constVec.size() << " different constants in rule set!\n";
  }

  Program
  generate(int length) {
    Program P(numParams, {});
    P.code.reserve(length);

    Sampler S;
    // wrap all arguments in pipes
    for (int a = 0; a < numParams; ++a) {
      P.push(build_pipe(-a - 1));
      S.addOptionalUseable(a);
    }

    int s = numParams;
    for (int i = 0; i < length - 1; ++i, ++s) {
      if (S.empty() || (constantRand(randGen) <= pConstant)) {
        // random constant
        std::uniform_int_distribution<int> constIdxRand(0, constVec.size() - 1);
        int idx = constIdxRand(randGen);
        P.push(build_const(constVec[idx]));

      } else {
        // pick random opCode and operands
        int beginBin = (int) OpCode::Begin_Binary;
        int endBin = (int) OpCode::End_Binary;

        std::uniform_int_distribution<int> ocRand(beginBin, endBin);
        OpCode oc = (OpCode) ocRand(randGen);
        int distToLimit = length - 1 - i;
        int firstOp = S.acquireOperand(distToLimit);
        int sndOp = S.acquireOperand(distToLimit);
        P.push(Statement(oc, firstOp, sndOp));
      }

      // publish the i-th instruction as useable in an operand position
      S.addUseable(s);
    }

    P.push(build_ret(length - 2));

    assert(P.verify());

    return P;
  }
};



struct Mutator {
  const RuleVec & rules;

  Mutator(const RuleVec & _rules)
  : rules(_rules)
  {}

  void mutate(Program & P, int steps) const {
    auto handler=[](int pc, int ruleId, bool leftMatch, const Program & P) { }; // TODO global mutation cache
    mutate(P, steps, handler);
  }

  void mutate(Program & P, int steps, std::function<void(int pc, int ruleId, bool leftMatch, const Program & P)> handler) const {
    for (int i = 0; i < steps; ) {
      // pick a random pc
      std::uniform_int_distribution<int> pcRand(0, P.size() - 1);
      int pc = pcRand(randGen);

      // pick a random rule
      std::uniform_int_distribution<int> flipRand(0, 1);
      bool leftMatch = flipRand(randGen) == 0;

      // number of applicable rules to skip
      std::uniform_int_distribution<int> ruleRand(0, rules.size() - 1);
      int numSkips = 0;//ruleRand(randGen);

      int32_t holes[16];

      std::cerr << "(" << pc << ", " << leftMatch << ", " << numSkips << ")\n";
      // check if any rule matches
      bool hasMatch = false;
      int ruleIdx = 0;
      for (int t = 0; t < rules.size(); ++t) {
        if (rules[t].match(leftMatch, P, pc, holes)) {
          hasMatch = true;
          ruleIdx = t;
          break;
        }
      }

      // no rule matches -> pick different rule
      if (!hasMatch) continue;

      rules[ruleIdx].dump();

      for (int skip = 1; skip < numSkips; ) {
        ruleIdx = (ruleIdx + 1) % rules.size();
        if (rules[ruleIdx].match(leftMatch, P, pc, holes)) {
          ++skip;
        }
      }

      // supplement holes
      if (!rules[ruleIdx].removesHoles(leftMatch)) {
        const auto & lhs = rules[ruleIdx].getMatchProg(leftMatch);
        const auto & rhs = rules[ruleIdx].getRewriteProg(leftMatch);

        int lowestVal = -(P.num_Params()) - 1;
        int highestVal = pc - rhs.size() - 1;
        assert(lowestVal <= highestVal);
        std::uniform_int_distribution<int> opRand(lowestVal, highestVal);

        for (int h = lhs.num_Params(); h < rhs.num_Params(); ++h) {
          holes[h] = opRand(randGen);
        }

        // TODO allow experssion invention
      }

      // apply rewrite
      rules[ruleIdx].rewrite(leftMatch, P, pc, holes);

      IF_DEBUG if (!P.verify()) {
        P.dump();
        abort();
      }

      ++i;
    }
  }
};


} // namespace apo



using namespace apo;

void
RunTests() {
  auto rules = BuildRules();

  {
    int32_t holes[4];
    std::cerr << "\nTEST: Shrinking re-write:\n";
    Program prog(2, {
        Statement(OpCode::Add, -1, -2),
        Statement(OpCode::Nop, 0, 0),
        Statement(OpCode::Sub, 0, -1),
        Statement(OpCode::Return, 2)
    });
    // define a simple program
    prog.compact();
    prog.dump();

    bool ok = rules[0].match(true, prog, 1, holes);
    assert(ok);

    // rewrite test
    rules[0].rewrite(true, prog, 1, holes);
    std::cerr << "after rewrite:\n";
    prog.dump();
  }

  {
    std::cerr << "\nTEST: Expanding re-write:\n";
    Program prog(2, {
        build_pipe(-1),
        Statement(OpCode::Mul, -1, 0),
        build_ret(0)
    });

    // define a simple program
    prog.compact();
    prog.dump();

    int32_t holes[4];
    bool ok = rules[0].match(false, prog, 0, holes);
    assert(ok);

    holes[0] = -3;
    holes[1] = -2;

    // rewrite test
    rules[0].rewrite(false, prog, 0, holes);
    std::cerr << "after rewrite:\n";
    prog.dump();
  }

  std::cerr << "END_OF_TESTS\n";
}

int main(int argc, char ** argv) {

  // RunTests();
  // return 0;

  RuleVec rules = BuildRules();
  std::cerr << "Loaded " << rules.size() << " rules!\n";

  std::cerr << "Generating some random programs:\n";

  const int stubLen = 3;
  const int mutSteps = 3;

  RPG rpg(rules, 3);
  Mutator mut(rules);

  for (int i = 0; i < 1; ++i) {
    Program p = rpg.generate(stubLen);
    std::cerr << "Rand " << i << " ";
    p.dump();
    mut.mutate(p, mutSteps);
    std::cerr << "Mutated " << i << " ";
    p.dump();
  }
}

