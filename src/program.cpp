#include "apo/program.h"
#include <cassert>
#include <sstream>

namespace apo {

static bool
IsAlpha(char c) { return 'a' <= c && c <= 'z'; }

static bool
IsFigure(char c) { return '0' <= c && c <= '9'; }

// complete operator decoding logic (params or operand pc)
static bool
DecodeOp(const std::string token, node_t & out) {
  if (token.size() < 2) return false; // leading % plus some payload

  assert(token[0] == '%');
  char firstChar = token[1];

  // argument reference "%a"
  if (IsAlpha(firstChar)) {
    assert(token.size() == 2);
    out = -(firstChar - 'a') - 1;
    return true;
  }

  // proper pc "%XY"
  auto numText = token.substr(1, std::string::npos);
  out = atoi(numText.c_str());
  return true;
}

static bool
DecodeOpCode(std::string ocText, OpCode & oc) {
 for (int i = (int) OpCode::Begin_OpCode; i < (int) OpCode::End_OpCode; ++i) {
   oc = (OpCode) i;
   std::stringstream buffer;
   PrintOpCode(oc, buffer);
   if (buffer.str() == ocText) return true;
 }
 return false;
}

Program*
Program::Parse(std::istream & in) {
  std::vector<Statement> statVec;

  // Program (XYZ)
  // TODO parse prologue
  int pc = 0;
  for (std::string line; std::getline(in, line); ++pc) {
    // std::cerr << pc << " line \"" << line << "\"\n";
    std::stringstream numBuffer;

    std::stringstream tokenStream(line);
    std::string pcText;
    tokenStream >> pcText;

    // verify "<PC>:"
    node_t codePc = atol(pcText.c_str());
    assert((codePc == pc) && "sanity check"); // TODO relaxed parsing

    // first statement token (OpCode or constant value)
    std::string firstStatToken;
    tokenStream >> firstStatToken;
    // std::cerr << "\"" << firstStatToken << "\"" << "\n";

    // constant
    if (IsFigure(firstStatToken[0])) {
        data_t constVal = atol(firstStatToken.c_str()); // FIXME does atoi support full data_t???
        statVec.push_back(build_const(constVal));
        continue;
    }

    // decode opCode
    OpCode oc;
    bool matchedOC = DecodeOpCode(firstStatToken, oc);
    assert(matchedOC);

    // decode operands
    NodeVec operandVec;
    for (int o = 0; o < Num_Operands(oc); ++o) {
      std::string operandToken;
      tokenStream >> operandToken;
      node_t opIdx;
      bool ok = DecodeOp(operandToken, opIdx);
      assert(ok && "could not parse operand!");
      operandVec.push_back(opIdx);
    }

    // construct operation
    statVec.emplace_back(oc, operandVec);
  }

  // all good, instantiate
  return new Program(3, statVec); // TODO read num params from prologue
}

}
