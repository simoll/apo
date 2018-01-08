#include "apo/parser.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace apo {

Parser::Parser(std::string fileName) {
  std::ifstream in(fileName);
  std::string key, value;

  in >> key >> value;
  while (!key.empty()) {
    // std::cerr << "K: " <<  key << " -> " << value << "\n";
    items[key] = value;
    key = "";
    in >> key >> value;
  }
}

Parser::Parser() {}

template<typename T>
T
Parser::get(const std::string & key) {
  if (!items.count(key)) return T();
  T val;
  std::stringstream str(items[key]);
  str >> val;
  return val;
}

template<>
std::string Parser::get(const std::string & key) {
  return items[key];
}

template int Parser::get<int>(const std::string & key);

} // namespace apo
