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
Parser::get(const std::string & key, const T defVal) {
  if (!items.count(key)) return defVal;
  T val;
  std::stringstream str(items[key]);
  str >> val;
  return val;
}

template<>
std::string Parser::get(const std::string & key, const std::string defVal) {
  if (!items.count(key)) return defVal;
  return items[key];
}

template<typename T>
T
Parser::get_or_fail(const std::string & key) {
  if (!items.count(key)) {
    std::cerr << "Missing key " << key << ". Aborting!\n";
    abort();
  }

  T val;
  std::stringstream str(items[key]);
  str >> val;
  std::cerr << key << " = " << val << "\n";

  return val;
}


// explicit instantiations
template int Parser::get<int>(const std::string & key, const int defVal);
template int Parser::get_or_fail<int>(const std::string & key);

template size_t Parser::get<size_t>(const std::string & key, const size_t defVal);
template size_t Parser::get_or_fail<size_t>(const std::string & key);

template double Parser::get<double>(const std::string & key, const double defVal);
template double Parser::get_or_fail<double>(const std::string & key);

template std::string Parser::get_or_fail<std::string>(const std::string & key);

} // namespace apo

