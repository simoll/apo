#ifndef APO_PARSER_H
#define APO_PARSER_H

#include <map>

namespace apo {

class Parser {
  std::map<std::string ,std::string> items;

public:
  Parser(std::string fileName);
  Parser();

  // read value from config file, return default value
  template<typename T>
  T get(const std::string & key, const T defVal=T());

  // read value from config file, fail if key does not exist
  template<typename T>
  T get_or_fail(const std::string & key);
};

} // namespace apo

#endif // APO_PARSER_H
