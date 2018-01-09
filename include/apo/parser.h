#ifndef APO_PARSER_H
#define APO_PARSER_H

#include <map>

namespace apo {

class Parser {
  std::map<std::string ,std::string> items;

public:
  Parser(std::string fileName);
  Parser();

  template<typename T>
  T get(const std::string & key);
};

} // namespace apo

#endif // APO_PARSER_H
