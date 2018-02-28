#include "apo/devices.h"

#include <sstream>
#include <functional>
#include <fstream>
#include <iostream>

namespace apo {

static void
ForToken(std::string text, char sep, std::function<void(std::string)> handler) {
  std::stringstream ss(text);
  std::vector<std::string> result;

  while( ss.good() )
  {
    std::string substr;
      getline( ss, substr, sep );
      handler(substr);
      // result.push_back( substr );
  }
}

Devices::Devices(std::string deviceFile)
: taskDevices()
{
  std::ifstream in(deviceFile);
  std::string devName, taskSet;
  int rating = 1;

  in >> devName >> taskSet >> rating;
  while (!devName.empty()) {
    // parse device mapping
    if (devName[0] != '#') {
      ForToken(taskSet, ',', [this, devName, rating](std::string taskToken) {
          taskDevices[taskToken].push_back(Device{devName, rating});
      });
    }

    // next device in list
    devName = "";
    in >> devName >> taskSet >> rating;
  }

  dump();
}

static
std::ostream&
DumpVec(const DeviceVec & devVec) {
 for (const auto & dev : devVec) std::cerr << " " << dev.tower << ",r=" << dev.rating;
 return std::cerr;
}

void
Devices::dump() const {
  std::cerr << "Devices {\n";
  for (const auto & itTask : taskDevices) {
    std::cerr << itTask.first << " -> "; DumpVec(itTask.second) << "\n";
  }
  std::cerr << "}\n";
}


} // namespace apo
