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
  // parse device file
  std::ifstream in(deviceFile);
  std::string devName, towerName, taskSet;
  int rating = 1;

  in >> devName >> towerName >> taskSet >> rating;
  while (!devName.empty()) {
    // parse device mapping
    if (devName[0] != '#') {
      ForToken(taskSet, ',', [this, devName, towerName, rating](std::string taskToken) {
          taskDevices[taskToken].push_back(Device{devName, towerName, rating, 0.0, 0.0});
      });
    }

    // next device in list
    devName = "";
    in >> devName >> towerName >> taskSet >> rating;
  }

  // compute relative rating per category
  for (auto & itTask : taskDevices) {
    size_t totalRating = 0;
    for (const auto & dev : itTask.second) {
      totalRating += dev.rating;
    }

    double relBase = 0.0;
    for (auto & dev : itTask.second) {
      dev.relRating = dev.rating / (double) totalRating;
      dev.relStart = relBase;
      relBase += dev.relRating;
    }
  }

  // TODO make conditional on verbose mode
  dump();
}

static
std::ostream&
DumpVec(const DeviceVec & devVec) {
 for (const auto & dev : devVec) std::cerr << " " << dev.tower << "@" << dev.tfDevice << ",r=" << dev.rating << "(" << dev.relRating << ")";
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
