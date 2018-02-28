#ifndef APO_DEVICES_H
#define APO_DEVICES_H

#include <vector>
#include <string>
#include <map>
#include <cassert>

namespace apo {

struct Device {
  std::string tower;
  int rating;
};

using DeviceVec = std::vector<Device>;

class Devices {
  std::map<std::string, DeviceVec> taskDevices;

public:
  Devices(std::string deviceFile);
  const DeviceVec & getDevices(std::string taskSet) const {
    auto it = taskDevices.find(taskSet);
    assert((it != taskDevices.end()) && "no registered device for this task");
    return it->second;
  }

  void dump() const;
};

} // namespace apo

#endif // APO_DEVICES_H
