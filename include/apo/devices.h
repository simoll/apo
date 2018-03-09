#ifndef APO_DEVICES_H
#define APO_DEVICES_H

#include <vector>
#include <string>
#include <map>
#include <cassert>

#include <thread>
#include <functional>
#include <cmath>

namespace apo {

struct Device {
  std::string tfDevice; // tensorflow device name
  std::string tower; // tower op suffix
  int rating; // performance rating (used for load distribution)
  double relRating; // relative rating in this category
  double relStart; // rating start (for faster distributino)
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


// distribute a parallel workload onto a vector of devices
static void
Distribute(const DeviceVec & devices, int numElements, std::function<void(int deviceId, int startSlice, int endSlice)> userFunc, const int costFactor=128) {
  if ((devices.size() > 1) && // there are multiple devices
     (numElements >= devices.size() * costFactor)  // there is enough work to offset the distribution overhread
  ) {
    std::vector<std::thread> threads;
    threads.reserve(devices.size());

    // there is a nested OpenMP loop in _ModelDriven. this is a poor mans parallel (outer) loop
    for (int i = 0; i < devices.size(); ++i) {
      const auto & dev = devices[i];
      int startSlice = (int) floor(dev.relStart * numElements);
      int endSlice = (i + 1 == devices.size()) ? numElements :  (int) floor((dev.relStart + dev.relRating) * numElements);
      threads.push_back(std::thread(
          [i, startSlice, endSlice, &userFunc](){
            userFunc(i, startSlice, endSlice);
          }
      ));
    }

    for (int i = 0; i < threads.size(); ++i) {
      auto & thread = threads[i];

      // wait for results
      thread.join();
    }
  } else {
    userFunc(0, 0, numElements);
  }
}

} // namespace apo

#endif // APO_DEVICES_H
