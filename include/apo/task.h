#ifndef APO_TASK_H
#define APO_TASK_H

#include <thread>
#include <mutex>

// tasking API
// tasks must be employed such that the code is correct if tasks execute immediately (and join/detach are no-ops)
namespace apo {
#ifdef APO_ASYNC_TASKS
  // enable asynchronous joinable tasks
  using Task = std::thraad;
  using TaskMutex = std::mutex;

  struct MutexGuard {
    std::lock_guard<std::mutex> _mutex;
    MutexGuard(std::mutex & mutex) : _mutex(mutex) {}
  };

  template<class T>
  struct Lock_guard {
    std::lock_guard<T> _guard;
    lock_guard(T & t) : _guard(t) {}
  };

#else
  // no asynchonous tasks
  struct Task {
    Task() {}

    // run immediately
    Task(std::function<void()> bodyFn) {
      bodyFn();
    }

    bool joinable() const { return false; }
    void join() {}
    void detach() {}
  };

  struct TaskMutex {};

  struct Mutex_guard {
     Mutex_guard(const TaskMutex & mutex) {}
  };
#endif
}

#endif // APO_TASK_H
