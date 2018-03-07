#ifndef APO_TASK_H
#define APO_TASK_H

#include <thread>
#include <mutex>

#if defined(__GNUC__) && (__GNUC__ >= 4)
#define ATTR_WARN_UNUSED __attribute__ ((warn_unused_result))
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
#define ATTR_WARN_UNUSED _Check_return_
#else
#define ATTR_WARN_UNUSED
#endif


// tasking API
// tasks must be employed such that the code is correct if tasks execute immediately (and join/detach are no-ops)
namespace apo {
#ifdef APO_ASYNC_TASKS
  // enable asynchronous joinable tasks
  using Task = std::thread;
  using TaskMutex = std::mutex;

  struct Mutex_guard {
    std::lock_guard<std::mutex> _mutex;
    Mutex_guard(std::mutex & mutex) : _mutex(mutex) {}
  };

  template<class T>
  struct Lock_guard {
    std::lock_guard<T> _guard;
    Lock_guard(T & t) : _guard(t) {}
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
