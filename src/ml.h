#ifndef APO_ML_H
#define APO_ML_H

#include <string>

namespace tensorflow {
  class Session;
}

namespace apo {

  class Model {
  // tensorflow state
    static tensorflow::Session * session;
    static bool initialized;
    // initialize tensorflow
    static int init_tflow();

  // graph definition

  public:
    Model(const std::string & fileName);

    static void shutdown();
  };
}

#endif // APO_ML_H

