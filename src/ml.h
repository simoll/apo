#ifndef APO_ML_H
#define APO_ML_H

namespace tensorflow {
  class Session;
}

namespace apo {

  class Model {
    static tensorflow::Session * session;
    static bool initialized;
    // initialize tensorflow
    static int init_tflow();

  public:
    Model();

    static void shutdown();
  };
}

#endif // APO_ML_H

