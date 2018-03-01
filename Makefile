SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(patsubst src/%.cpp,build/%.o, ${SOURCES})
HEADERS=$(wildcard include/apo/*.h) $(wildcard include/apo/*/*.h)

include make.conf

ifndef TFLOW
  $(error TFLOW is not set: path to Tensorflow site-package)
endif

FLAGS=
# -DAPO_ENABLE_DER_CACHE # problematic .. disengaged

INC=-Isrc/ \
    -Iinclude \
    -I${TFLOW}/include \
    -I${TFLOW}/include/external/nsync/public \
    -I$(TFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src

LIBS=-ltensorflow_cc \
     -ltensorflow_framework \
     $(TFLOW)/contrib/rnn/python/ops/_lstm_ops.so \
     -lpthread

LIBPATH=-L${TFLOW}


$(info apo: Using tensorflow installation at ${TFLOW})

# enable TensorFlow on the GPU
# APO_ENABLE_CUDA:=1
ifeq ($(strip $(APO_ENABLE_CUDA)),1)
    $(info apo: Building with GPU support.)
    LIBS:=${LIBS} -lcuda
    FLAGS:=${FLAGS} -DAPO_ENABLE_CUDA
endif

# enable asynchronous model queries
# this should only be enabled if (APO_ENABLE_CUDA) is set as well and a GPU is available on the system
# otherwise openmp threads will compete with TensorFlows internal threads got the CPUs, helping nobody
ifeq ($(strip $(APO_ASYNC_TASKS)),1)
  $(info apo: Building with async task support.)
  FLAGS:=${FLAGS} -DAPO_ASYNC_TASKS
endif

LDFLAGS=-Wl,--allow-multiple-definition -Wl,--whole-archive ${LIBPATH} ${LIBS}

CFLAGS=${INC} 

CXX=clang++ -std=c++17 $(FLAGS) ${OPTFLAGS}
# -D_GLIBCXX_USE_CXX11_ABI=0

METAGRAPH=build/rdn.meta

apo: $(OBJECTS) ${METAGRAPH} Makefile make.conf
	$(CXX) ${CFLAGS} ${OBJECTS} ${LIBS} -o $@ $(LDFLAGS)

build/%.o: src/%.cpp Makefile $(HEADERS) make.conf
	mkdir -p $(dir $@)
	$(CXX) ${CFLAGS} $< -c -o $@ 

${METAGRAPH}: src/model.py model.conf devices.conf
	mkdir -p $(dir ${METAGRAPH})
	$(PYTHON) src/model.py

.PHONY: clean
clean:
	rm -rf apo build/**/*.o build/*.o

purge: clean
	rm -rf apo build/
