TFLOW=/usr/lib/python3.6/site-packages/tensorflow


SOURCES=$(wildcard src/*.cpp)
INC= -Isrc/ \
    -I${TFLOW}/include \
    -I${TFLOW}/include/external/nsync/public \
    -I$(TFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src

LDFLAGS=-Wl,--allow-multiple-definition -Wl,--whole-archive

LIBS=-ltensorflow_framework \
    $(TFLOW)/python/_pywrap_tensorflow_internal.so \
    $(TFLOW)/python/framework/fast_tensor_util.so \
    -lcuda \
    -lpython3
LIBPATH=-L${TFLOW}

    # $(TFLOW)/../pycuda/_driver.cpython-36m-x86_64-linux-gnu.so \

CFLAGS=${INC} ${LIBPATH} ${LIBS}

PYTHON=python3

CXX=clang++ -std=c++14 -O3 -g -Isrc/ #-fsanitize=address
# CXX=clang++ -std=c++14 -O3 -Isrc/ 


apo: $(SOURCES) build/apo_graph.pb Makefile
	$(CXX) ${CFLAGS} ${SOURCES} -o $@ $(LDFLAGS)

build/apo_graph.pb: src/model.py
	$(PYTHON) src/model.py

.PHONY: clean
clean:
	rm -f apo build/*
