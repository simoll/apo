TFLOW=/usr/lib/python3.6/site-packages/tensorflow


SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(patsubst src/%.cpp,build/%.o, ${SOURCES})
HEADERS=$(wildcard include/apo/*.h) $(wildcard include/apo/*/*.h)


INC=-Isrc/ \
    -Iinclude \
    -I${TFLOW}/include \
    -I${TFLOW}/include/external/nsync/public \
    -I$(TFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src

LIBS=-ltensorflow_framework \
    $(TFLOW)/python/_pywrap_tensorflow_internal.so \
    $(TFLOW)/python/framework/fast_tensor_util.so \
    -lcuda \
    -lpython3
LIBPATH=-L${TFLOW}

LDFLAGS=-Wl,--allow-multiple-definition -Wl,--whole-archive ${LIBPATH} ${LIBS}

CFLAGS=${INC}

PYTHON=python3

# OPTFLAGS=-O3 -DNDEBUG -fopenmp -g
# OPTFLAGS=-O3 -DNDEBUG -g
OPTFLAGS=-O0 -g -fsanitize=address

CXX=clang++ -std=c++14 ${OPTFLAGS}
# CXX=clang++ -std=c++14 -O3 -Isrc/ 


apo: $(OBJECTS) build/apo_graph.pb Makefile
	$(CXX) ${CFLAGS} ${OBJECTS} -o $@ $(LDFLAGS)

build/%.o: src/%.cpp Makefile $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) ${CFLAGS} $< -c -o $@ 

build/apo_graph.pb: src/model.py model.conf $(HEADERS)
	$(PYTHON) src/model.py

.PHONY: clean
clean:
	rm -rf apo build/*
