TFLOW=/usr/lib/python3.6/site-packages/tensorflow

SOURCES=$(wildcard src/*.cpp)

INC=-I${TFLOW}/include -Isrc/
# LIBS=-ltensorflow_framework
# LIBPATH=-L${TFLOW}

CFLAGS=${INC} ${LIBPATH} ${LIBS}

PYTHON=python3

CXX=clang++ -std=c++14 -O3 -g -Isrc/ -fsanitize=address
# CXX=clang++ -std=c++14 -O3 -Isrc/ 


apo: $(SOURCES) build/apo_graph.pb Makefile
	$(CXX) ${CFLAGS} ${SOURCES} -o $@

build/apo_graph.pb: src/model.py
	$(PYTHON) src/model.py

.PHONY: clean
clean:
	rm -f apo build/*
