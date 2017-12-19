TFLOW=/usr/lib/python3.6/site-packages/tensorflow

# INC=-I${TFLOW}/include
# LIBS=-ltensorflow_framework
# LIBPATH=-L${TFLOW}

CFLAGS=${INC} ${LIBPATH} ${LIBS}

PYTHON=python3

CXX=clang++ -std=c++14 -O3 -g -Isrc/ -fsanitize=address
# CXX=clang++ -std=c++14 -O3 -Isrc/ 


apo: src/apo.cpp build/apo_graph.pb Makefile
	$(CXX) ${CFLAGS} $< -o $@

build/apo_graph.pb: src/model.py
	$(PYTHON) src/model.py

.PHONY: clean
clean:
	rm -f apo
