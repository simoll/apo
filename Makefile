TFLOW=/usr/lib/python3.6/site-packages/tensorflow

# INC=-I${TFLOW}/include
# LIBS=-ltensorflow_framework
# LIBPATH=-L${TFLOW}

CFLAGS=${INC} ${LIBPATH} ${LIBS}

CXX=g++ -std=c++14 -O0 -g -Isrc/


apo: src/apo.cpp Makefile
	$(CXX) ${CFLAGS} $< -o $@

.PHONY: clean
clean:
	rm -f apo
