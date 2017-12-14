TFLOW=/usr/lib/python3.6/site-packages/tensorflow

# INC=-I${TFLOW}/include
# LIBS=-ltensorflow_framework
# LIBPATH=-L${TFLOW}

CFLAGS=${INC} ${LIBPATH} ${LIBS}

CXX=g++ -O0 -g -Isrc/


apo: src/apo.cpp
	$(CXX) ${CFLAGS} $^ -o $@
