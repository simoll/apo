CXX=g++ -O0 -g -Isrc/

apo: src/apo.cpp
	$(CXX) $^ -o $@
