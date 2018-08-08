d: FORCE
	g++ -std=c++11 -O3 -ffast-math -g dijkstra3d.cpp -o d

test: FORCE
	g++ -std=c++11 -O3 -ffast-math -g test.cpp -o test

FORCE: 