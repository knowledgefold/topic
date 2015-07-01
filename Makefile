SHELL   =  /bin/bash
PROJECT := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

CXX      = g++
CXXFLAGS = -O3 -std=c++11 -Wall -Wno-deprecated-declarations

BIN = sparselda
SRC = $(wildcard *.cc)
HDR = $(wildcard *.h)
DYN = -lm -lrt

all: $(BIN)

Eigen:
	curl -sL http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2 | tar jx --strip=1 eigen-eigen-bdd17ee3b1b3/Eigen

$(BIN): Eigen $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) $(DYN) -o $@

clean:
	rm -f $(BIN)

.PHONY: all clean
