#pragma once

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

// Container hack
#define RANGE(x) ((x).begin()), ((x).end())
#define SUM(x)   (std::accumulate(RANGE(x), .0))

// Random hack
#include <chrono>
#include <random>
#define CLOCK (std::chrono::system_clock::now().time_since_epoch().count())
static std::mt19937 _rng(CLOCK);
static std::uniform_real_distribution<double> _unif01;

static int _jxr = 1234567;

inline static float Unif01() {
  //return _unif01(_rng);
  _jxr ^= (_jxr << 13);
  _jxr ^= (_jxr >> 17);
  _jxr ^= (_jxr << 5);
  return (_jxr & 0x7fffffff) * 4.6566125e-10;
}

inline static int Dice(int n) {
  return (int)(Unif01() * n);
}

// Eigen
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#define EIGEN_DEFAULT_IO_FORMAT \
        Eigen::IOFormat(FullPrecision,1," "," ","","","[","]")
#define EREAL(x) ((x).cast<real>())
#include "Eigen/Dense"
using real = float;
using EMatrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using EVector = Eigen::Matrix<real, Eigen::Dynamic, 1,              Eigen::ColMajor>;
using EMAtrix = Eigen::Array <real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using EArray  = Eigen::Array <real, Eigen::Dynamic, 1,              Eigen::ColMajor>;
using IMatrix = Eigen::Matrix<int,  Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using IVector = Eigen::Matrix<int,  Eigen::Dynamic, 1,              Eigen::ColMajor>;
using IMAtrix = Eigen::Array <int,  Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using IArray  = Eigen::Array <int,  Eigen::Dynamic, 1,              Eigen::ColMajor>;
