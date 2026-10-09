# Path to your Conda environment (must contain Arrow/Parquet headers and libs).
# Override at the command line:
#   make CONDA_ENV=$CONDA_PREFIX apriori_cumulate_cpp
# On SLURM, load the arrow module first, then pass CONDA_ENV=$CONDA_PREFIX.
ifndef CONDA_ENV
  $(error CONDA_ENV is not set. Run: make CONDA_ENV=$$CONDA_PREFIX apriori_cumulate_cpp)
endif

INC := $(CONDA_ENV)/include
LIB := $(CONDA_ENV)/lib

# Use g++ by default (works on Linux clusters); override with CXX=clang++ on macOS.
CXX    ?= g++

# C++ standard: c++17 matches the paper and the HPC Arrow build.
# Arrow >= 21 headers require C++20 features (std::span, std::bit_width);
# if compilation fails inside Arrow headers, rebuild with:
#   make CONDA_ENV=... CXXSTD=c++20 apriori_cumulate_cpp
CXXSTD ?= c++17

# Transaction capacity of the per-token std::bitset (paper §4.4/§6.3).
# Must be >= the number of transactions in the dataset; the binary aborts
# with a clear message otherwise. Support counting only touches the live
# word prefix, so a generous value does not slow down the algorithm core.
#   make CONDA_ENV=... MAX_TRANSACTIONS=2097152 apriori_cumulate_cpp
MAX_TRANSACTIONS ?= 1048576

CXXFLAGS_FAST := -std=$(CXXSTD) -O3 -march=native \
                 -DMAX_TRANSACTIONS=$(MAX_TRANSACTIONS) -I$(INC)
LDFLAGS       := -L$(LIB) -larrow -larrow_dataset -lparquet -Wl,-rpath,$(LIB)

apriori_cumulate_cpp: Apriori_Cumulate_CPP.cpp apriori.hpp association_rules.hpp fpcommon.hpp
	$(CXX) $(CXXFLAGS_FAST) Apriori_Cumulate_CPP.cpp -o apriori_cumulate_cpp $(LDFLAGS)

clean:
	rm -f apriori_cumulate_cpp