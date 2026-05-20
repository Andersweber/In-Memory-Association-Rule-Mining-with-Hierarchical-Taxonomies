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
CXX           ?= g++
CXXFLAGS_FAST := -std=c++17 -O3 -march=native -I$(INC)
LDFLAGS       := -L$(LIB) -larrow -larrow_dataset -lparquet -Wl,-rpath,$(LIB)

apriori_cumulate_cpp: Apriori_Cumulate_CPP.cpp apriori.hpp association_rules.hpp fpcommon.hpp
	$(CXX) $(CXXFLAGS_FAST) Apriori_Cumulate_CPP.cpp -o apriori_cumulate_cpp $(LDFLAGS)

clean:
	rm -f apriori_cumulate_cpp
