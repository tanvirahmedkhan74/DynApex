LLVM_CONFIG = llvm-config
CLANG = clang++

CXXFLAGS := -g `$(LLVM_CONFIG) --cxxflags`
LDFLAGS := `$(LLVM_CONFIG) --ldflags --libs core` -lpthread

PASS_SRC = src/LoopExtractor.cpp
PASS_SO = LoopExtractor.so
MAIN_SRC = src/main.cpp
MAIN_LL = main.ll

all: $(PASS_SO) $(MAIN_LL)

$(PASS_SO): $(PASS_SRC)
	$(CLANG) -shared -fPIC $(CXXFLAGS) $< -o $@

$(MAIN_LL): $(MAIN_SRC)
	$(CLANG) -S -emit-llvm $(CXXFLAGS) $< -o $@

test: $(PASS_SO) $(MAIN_LL)
	opt -load-pass-plugin=./$(PASS_SO) -passes="loop-extract" < $(MAIN_LL) > /dev/null

clean:
	rm -f $(PASS_SO) $(MAIN_LL)
