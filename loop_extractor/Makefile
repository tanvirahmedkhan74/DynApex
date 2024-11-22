LLVM_CONFIG = llvm-config-18
CLANG = clang-18
OPT = opt-18

CXXFLAGS := -g `$(LLVM_CONFIG) --cxxflags`
LDFLAGS := `$(LLVM_CONFIG) --ldflags --libs core` -lpthread

PASS_SRC = src/LoopExtractor.cpp
PASS_SO = LoopExtractor.so
MAIN_SRC = src/main.cpp
MAIN_LL = main.ll
DDG_DOT = ddg.main..dot
DDG_PNG = loop-dependency-graph.png

all: $(PASS_SO) $(MAIN_LL)

$(PASS_SO): $(PASS_SRC)
	$(CLANG) -shared -fPIC $(CXXFLAGS) $< -o $@

$(MAIN_LL): $(MAIN_SRC)
	$(CLANG) -S -emit-llvm $(CXXFLAGS) $< -o $@

test: $(PASS_SO) $(MAIN_LL)
	$(OPT) -load-pass-plugin=./$(PASS_SO) -passes="loop-extract" < $(MAIN_LL) > /dev/null

dot-graph: $(MAIN_LL)
	$(OPT) -disable-output -passes=dot-ddg $(MAIN_LL)
	dot -Tpng $(DDG_DOT) -o $(DDG_PNG)

clean:
	rm -f $(PASS_SO) $(MAIN_LL) $(DDG_DOT) $(DDG_PNG)