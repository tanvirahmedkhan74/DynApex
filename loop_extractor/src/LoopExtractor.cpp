#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassPlugin.h"  // For PassPluginLibraryInfo
#include "llvm/Passes/PassBuilder.h" // For PassBuilder

using namespace llvm;

struct LoopExtractor : public PassInfoMixin<LoopExtractor> {
    PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                          LoopStandardAnalysisResults &AR, LPMUpdater &) {
        errs() << "Loop Found:\n";
        L.print(errs(), /*Header=*/true);
        return PreservedAnalyses::all();
    }
};

// Register the pass
llvm::PassPluginLibraryInfo getLoopExtractorPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "LoopExtractor", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, LoopPassManager &LPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "loop-extract") {
                            LPM.addPass(LoopExtractor());
                            return true;
                        }
                        return false;
                    });
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return getLoopExtractorPluginInfo();
}
