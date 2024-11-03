#ifndef ARRAY_SUM_H
#define ARRAY_SUM_H

#ifdef __cplusplus
extern "C" {
#endif

void array_sum(float* A, float* B, float* C, int n, const char* method);

void sumArrayWithAVX(float* A, float* B, float* C, int N);
void sumArrayWithOpenMP(float* A, float* B, float* C, int N);

#ifdef __cplusplus
}
#endif

#endif
