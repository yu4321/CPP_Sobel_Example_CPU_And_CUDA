/// 2015110758 류영석 20210409

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "..\usr\include\GL\freeglut.h";
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define FILE_NAME "../image2.raw"

const int Size = 512;
unsigned char* pSrcImage = NULL;
unsigned char* pOutImage = NULL;
bool flag = false;

void Render();
void Reshape(int w, int h);
void Keyboard(unsigned char key, int x, int y);
void SobelEdge();

#pragma region CUDA 함수변수들

#define TILE_WIDTH 1024

__global__ void SobelEdgeKernel(float* cpSobelResult, unsigned char* pSrcImage);
__global__ void SobelApplyKernel(float* pSobelResult, unsigned char* pOutImage, int min, int max);
__global__ void GetMaxKernel(float* pSobelResult, float* arrMinMax);

#pragma endregion

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB);

	glutInitWindowSize(Size, Size);
	glutCreateWindow("Sobel Edge Detector(GPU)");

	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Keyboard);

	pSrcImage = new unsigned char[Size * Size];
	pOutImage = new unsigned char[Size * Size];

	FILE* infile;
	fopen_s(&infile, FILE_NAME, "rb");
	fread(pOutImage, sizeof(unsigned char), Size * Size, infile);
	for (int i = 0; i < Size * Size; ++i) {
		pSrcImage[i] = pOutImage[Size * Size - i - 1];
	}
	fclose(infile);

	clock_t st = clock();
	SobelEdge();
	printf("Elapsed time = %u ms\n", clock() - st);

	glutMainLoop();

	delete[] pSrcImage;
	delete[] pOutImage;
	return 0;
}

/// <summary>
/// Sobel 관련 기본 함수. 모든 CUDA 연산은 이 함수 안에서 창 띄우기 전에 끝나도록
/// </summary>
void SobelEdge()
{
	auto memorySizeFloat = sizeof(float) * Size * Size;
	auto memorySizeUChar = Size * Size * sizeof(unsigned char);

	// GPU 사용 설정
	cudaSetDevice(0);

	// 소벨 결과를 담을 호스트 변수
	float* pSobelResult = new float[Size * Size];
	memset(pSobelResult, 0, memorySizeFloat);

	// pSobelResult의 값을 복사하여 디바이스에서 사용될 변수
	float* cpSobelResult;
	cudaMalloc((void**)&cpSobelResult, memorySizeFloat);
	cudaMemcpy(cpSobelResult, pSobelResult, memorySizeFloat, cudaMemcpyHostToDevice);

	// pSrcImage또는 pOutImage를 복사하여 디바이스에서 사용될 변수
	unsigned char* cpyImage;
	cudaMalloc((void**)&cpyImage, memorySizeUChar);
	cudaMemcpy(cpyImage, pSrcImage, memorySizeUChar, cudaMemcpyHostToDevice);

	// 여러 차례 계산해보았으나, 다차원 블록이 속도가 최소 10 ~ 최대 60ms까지 차이가 나서 1차원/1차원 방법을 사용했습니다.
	// 주석처리된 다차원 블록 계산시 실행시간 최저 133 최대 190, 현재 방법 최저 120 최대 144
	/*auto tWidth = (Size - 1) / TILE_WIDTH + 1;
	dim3 gridDim(tWidth, tWidth);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH);*/
	dim3 gridDim(Size, 1);
	dim3 blockDim(Size, 1);

	// 소벨 계산 커널함수 실행 후 완료 대기. 이후 호스트 변수로 결과 복사
	SobelEdgeKernel << <gridDim, blockDim, 1 >> > (cpSobelResult, cpyImage);
	cudaDeviceSynchronize();
	cudaMemcpy(pSobelResult, cpSobelResult, memorySizeFloat, cudaMemcpyDeviceToHost);

	// 일반적인 방법으로 커널 함수로 바꿀 시 필터 결과에 문제 생김. min max 변수 공유 문제로 보임
	float min = 10000000.0f, max = -10000000.0f;
	for (int i = 1; i < Size - 1; i++) {
		for (int j = 1; j < Size - 1; j++) {
			int idx = i * Size + j;
			min = (pSobelResult[idx] < min) ? pSobelResult[idx] : min;
			max = (pSobelResult[idx] > max) ? pSobelResult[idx] : max;
		}
	}

	// cpyImage 재사용
	cudaFree(cpyImage);
	cudaMalloc((void**)&cpyImage, memorySizeUChar);
	cudaMemcpy(cpyImage, pOutImage, memorySizeUChar, cudaMemcpyHostToDevice);

	// 소벨 적용 커널함수 실행 후 완료 대기. 이후 호스트 변수로 결과 복사
	SobelApplyKernel << < gridDim, blockDim, 1 >> > (cpSobelResult, cpyImage, min, max);
	cudaDeviceSynchronize();
	cudaMemcpy(pOutImage, cpyImage, memorySizeUChar, cudaMemcpyDeviceToHost);

	// 전체 변수들 메모리 해제 및 CUDA 계산 종료.
	delete[] pSobelResult;
	cudaFree(cpSobelResult);
	cudaFree(cpyImage);
	cudaDeviceReset();
}

void Render()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	if (flag == true) {
		glDrawPixels(Size, Size, GL_LUMINANCE, GL_UNSIGNED_BYTE, pOutImage);
	}
	else {
		glDrawPixels(Size, Size, GL_LUMINANCE, GL_UNSIGNED_BYTE, pSrcImage);
	}

	glFinish();
}

void Reshape(int w, int h)
{
	glViewport(0, 0, w, h);
}

void Keyboard(unsigned char key, int x, int y)
{
	// 27=esc
	if (key == 27) {
		//glutLeaveMainLoop();
		exit(-1);
	}

	if (key == 's') {
		flag = !flag;
	}

	glutPostRedisplay();
}

#pragma region kernel functions
/// <summary>
/// 소벨 엣지 검출 커널 함수. 3중 루프문에서 바깥 2중 루프문을 간략화.  
/// </summary>
/// <param name="cpSobelResult"></param>
/// <param name="pSrcImage"></param>
/// <returns></returns>
__global__ void SobelEdgeKernel(float* cpSobelResult, unsigned char* pSrcImage)
{
	int MaskSobelX[] = {
		-1, 0, 1,
		-2,0,2,
		-1,0,1
	};

	int MaskSobelY[] = {
		1,2, 1,
		0,0,0,
		-1,-2,-1
	};

	int i = blockIdx.x;
	int j = threadIdx.x;
	//int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	//int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	// 지워서 원본 for문보다 아래위로 1씩 더 보게 되면 끄트머리 1픽셀 백화
	if (i <= 0 || j <= 0 || i >= Size - 1 || j >= Size - 1) {
		return;
	}

	int Gx = 0, Gy = 0;
	// 누적하는 값이므로 변수 공유 대책 없이는 커널화 하지 않음
	for (int k = 0; k < 9; ++k) {
		int r = k / 3, c = k % 3;
		int idx = (i + r - 1) * Size + j + c - 1;
		Gx = Gx + MaskSobelX[k] * pSrcImage[idx];
		Gy = Gy + MaskSobelY[k] * pSrcImage[idx];
	}

	cpSobelResult[i * Size + j] = sqrtf(Gx * Gx + Gy * Gy);
}

/// <summary>
/// 소벨 필터 값에 따라 이미지에 적용하는 커널 함수.
/// </summary>
/// <param name="pSobelResult"></param>
/// <param name="pOutImage"></param>
/// <param name="min"></param>
/// <param name="max"></param>
/// <returns></returns>
__global__ void SobelApplyKernel(float* pSobelResult, unsigned char* pOutImage, int min, int max) {
	int i = blockIdx.x;
	int j = threadIdx.x;

	//int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	//int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	if (i <= 1 || j <= 1 || i >= Size - 1 || j >= Size) {
		return;
	}
	int idx = i * Size + j;
	float t = (pSobelResult[idx] - min) / (max - min);
	pOutImage[idx] = (unsigned char)(255 * t);
}

/// <summary>
/// SobelEdge 함수에서 min max 값 구할 때 사용하려 했던 함수. 변수 공유 관련 적용 안했으므로 미사용
/// </summary>
/// <param name="pSobelResult"></param>
/// <param name="arrMinMax"></param>
/// <returns></returns>
__global__ void GetMaxKernel(float* pSobelResult, float* arrMinMax) {

	int i = blockIdx.x;
	int j = threadIdx.x;

	/*int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;*/

	if (i <= 1 || j <= 1 || i >= Size - 1 || j >= Size - 1) {
		return;
	}

	int idx = i * Size + j;

	arrMinMax[0] = (pSobelResult[idx] < arrMinMax[0]) ? pSobelResult[idx] : arrMinMax[0];
	arrMinMax[1] = (pSobelResult[idx] > arrMinMax[1]) ? pSobelResult[idx] : arrMinMax[1];
}
#pragma endregion
