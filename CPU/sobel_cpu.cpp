#include "..\usr\include\GL\freeglut.h";
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


const int Size = 512;
unsigned char* pSrcImage = NULL;
unsigned char* pOutImage = NULL;
bool flag = false;

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

void Render();
void Reshape(int w, int h);
void Keyboard(unsigned char key, int x, int y);

void SobelEdge();

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB);

	glutInitWindowSize(Size, Size);
	glutCreateWindow("Sobel Edge Detector(CPU)");

	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Keyboard);

	pSrcImage = new unsigned char[Size * Size];
	pOutImage = new unsigned char[Size * Size];
	
	FILE* infile;//= fopen("..\image2.raw", "rb");
	fopen_s(&infile, "../image2.raw", "rb");
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

void SobelEdge()
{
	float* pSobelResult = new float[Size * Size];
	memset(pSobelResult, 0, Size * Size * sizeof(float));

	//병렬화 가능
	for (int i = 1; i < Size - 1; i++) {
		for (int j = 1; j < Size - 1; j++) {
			int Gx = 0, Gy = 0;
			for (int k = 0; k < 9; ++k) {
				int r = k / 3, c = k % 3;
				int idx = (i + r - 1) * Size + j + c - 1;
				Gx = Gx + MaskSobelX[k] * pSrcImage[idx];
				Gy = Gy + MaskSobelY[k] * pSrcImage[idx];
			}
			pSobelResult[i * Size + j] = sqrtf(Gx * Gx + Gy * Gy);
		}
	}


	float min = 10000000.0f, max = -10000000.0f;
	for (int i = 1; i < Size-1; i++) {
		for (int j = 1; j < Size-1; j++) {
			int idx = i * Size + j;
			min = (pSobelResult[idx] < min) ? pSobelResult[idx] : min;
			max = (pSobelResult[idx] > max) ? pSobelResult[idx] : max;
		}
	}

	for (int i = 1; i < Size-1; i++) {
		for (int j = 1; j < Size; j++) {
			int idx = i * Size + j;
			float t = (pSobelResult[idx] - min) / (max - min);
			pOutImage[idx] = (unsigned char)(255 * t);
		}
	}

	delete[] pSobelResult;
}
