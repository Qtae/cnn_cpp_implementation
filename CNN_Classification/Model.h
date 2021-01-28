#pragma once
#include "pch.h"

#include <iostream>
#include <assert.h>

typedef struct Architecture
{
	int Layers;
	int LayerType[MAX_LAYERS];
	int Depth[MAX_LAYERS];
	int HalfWidthH[MAX_LAYERS];
	int HalfWidthV[MAX_LAYERS];
	int PaddingH[MAX_LAYERS];
	int PaddingV[MAX_LAYERS];
	int StrideH[MAX_LAYERS];
	int StrideV[MAX_LAYERS];
	int PoolWidthH[MAX_LAYERS];
	int PoolWidthV[MAX_LAYERS];
}Architecture;

typedef struct TrainParams
{

}TrainParams;

class Model
{
public:
	Model(Architecture *pArc, int Preds, int Classes);
	~Model();
	bool IsMemoryAllocSucceeded();
	bool IsModelTrained();
	bool Train();

	void CalculateForward(double *pInput);
	double CalculateLoss(int StartIndex, int StopIndex);
	double CalculateGradients(int StartIndex, int StopIndex);

	void ActivateLocalLayer(int LayerIndex, double * pInput);
	void ActivateConv2DLayer(int LayerIndex, double * pInput);
	void ActivateFCLayer(int LayerIndex, double * pInput, bool bNonLinear);
	void ActivateAvgPoolingLayer(int LayerIndex, double * pInput);
	void ActivateMaxPoolingLayer(int LayerIndex, double * pInput);

	void CalculateGradientFC();
	void CalculateGradientLocal();
	void CalculateGradientConv2D();
	void CalculateGradientPool();

private:
	bool m_bIsMemoryAllocSucceeded;
	bool m_bIsModelTrained;
	//
	int m_Preds;
	int m_Classes;
	int m_Layers;
	int m_LayerType[MAX_LAYERS];
	int m_Height[MAX_LAYERS];
	int m_Width[MAX_LAYERS];
	int m_Depth[MAX_LAYERS];
	int m_Neurons[MAX_LAYERS]; // m_nHeight * m_nWidth * m_nDepth
	int m_HalfWidthH[MAX_LAYERS];
	int m_HalfWidthV[MAX_LAYERS];
	int m_PaddingH[MAX_LAYERS];
	int m_PaddingV[MAX_LAYERS];
	int m_StrideH[MAX_LAYERS];
	int m_StrideV[MAX_LAYERS];
	int m_PoolWidthH[MAX_LAYERS];
	int m_PoolWidthV[MAX_LAYERS];
	int m_PriorWeights[MAX_LAYERS + 1];
	int m_HidWeights;
	int m_AllWeights;
	int m_MaxNeurons;

	int m_ImageRows;
	int m_ImageCols;
	int m_ImageChannels;

	double *m_pWeights;
	double *m_pLayerWeights[MAX_LAYERS + 1];
	double *m_pBestWts;
	double *m_pCenterWts;
	double *m_pGradient;
	double *m_pLayerGradient[MAX_LAYERS + 1];
	double *m_pActivity[MAX_LAYERS];
	double *m_pThisDelta;
	double *m_pPriorDelta;
	double m_Output[MAX_CLASSES];
	double *m_pConfScratch;
	int *m_pPoolMaxID[MAX_LAYERS];
	int *m_pConfusion;
	double *m_pPred;
	double *m_pThresh;
	//These are allocated and freed in Model::train() to provide multiple threads with private work areas
	double *m_pThrOutput;
	double *m_pThrThisDelta;
	double *m_pThrPriorDelta;
	double *m_pThrActivity[MAX_THREADS][MAX_LAYERS];
	int *m_pThrPoolMaxID[MAX_THREADS][MAX_LAYERS];
	double *m_pThrGradient[MAX_THREADS];
	double *m_pThrLayerGradient[MAX_THREADS][MAX_LAYERS + 1];
	//
	int m_ClassType;
	double m_Median;
	double m_Quantile33;
	double m_Quantile67;

	int m_ImageRows;
	int m_ImageCols;
	int m_ImageChannels;
};