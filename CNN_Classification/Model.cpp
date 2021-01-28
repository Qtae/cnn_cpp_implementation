#include "Model.h"


Model::Model(Architecture *pArc, int Preds, int Classes)
{
	int k;
	int nfH, nfV;
	char msg[256];

	m_bIsMemoryAllocSucceeded = true;
	m_bIsModelTrained = 0;

	m_Preds = Preds;
	m_Classes = Classes;
	m_Layers = pArc->Layers;
	for (int LayerIndex = 0; LayerIndex < m_Layers; ++LayerIndex)
	{
		m_LayerType[LayerIndex] = pArc->LayerType[LayerIndex];
		m_Depth[LayerIndex] = pArc->Depth[LayerIndex];
		m_HalfWidthH[LayerIndex] = pArc->HalfWidthH[LayerIndex];
		m_HalfWidthV[LayerIndex] = pArc->HalfWidthV[LayerIndex];
		m_PaddingH[LayerIndex] = pArc->PaddingH[LayerIndex];
		m_PaddingV[LayerIndex] = pArc->PaddingV[LayerIndex];
		m_StrideH[LayerIndex] = pArc->StrideH[LayerIndex];
		m_StrideV[LayerIndex] = pArc->StrideV[LayerIndex];
		m_PoolWidthH[LayerIndex] = pArc->PoolWidthH[LayerIndex];
		m_PoolWidthV[LayerIndex] = pArc->PoolWidthV[LayerIndex];
		nfH = 2 * m_HalfWidthH[LayerIndex] + 1;
		nfV = 2 * m_HalfWidthV[LayerIndex] + 1;

		if (m_LayerType[LayerIndex] == TYPE_LOCAL || m_LayerType[LayerIndex] == TYPE_CONV2D)
		{
			m_PriorWeights[LayerIndex] = nfH * nfV;
			if (LayerIndex == 0)
			{
				m_Height[LayerIndex] = (m_ImageRows - nfV + 2 * m_PaddingV[LayerIndex]) / m_StrideV[LayerIndex] + 1;
				m_Width[LayerIndex] = (m_ImageCols - nfV + 2 * m_PaddingH[LayerIndex]) / m_StrideH[LayerIndex] + 1;
				m_PriorWeights[LayerIndex] *= m_ImageChannels;
			}
			else
			{
				m_Height[LayerIndex] = (m_Height[LayerIndex - 1] - nfV + 2 * m_PaddingV[LayerIndex]) / m_StrideV[LayerIndex] + 1;
				m_Width[LayerIndex] = (m_Width[LayerIndex - 1] - nfV + 2 * m_PaddingH[LayerIndex]) / m_StrideH[LayerIndex] + 1;
				m_PriorWeights[LayerIndex] *= m_Depth[LayerIndex - 1];
			}
			m_PriorWeights[LayerIndex] += 1;
		}

		else if (m_LayerType[LayerIndex] == TYPE_FC)
		{
			m_Height[LayerIndex] = 1;
			m_Width[LayerIndex] = 1;
			if (LayerIndex == 0)
				m_PriorWeights[LayerIndex] = m_Preds + 1;
			else
				m_PriorWeights[LayerIndex] = m_Neurons[LayerIndex - 1] + 1;
		}

		else if (m_LayerType[LayerIndex] == TYPE_AVGPOOL || m_LayerType[LayerIndex] == TYPE_MAXPOOL)
		{
			if (LayerIndex == 0)
			{
				m_Height[LayerIndex] = (m_ImageRows - m_PoolWidthV[LayerIndex]) / m_StrideV[LayerIndex] + 1;
				m_Width[LayerIndex] = (m_ImageCols - m_PoolWidthH[LayerIndex]) / m_StrideH[LayerIndex] + 1;
				m_Depth[LayerIndex] = m_ImageChannels;
			}
			else
			{
				m_Height[LayerIndex] = (m_Height[LayerIndex - 1] - m_PoolWidthV[LayerIndex]) / m_StrideV[LayerIndex] + 1;
				m_Width[LayerIndex] = (m_Width[LayerIndex - 1] - m_PoolWidthH[LayerIndex]) / m_StrideH[LayerIndex] + 1;
				m_Depth[LayerIndex] = m_Depth[LayerIndex - 1];
			}
			m_PriorWeights[LayerIndex] = 0;
		}

		else
		{
			assert(false);
		}

		m_Neurons[LayerIndex] = m_Height[LayerIndex] * m_Width[LayerIndex] * m_Depth[LayerIndex];
	}

	if (m_Layers == 0)
		m_PriorWeights[m_Layers] = m_Preds + 1;
	else
		m_PriorWeights[m_Layers] = m_Neurons[m_Layers - 1] + 1;

	m_pWeights = NULL;
	m_pCenterWts = NULL;
	m_pBestWts = NULL;
	m_pGradient = NULL;
	m_pThisDelta = NULL;
	m_pPriorDelta = NULL;
	m_pThrThisDelta = NULL;
	m_pThrPriorDelta = NULL;
	m_pPred = NULL;
	m_pConfScratch = NULL;
	m_pThresh = NULL;
	m_pConfusion = NULL;
	m_pThrOutput = NULL;
	m_pThrGradient[0] = NULL;
	for (int i = 0; i < MAX_LAYERS; ++i)
	{
		m_pActivity[i] = NULL;
		m_pThrActivity[0][i] = NULL;
		m_pPoolMaxID[i] = NULL;
		m_pThrPoolMaxID[0][i] = NULL;
	}

	//Check for invalid user parameters
	for (int LayerIndex = 0; LayerIndex < m_Layers; ++LayerIndex)
	{
		if (m_Height[LayerIndex] < 0 || m_Width[LayerIndex] <= 0 || m_Depth[LayerIndex] <= 0)
		{
			sprintf_s(msg, "User parameters in layer %d are invalid", LayerIndex + 1);
			//audit(msg);
			m_bIsMemoryAllocSucceeded = false;
			goto FINISH;
		}
	}

	m_MaxNeurons = m_Preds;
	if (m_Classes > m_MaxNeurons)
		m_MaxNeurons = m_Classes;

	m_HidWeights = 0;

	for (int LayerIndex = 0; LayerIndex < m_Layers; ++LayerIndex)
	{
		if (m_Neurons[LayerIndex] > m_MaxNeurons)
			m_MaxNeurons = m_Neurons[LayerIndex];

		if (m_LayerType[LayerIndex] == TYPE_FC || m_LayerType[LayerIndex] == TYPE_LOCAL)
			m_HidWeights += m_Neurons[LayerIndex] * m_PriorWeights[LayerIndex];
		else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
			m_HidWeights += m_Depth[LayerIndex] * m_PriorWeights[LayerIndex];
		else if (m_LayerType[LayerIndex] == TYPE_AVGPOOL || m_LayerType[LayerIndex] == TYPE_MAXPOOL)
			m_HidWeights += 0;
	}

	m_AllWeights = m_HidWeights + m_Classes * m_PriorWeights[m_Layers];

	//Allocate memory
	m_pWeights     = (double*)malloc(m_AllWeights * sizeof(double));
	m_pCenterWts   = (double*)malloc(m_AllWeights * sizeof(double));
	m_pBestWts     = (double*)malloc(m_AllWeights * sizeof(double));
	m_pGradient    = (double*)malloc(m_AllWeights * sizeof(double));
	m_pThisDelta   = (double*)malloc(m_MaxNeurons * sizeof(double));
	m_pPriorDelta  = (double*)malloc(m_MaxNeurons * sizeof(double));
	m_pPred        = (double*)malloc(n_cases * m_Classes * sizeof(double));
	m_pConfScratch = (double*)malloc(n_cases * sizeof(double));
	m_pThresh      = (double*)malloc(6 * m_Classes * sizeof(double));
	m_pConfusion   =    (int*)malloc(m_Classes * m_Classes * sizeof(int));

	if (m_pWeights == NULL || m_pCenterWts == NULL || m_pBestWts == NULL || m_pGradient == NULL || m_pThisDelta == NULL || m_pPriorDelta == NULL || m_pPred == NULL || m_pConfScratch == NULL || m_pThresh == NULL)
	{
		if (m_pWeights != NULL)
		{
			free(m_pWeights);
			m_pWeights = NULL;
		}
		if (m_pCenterWts != NULL)
		{
			free(m_pCenterWts);
			m_pCenterWts = NULL;
		}
		if (m_pBestWts != NULL)
		{
			free(m_pBestWts);
			m_pBestWts = NULL;
		}
		if (m_pGradient != NULL)
		{
			free(m_pGradient);
			m_pGradient = NULL;
		}
		if (m_pThisDelta != NULL)
		{
			free(m_pThisDelta);
			m_pThisDelta = NULL;
		}
		if (m_pPriorDelta != NULL)
		{
			free(m_pPriorDelta);
			m_pPriorDelta = NULL;
		}
		if (m_pPred != NULL)
		{
			free(m_pPred);
			m_pPred = NULL;
		}
		if (m_pConfScratch != NULL)
		{
			free(m_pConfScratch);
			m_pConfScratch = NULL;
		}
		if (m_pThresh != NULL)
		{
			free(m_pThresh);
			m_pThresh = NULL;
		}
		if (m_pConfusion != NULL)
		{
			free(m_pConfusion);
			m_pConfusion = NULL;
		}
		m_bIsMemoryAllocSucceeded = false;
		goto FINISH;
	}

	k = 0;
	
	for (int LayerIndex = 0; ; ++LayerIndex)
	{
		m_pLayerWeights[LayerIndex] = m_pWeights + k;
		m_pLayerGradient[LayerIndex] = m_pGradient + k;
		if (LayerIndex >= m_Layers) break;
		if (m_LayerType[LayerIndex] == TYPE_FC || m_LayerType[LayerIndex] == TYPE_LOCAL)
			k += m_Neurons[LayerIndex] * m_PriorWeights[LayerIndex];
		else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
			k += m_Depth[LayerIndex] * m_PriorWeights[LayerIndex];
		else if (m_LayerType[LayerIndex] == TYPE_MAXPOOL || m_LayerType[LayerIndex] == TYPE_AVGPOOL)
			k += 0;
	}

	for (int LayerIndex = 0; LayerIndex < m_Layers; ++LayerIndex)
	{
		m_pActivity[LayerIndex] = (double*)malloc(m_Neurons[LayerIndex] * sizeof(double));
		if (m_LayerType[LayerIndex] == TYPE_MAXPOOL)
			m_pPoolMaxID[LayerIndex] = (int*)malloc(m_Neurons[LayerIndex] * sizeof(int));
		if (m_pActivity[LayerIndex] == NULL || (m_LayerType[LayerIndex] == TYPE_MAXPOOL && m_pPoolMaxID[LayerIndex] == NULL))
		{
			for (int i = 0; i < LayerIndex; ++i)
			{
				if (m_pActivity[LayerIndex] != NULL)
				{
					free(m_pActivity[LayerIndex]);
					m_pActivity[LayerIndex] = NULL;
				}
				if (m_pPoolMaxID[LayerIndex] != NULL)
				{
					free(m_pPoolMaxID[LayerIndex]);
					m_pPoolMaxID[LayerIndex] = NULL;
				}
			}
			free(m_pWeights);
			free(m_pCenterWts);
			free(m_pBestWts);
			free(m_pGradient);
			free(m_pThisDelta);
			free(m_pPriorDelta);
			free(m_pPred);
			free(m_pConfScratch);
			free(m_pThresh);
			free(m_pConfusion);
			m_pWeights = NULL;
			m_pCenterWts = NULL;
			m_pBestWts = NULL;
			m_pGradient = NULL;
			m_pThisDelta = NULL;
			m_pPriorDelta = NULL;
			m_pPred = NULL;
			m_pConfScratch = NULL;
			m_pThresh = NULL;
			m_pConfusion = NULL;

			m_bIsMemoryAllocSucceeded = false;
			goto FINISH;
		}
	}

	//Threading

	m_pThrOutput = (double*)malloc(m_Classes * m_MaxThreads * sizeof(double));
	m_pThrThisDelta = (double*)malloc(m_MaxNeurons * m_MaxThreads * sizeof(double));
	m_pThrPriorDelta = (double*)malloc(m_MaxNeurons * m_MaxThreads * sizeof(double));
	m_pThrGradient[0] = (double*)malloc(m_AllWeights * m_MaxThreads * sizeof(double));

	if (m_pThrOutput == NULL || m_pThrThisDelta == NULL || m_pThrPriorDelta == NULL || m_pThrGradient == NULL)
	{
		if (m_pThrOutput != NULL)
		{
			free(m_pThrOutput);
			m_pThrOutput = NULL;
		}
		if (m_pThrThisDelta != NULL)
		{
			free(m_pThrThisDelta);
			m_pThrThisDelta = NULL;
		}
		if (m_pThrPriorDelta != NULL)
		{
			free(m_pThrPriorDelta);
			m_pThrPriorDelta = NULL;
		}
		if (m_pThrGradient[0] != NULL)
		{
			free(m_pThrGradient[0]);
			m_pThrGradient[0] = NULL;
		}
		m_bIsMemoryAllocSucceeded = false;
		goto FINISH;
	}

	for (int LayerIndex = 0; LayerIndex < m_Layers; ++LayerIndex)
	{
		m_pThrActivity[0][LayerIndex] = (double*)malloc(m_MaxThreads * m_Neurons[LayerIndex] * sizeof(double));
		if (m_LayerType[LayerIndex] == TYPE_MAXPOOL)
			m_pThrPoolMaxID[0][LayerIndex] = (int*)malloc(m_MaxThreads * m_Neurons[LayerIndex] * sizeof(int));
		if (m_pThrActivity[0][LayerIndex] == NULL || (m_LayerType[LayerIndex] == TYPE_MAXPOOL && m_pThrPoolMaxID[0][LayerIndex] == NULL))
		{
			for (int i = 0; i < LayerIndex; ++i)
			{
				//Memory Leakage?
				if (m_pThrActivity[0][i] != NULL)
				{
					free(m_pThrActivity[0][i]);
					m_pThrActivity[0][i] = NULL;
				}
				if (m_pThrPoolMaxID[0][i] != NULL)
				{
					free(m_pThrPoolMaxID[0][i]);
					m_pThrPoolMaxID[0][i] = NULL;
				}
				free(m_pThrOutput);
				free(m_pThrThisDelta);
				free(m_pThrPriorDelta);
				free(m_pThrGradient[0]);
				m_pThrOutput = NULL;
				m_pThrThisDelta = NULL;
				m_pThrPriorDelta = NULL;
				m_pThrGradient[0] = NULL;

				m_bIsMemoryAllocSucceeded = false;
				goto FINISH;
			}

			for (int ThreadIndex = 1; ThreadIndex < m_MaxThreads; ++ThreadIndex)
			{
				m_pThrActivity[ThreadIndex][LayerIndex] = m_pThrActivity[0][LayerIndex] + ThreadIndex * m_Neurons[LayerIndex];
				if (m_LayerType[LayerIndex] == TYPE_MAXPOOL)
					m_pThrPoolMaxID[ThreadIndex][LayerIndex] = m_pThrPoolMaxID[0][LayerIndex] + ThreadIndex * m_Neurons[LayerIndex];
			}
		}
	}

	for (int ThreadIndex = 0; ThreadIndex < m_MaxThreads; ++ThreadIndex)
	{
		k = 0;
		double *pGrad;
		pGrad = m_pThrGradient[0] + ThreadIndex * m_AllWeights;
		m_pThrGradient[ThreadIndex] = pGrad;
		for (int LayerIndex = 0; ; ++LayerIndex)
		{
			m_pThrLayerGradient[ThreadIndex][LayerIndex] = pGrad + k;

			if (LayerIndex >= m_Layers) break;
			if (m_LayerType[LayerIndex] == TYPE_FC || m_LayerType[LayerIndex] == TYPE_LOCAL)
				k += m_Neurons[LayerIndex] * m_PriorWeights[LayerIndex];
			else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
				k += m_Depth[LayerIndex] * m_PriorWeights[LayerIndex];
			else if (m_LayerType[LayerIndex] == TYPE_AVGPOOL || m_LayerType[LayerIndex] == TYPE_MAXPOOL)
				k += 0;
		}
	}

FINISH:
	m_ClassType = TrainParams.ClassType;
	m_Median = TrainParams.Median;
	m_Quantile33 = TrainParams.Quantile33;
	m_Quantile67 = TrainParams.Quantile67;

	if (!m_bIsMemoryAllocSucceeded)
	{
		//error message
	}
}

Model::~Model()
{
	if (m_pWeights != NULL) free(m_pWeights);
	if (m_pCenterWts != NULL) free(m_pCenterWts);
	if (m_pBestWts != NULL) free(m_pBestWts);
}

bool Model::IsMemoryAllocSucceeded()
{
	return m_bIsMemoryAllocSucceeded;
}

bool Model::IsModelTrained()
{
	return m_bIsModelTrained;
}

bool Model::Train()
{

}

void Model::CalculateForward(double* pInput)
{
	double sum;
	for (int LayerIndex = 0; LayerIndex < m_Layers; ++LayerIndex)
	{
		if (m_LayerType[LayerIndex] == TYPE_LOCAL)
			ActivateLocalLayer(LayerIndex, pInput);
		else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
			ActivateConv2DLayer(LayerIndex, pInput);
		else if (m_LayerType[LayerIndex] == TYPE_FC)
			ActivateFCLayer(LayerIndex, pInput, true);
		else if (m_LayerType[LayerIndex] == TYPE_AVGPOOL)
			ActivateAvgPoolingLayer(LayerIndex, pInput);
		else if (m_LayerType[LayerIndex] == TYPE_MAXPOOL)
			ActivateMaxPoolingLayer(LayerIndex, pInput);
		else
			assert(false);

		ActivateFCLayer(LayerIndex, pInput, false);

		sum = 1.e-60;
		for (int ClassIndex; ClassIndex < m_Classes; ++ClassIndex)
		{
			if (m_Output[ClassIndex] < 300.0)
				m_Output[ClassIndex] = exp(m_Output[ClassIndex]);
			else
				m_Output[ClassIndex] = exp(300.0);
			sum += m_Output[ClassIndex];
		}
		for (int ClassIndex; ClassIndex < m_Classes; ++ClassIndex)
		{
			m_Output[ClassIndex] /= sum;
		}
	}
}

double Model::CalculateLoss(int StartIndex, int StopIndex)
{
	double Error, TotalError;
	TotalError = 0.0;
	double *pData;
	int MaxIndex;
	double tmax;
	for (int DataIndex = StartIndex; DataIndex < StopIndex; ++DataIndex)
	{
		pData = m_Database + DataIndex * m_DBCols; // Database : data pointer, m_DBCols : Number of DB.
		CalculateForward(pData);
		/*
		if (DataIndex % 10 && (escape_key_pressed || user_pressed_escape()))
			return -1.0; //!!!User Escape
		*/
		Error = 0.0;

		MaxIndex = 0;
		tmax = -1.e30;
		for (int Index = 0; Index < m_Classes; ++Index)
		{
			m_pPred[DataIndex * m_Classes + Index] = m_Output[Index];
			if (pData[m_Preds + Index] > tmax)//!!! Is "pData" right? i think this should be m_pOutput..
			{
				MaxIndex = Index;
				tmax = pData[m_Preds + Index];
			}
		}
		Error = -log(m_Output[MaxIndex] + 1.e-30); //!!!Error Function
		TotalError += Error;
	}

	//Weight Penalty :: Regularization Methods!!!
	int Priors;
	double *pCurrWeights;
	double Weight;
	//double wpen = TrainParams.wpen / m_AllWeights;
	double HalfLambda = TrainParams.HalfLambda;//!!!
	double Penalty = 0.0;
	for (int LayerIndex = 0; LayerIndex <= m_Layers; ++LayerIndex)
	{
		pCurrWeights = m_pLayerWeights[LayerIndex];
		Priors = m_PriorWeights[LayerIndex];

		if (LayerIndex == m_Layers)
		{
			for (int NeuronIndex = 0; NeuronIndex < m_Classes; ++NeuronIndex)
			{
				for (int VarIndex = 0; VarIndex < Priors - 1; ++VarIndex) //not include bias
				{
					Weight = pCurrWeights[NeuronIndex * Priors + VarIndex];
					Penalty += Weight * Weight;
				}
			}
		}

		else if (m_LayerType[LayerIndex] == TYPE_FC)
		{
			for (int NeuronIndex = 0; NeuronIndex < m_Neurons[LayerIndex]; ++NeuronIndex)
			{
				for (int VarIndex = 0; VarIndex < Priors - 1; ++VarIndex)
				{
					Weight = pCurrWeights[NeuronIndex * Priors + VarIndex];
					Penalty += Weight * Weight;
				}
			}
		}

		else if (m_LayerType[LayerIndex] == TYPE_LOCAL)
		{
			for (int NeuronIndex = 0; NeuronIndex < m_Neurons[LayerIndex]; ++NeuronIndex)
			{
				for (int VarIndex = 0; VarIndex < Priors - 1; ++VarIndex)
				{
					Weight = pCurrWeights[NeuronIndex * Priors + VarIndex];
					Penalty += Weight * Weight;
				}
			}
		}

		else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
		{
			for (int NeuronIndex = 0; NeuronIndex < m_Neurons[LayerIndex]; ++NeuronIndex)
			{
				for (int VarIndex = 0; VarIndex < Priors - 1; ++VarIndex)
				{
					Weight = pCurrWeights[NeuronIndex * Priors + VarIndex];
					Penalty += Weight * Weight;
				}
			}
		}
	}

	Penalty *= HalfLambda;
	return TotalError / ((StopIndex - StartIndex) * m_Classes) + Penalty;
}

double Model::CalculateGradients(int StartIndex, int StopIndex)
{
	for (int i = 0; i < m_AllWeights; ++i)
		m_pGradient[i] = 0.0;

	double Error, Delta;
	Error = 0.0;
	double *pData;
	int MaxIndex;
	double tmax;
	int PrevNeurons, NextNeurons;
	double *PrevAct, *pGradient;
	for (int DataIndex = StartIndex; DataIndex < StopIndex; ++DataIndex)
	{
		pData = m_Database + DataIndex * m_DBCols;
		CalculateForward(pData);
		/*
		if (DataIndex % 10 && (escape_key_pressed || user_pressed_escape()))
			return -1.0; //!!!User Escape
		*/
		MaxIndex = 0;
		tmax = -1.e30;
		for (int Index = 0; Index < m_Classes; ++Index)
		{
			if (pData[m_Preds + Index] > tmax)//!!! Is "pData" right? i think this should be m_pOutput..
			{
				MaxIndex = Index;
				tmax = pData[m_Preds + Index];
			}
			m_pThisDelta[Index] = pData[m_Preds + Index] - m_Output[Index];
		}
		Error -= log(m_Output[MaxIndex] + 1.e-30);

		//Output Gradient
		if (m_Layers == 0)
		{
			PrevNeurons = m_Preds;
			PrevAct = pData;
		}
		else
		{
			PrevNeurons = m_Neurons[m_Layers - 1];
			PrevAct = m_pActivity[m_Layers - 1];
		}
		assert(PrevNeurons + 1 == m_PriorWeights[m_Layers]);
		pGradient = m_pLayerGradient[m_Layers];
		for (int i = 0; i < m_Classes; ++i)
		{
			Delta = m_pThisDelta[i];
			for (int j = 0; j < PrevNeurons; ++j)
				*pGradient++ += Delta * PrevAct[j];
			*pGradient++ += Delta;
		}
		NextNeurons = m_Classes;

		//Hidden Gradients
		for (int LayerIndex = m_Layers - 1; LayerIndex >= 0; --LayerIndex)
		{
			if (m_LayerType[LayerIndex] == TYPE_FC)
				CalculateGradientFC();
			else if (m_LayerType[LayerIndex] == TYPE_LOCAL)
				CalculateGradientLocal();
			else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
				CalculateGradientConv2D();
			else if (m_LayerType[LayerIndex] == TYPE_AVGPOOL || m_LayerType[LayerIndex] == TYPE_MAXPOOL)
				CalculateGradientPool();
			else
				assert(false);

			for (int i = 0; i < m_Neurons[LayerIndex]; ++i)
				m_pThisDelta[i] = m_pPriorDelta[i];
		}
	}

	for (int i = 0; i < m_AllWeights; ++i)
		m_pGradient[i] /= (StopIndex - StartIndex) * m_Classes;

	//Regularization (Weight Penalty)
	int Priors;
	double *pCurrWeights;
	double *pCurrGrad;
	double Weight;
	//double wpen = TrainParams.wpen / m_AllWeights;
	double HalfLambda = TrainParams.HalfLambda;//!!!
	double Penalty = 0.0;
	for (int LayerIndex = 0; LayerIndex <= m_Layers; ++LayerIndex)
	{
		pCurrWeights = m_pLayerWeights[LayerIndex];
		pCurrGrad = m_pLayerGradient[LayerIndex];
		Priors = m_PriorWeights[LayerIndex];

		if (LayerIndex == m_Layers)
		{

		}
		else if (m_LayerType[LayerIndex] == TYPE_FC)
		{

		}
		else if (m_LayerType[LayerIndex] == TYPE_LOCAL)
		{

		}
		else if (m_LayerType[LayerIndex] == TYPE_CONV2D)
		{

		}
	}
	Penalty *= HalfLambda;
	return Error / ((StopIndex - StartIndex) * m_Classes) + Penalty;
}

void Model::ActivateLocalLayer(int LayerIndex, double * pInput)
{
	assert(LayerIndex != m_Layers);

	int Rows, Cols, Slices;
	double *pIn, *pOut, *pCurrWeights;
	if (LayerIndex == 0)
	{
		Rows = m_ImageRows;
		Cols = m_ImageCols;
		Slices = m_ImageChannels;
		pIn = pInput;
	}
	else
	{
		Rows = m_Height[LayerIndex - 1];
		Cols = m_Width[LayerIndex - 1];
		Slices = m_Depth[LayerIndex - 1];
		pIn = m_pActivity[LayerIndex - 1];
	}
	pCurrWeights = m_pLayerWeights[LayerIndex];
	pOut = m_pActivity[LayerIndex];

	int k = 0;
	int StartRow, StopRow, StartCol, StopCol;
	double sum, x;
	for (int DepthIndex = 0; DepthIndex < m_Depth[LayerIndex]; ++DepthIndex)
	{
		for (int HeightIndex = 0; HeightIndex < m_Height[LayerIndex]; ++HeightIndex)
		{
			for (int WidthIndex = 0; WidthIndex < m_Width[LayerIndex]; ++WidthIndex)
			{
				sum = 0.0;
				StartRow = m_StrideV[LayerIndex] * HeightIndex - m_PaddingV[LayerIndex];
				StopRow = StartRow + 2 * m_HalfWidthV[LayerIndex];
				StartCol = m_StrideH[LayerIndex] * WidthIndex - m_PaddingH[LayerIndex];
				StopCol = StartCol + 2 * m_HalfWidthH[LayerIndex];

				for (int SliceIndex = 0; SliceIndex < Slices; ++SliceIndex)
				{
					for (int RowIndex = StartRow; RowIndex < StopRow; ++RowIndex)
					{
						for (int ColIndex = StartCol; ColIndex < StopCol; ++ColIndex)
						{
							if (RowIndex >= 0 && RowIndex < Rows && ColIndex >= 0 && ColIndex < Cols)
								x = pIn[(SliceIndex * Rows + RowIndex) * Cols + ColIndex];
							else
								x = 0.0;
							sum += x * *pCurrWeights++;
						}
					}
				}
				sum += *pCurrWeights++;
				//!!! activation function
				sum = exp(2.0 * sum);
				sum = (sum - 1.0) / (sum + 1.0);
				pOut[k++] = sum;
			}
		}
	}

	assert(k == m_Neurons[LayerIndex]);
	assert(m_pLayerWeights[LayerIndex] + m_Neurons[LayerIndex] * m_PriorWeights[LayerIndex] == pCurrWeights);
}

void Model::ActivateConv2DLayer(int LayerIndex, double * pInput)
{
	assert(LayerIndex != m_Layers);
	int Rows, Cols, Slices;
	double *pIn, *pOut, *pCurrWeights;
	if (LayerIndex == 0)
	{
		Rows = m_ImageRows;
		Cols = m_ImageCols;
		Slices = m_ImageChannels;
		pIn = pInput;
	}
	else
	{
		Rows = m_Height[LayerIndex - 1];
		Cols = m_Width[LayerIndex - 1];
		Slices = m_Depth[LayerIndex - 1];
		pIn = m_pActivity[LayerIndex - 1];
	}

	pOut = m_pActivity[LayerIndex];

	int k = 0;
	int StartRow, StopRow, StartCol, StopCol;
	double sum, x;
	pCurrWeights = NULL;
	for (int DepthIndex = 0; DepthIndex < m_Depth[LayerIndex]; ++DepthIndex)
	{
		for (int HeightIndex = 0; HeightIndex < m_Height[LayerIndex]; ++HeightIndex)
		{
			for (int WidthIndex = 0; WidthIndex < m_Width[LayerIndex]; ++WidthIndex)
			{
				pCurrWeights = m_pLayerWeights[LayerIndex] + DepthIndex * m_PriorWeights[LayerIndex];
				sum = 0.0;

				StartRow = m_StrideV[LayerIndex] * HeightIndex - m_PaddingV[LayerIndex];
				StopRow = StartRow + 2 * m_HalfWidthV[LayerIndex];
				StartCol = m_StrideH[LayerIndex] * WidthIndex - m_PaddingH[LayerIndex];
				StopCol = StartCol + 2 * m_HalfWidthH[LayerIndex];

				for (int SliceIndex = 0; SliceIndex < Slices; ++SliceIndex)
				{
					for (int RowIndex = StartRow; RowIndex < StopRow; ++RowIndex)
					{
						for (int ColIndex = StartCol; ColIndex < StopCol; ++ColIndex)
						{
							if (RowIndex >= 0 && RowIndex < Rows && ColIndex >= 0 && ColIndex < Cols)
								x = pIn[(SliceIndex * Rows + RowIndex) * Cols + ColIndex];
							else
								x = 0.0;
							sum += x * *pCurrWeights++;
						}
					}
				}
				sum += *pCurrWeights++;
				//!!!activation function
				sum = exp( 2.0 * sum );
				sum = (sum - 1.0) / (sum + 1.0);
				pOut[k++] = sum;
			}
		}
	}
	
	assert(k == m_Neurons[LayerIndex]);
	assert(m_pLayerWeights[LayerIndex] + m_Depth[LayerIndex] * m_PriorWeights[LayerIndex] == pCurrWeights);//!!!what is it?
}

void Model::ActivateFCLayer(int LayerIndex, double * pInput, bool bNonLinear)
{
	int nIn, nOut;
	double *pCurrWeights = m_pLayerWeights[LayerIndex];
	double *pIn, *pOut;
	if (LayerIndex == 0)
	{
		nIn = m_Preds;
		pIn = pInput;
	}
	else
	{
		nIn = m_Neurons[LayerIndex - 1];
		pIn = m_pActivity[LayerIndex - 1];
	}

	assert((nIn + 1) == m_PriorWeights[LayerIndex]);

	if (LayerIndex == m_Layers)
	{
		nOut = m_Classes;
		pOut = m_Output;
	}
	else
	{
		nOut = m_Neurons[LayerIndex];
		pOut = m_pActivity[LayerIndex];
	}

	for (int OutputIndex; OutputIndex < nOut; ++OutputIndex)
	{
		double sum = 0.0;
		for (int InputIndex; InputIndex < nIn; ++InputIndex)
			sum += pIn[InputIndex] * *pCurrWeights++;
		sum += *pCurrWeights++; // Bias
		if (bNonLinear)//Change it to Activation Function!!!
		{
			sum = exp(2.0 * sum);
			sum = (sum - 1.0) / (sum - 1.0);
		}
		pOut[OutputIndex] = sum;
	}
}

void Model::ActivateAvgPoolingLayer(int LayerIndex, double * pInput)
{
	assert(LayerIndex != m_Layers);

	int Rows, Cols, Slices;
	double *pIn, *pOut, *pCurrWeights;
	if (LayerIndex == 0)
	{
		Rows = m_ImageRows;
		Cols = m_ImageCols;
		Slices = m_ImageChannels;
		pIn = pInput;
	}
	else
	{
		Rows = m_Height[LayerIndex - 1];
		Cols = m_Width[LayerIndex - 1];
		Slices = m_Depth[LayerIndex - 1];
		pIn = m_pActivity[LayerIndex - 1];
	}

	pOut = m_pActivity[LayerIndex];

	int k = 0;
	int StartRow, StopRow, StartCol, StopCol;
	double value, x;
	for (int DepthIndex = 0; DepthIndex < m_Depth[LayerIndex]; ++DepthIndex)
	{
		for (int HeightIndex = 0; HeightIndex < m_Height[LayerIndex]; ++HeightIndex)
		{
			for (int WidthIndex = 0; WidthIndex < m_Width[LayerIndex]; ++WidthIndex)
			{
				StartRow = m_StrideV[LayerIndex] * HeightIndex;
				StopRow = StartRow + m_PoolWidthV[LayerIndex] - 1;
				StartCol = m_StrideH[LayerIndex] * WidthIndex;
				StopCol = StartCol + m_PoolWidthH[LayerIndex] - 1;

				assert(StopRow < Rows);
				assert(StopCol < Cols);

				value = 0.0;
				for (int RowIndex = StartRow; RowIndex <= StopRow; ++RowIndex)
				{
					for (int ColIndex = StartCol; ColIndex <= StopCol; ++ ColIndex)
						value += pIn[(DepthIndex * Rows + RowIndex) * Cols + ColIndex];
				}
				value /= m_PoolWidthV[LayerIndex] * m_PoolWidthH[LayerIndex];

				pOut[k++] = value;
			}
		}
	}
	assert(k == m_Neurons[LayerIndex]);
}

void Model::ActivateMaxPoolingLayer(int LayerIndex, double * pInput)
{
	assert(LayerIndex != m_Layers);

	int Rows, Cols, Slices;
	double *pIn, *pOut, *pCurrWeights;
	if (LayerIndex == 0)
	{
		Rows = m_ImageRows;
		Cols = m_ImageCols;
		Slices = m_ImageChannels;
		pIn = pInput;
	}
	else
	{
		Rows = m_Height[LayerIndex - 1];
		Cols = m_Width[LayerIndex - 1];
		Slices = m_Depth[LayerIndex - 1];
		pIn = m_pActivity[LayerIndex - 1];
	}

	pOut = m_pActivity[LayerIndex];

	int k = 0;
	int StartRow, StopRow, StartCol, StopCol;
	double value, x;
	for (int DepthIndex = 0; DepthIndex < m_Depth[LayerIndex]; ++DepthIndex)
	{
		for (int HeightIndex = 0; HeightIndex < m_Height[LayerIndex]; ++HeightIndex)
		{
			for (int WidthIndex = 0; WidthIndex < m_Width[LayerIndex]; ++WidthIndex)
			{
				StartRow = m_StrideV[LayerIndex] * HeightIndex;
				StopRow = StartRow + m_PoolWidthV[LayerIndex] - 1;
				StartCol = m_StrideH[LayerIndex] * WidthIndex;
				StopCol = StartCol + m_PoolWidthH[LayerIndex] - 1;

				assert(StopRow < Rows);
				assert(StopCol < Cols);

				value = -1.e60;
				for (int RowIndex = StartRow; RowIndex <= StopRow; ++RowIndex)
				{
					for (int ColIndex = StartCol; ColIndex <= StopCol; ++ColIndex)
					{
						x = pIn[(DepthIndex * Rows + RowIndex) * Cols + ColIndex];
						if (x > value)
						{
							value = x;
							m_pPoolMaxID[LayerIndex][k] = RowIndex * Cols + ColIndex;
						}
					}
				}
				pOut[k++] = value;
			}
		}
	}
	assert(k == m_Neurons[LayerIndex]);
}