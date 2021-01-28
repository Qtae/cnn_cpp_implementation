#ifndef PCH_H
#define PCH_H


#include "framework.h"

//Max values
#define MAX_LAYERS 100
#define MAX_CLASSES 100
#define MAX_THREADS 10

//Layer Type
#define TYPE_LOCAL		0
#define TYPE_CONV2D		1
#define TYPE_CONV3D		2
#define TYPE_FC			3
#define TYPE_DENSE		4
#define TYPE_MAXPOOL	5
#define TYPE_AVGPOOL	6
#define TYPE_FLATTEN	7

#endif //PCH_H
