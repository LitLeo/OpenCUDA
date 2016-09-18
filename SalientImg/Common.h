#include "Image.h"

class cudaCommon {
public:
	int deviceCount;

	 __host__ ImageCuda*
    imageCut(
        Image *img
    );
};

void warmup();