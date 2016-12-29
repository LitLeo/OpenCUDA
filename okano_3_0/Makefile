ALGNAMES := Image Matrix Template CoordiSet TemplateFactory AffineTrans    \
            RotateTable SelectShape SortArray Histogram HistogramSpec      \
            Threshold BoundingRect Flip Binarize Morphology ConnectRegion  \
            LabelIslandSortArea LinearEnhancement FillUp ImageDiff         \
            ImageHide Moments GeometryProperties Normalization             \
            FreckleFilter LocalCluster DownSampleImage HoughLine           \
            BilateralFilter MultiThreshold BilinearInterpolation           \
            FeatureVecCalc FeatureVecArray ImageMatch ScanArray            \
            SalientRegionDetect RoiCopy RobustEdgeDetection                \
            PerspectiveTrans FanczosIpl OtsuBinarize LinearFilter          \
            WorkAndObjectMatch Thinning ImageScaling EdgeCheck             \
            SmoothVector SegmentedScan HoughCircle ConvexHull              \
            Segmentation SimpleRegionDetect Rectangle SmallestDirRect      \
            HoughRec GetContourSet ImageDrawer ImgConvert ImageOverlay     \
            TorusSegmentation IsolatedPoints InnerDigger ImageToText       \
            Tattoo ImageStretch Mosaic Complex Mandelbrot Binarization     \
            Compact FlutterBinarize Curve Zoom Julia Reduce                \
            ConnectRegionNew Multidecrease DiffPattern CurveFluctuPropers  \
            CurveFluctuation SuperSmooth ThresholdSegmentation             \
            DynamicArrays Graph LearningFilter FillCoor RegionGrow         \
            CurveTracing GaussianElimination PriFeatureCheckC              \
            HistogramDifference CurveTopology ICcircleRadii                \
            GaussianSmoothImage GaussianSmoothxy DouglasPeucker SalientImg \
            LocalHistogramEqualization OtsuForThree ClusterLocalGray       \
            ContourMatch CovarianceMatrix InscribedCircle EdgeDetection    \
            GeoPrimaryProperties OperationFunctor CombineImage             \
            ConsolidateAndIdentifyContours ImageFilter CurveConverter      \
            FillCurve SimpleBrightnessGradient MultiConnectRegion 
			
INCFILES := ErrorCode.h $(addsuffix .h, $(ALGNAMES))
OBJFILES := main.o $(addsuffix .o, $(ALGNAMES))

EXEFILE  := okanoexec

NVCCCMD  := nvcc
NVCCFLAG := -arch=sm_20
NVLDFLAG := -lnppi

world: $(EXEFILE)

$(EXEFILE): $(OBJFILES)
	$(NVCCCMD) $(OBJFILES) -o $(EXEFILE) $(NVLDFLAG)

$(OBJFILES): %.o:%.cu $(INCFILES)
	$(NVCCCMD) -c $(filter %.cu, $<) -o $@ $(NVCCFLAG)

clean:
	rm -rf $(OBJFILES) $(EXEFILE)
