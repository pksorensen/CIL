#include "mex.h"
#include "io64.h"
#include "cil/algorithms/algorithms.h"


void timestwo(double y[], double x[])
{
  y[0] = 2.0*x[0];
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  
    size_t mrows, ncols;
    char buf[100];
    int n;
	 

    
     
    #define SHAPES prhs[0]    
	#define ALIGNED_SHAPES plhs[0]
    #define MEAN_SHAPE plhs[1] 
  
    if (nrhs < 1)
    {
        mexErrMsgIdAndTxt( "MATLAB:gpa:inputNotDefined",
            "nargin should be > 0");
    }
         
    mrows = mxGetM(SHAPES);
    ncols = mxGetN(SHAPES);
    
    if (!(mxIsSingle(SHAPES) || mxIsDouble(SHAPES)) || mrows<2 || ncols <4|| ncols/2*2!=ncols){
         mexErrMsgIdAndTxt( "MATLAB:gpa:inputNotFloat",
            "First input must be a real matrix with more than one row and even columns");
    }
    /* The input must be a noncomplex scalar double.*/
    
	

	ALIGNED_SHAPES  = mxCreateDoubleMatrix(mrows,ncols,mxREAL);
    MEAN_SHAPE      = mxCreateDoubleMatrix(ncols,1,mxREAL);
	if(!ALIGNED_SHAPES)
		mexPrintf("MEMORY ERROR\n");

	double * test = mxGetPr(SHAPES);
	//mexPrintf("%f %f %f %f\n", test[0], test[1], test[2], test[3]);
	Eigen::Map<Eigen::MatrixXd> Shapes(mxGetPr(SHAPES), mrows,ncols );
	Eigen::Map<Eigen::MatrixXd> AShapes(mxGetPr(ALIGNED_SHAPES), mrows,ncols);
    Eigen::Map<Eigen::MatrixXd> MeanShape(mxGetPr(MEAN_SHAPE),ncols,1);
	//mexPrintf("%f %f %f %f\n", Shapes(0,0), Shapes(0,1), Shapes(0,2),Shapes(0,3));

    Eigen::MatrixXcd X(mrows, ncols/2);
	X.real() = Shapes.leftCols(ncols/2);
	X.imag() = Shapes.rightCols(ncols/2);
	X.array().colwise() -= X.rowwise().mean().array();
    

	Eigen::MatrixXcd Mean;
	Eigen::MatrixXcd C;
    cil::alg::gpa(X,C,Mean);
    mexPrintf("%f %f %f %f",Mean(0),Mean(1),Mean(2),Mean(3));
    
	AShapes.leftCols(ncols/2)       = C.real();
	AShapes.rightCols(ncols/2)      = C.imag();
    MeanShape.topRows(ncols/2)      = Mean.real();
    MeanShape.bottomRows(ncols/2)   = Mean.imag();

	mexPrintf("%d , %d\n", X.rows(), X.cols());
    mexPrintf("%d , %d\n", C.rows(), C.cols());
    mexPrintf("%d , %d\n", Mean.rows(), Mean.cols());
    
 //   mxGetString(SHAPES, buf, ncols+1);
 //   mexPrintf(buf);
 //   mexPrintf("\n");
      
          
          
//    /* Check for proper number of arguments. */
//   if(nrhs!=1) {
//    
//   } else if(nlhs>1) {
//     mexErrMsgIdAndTxt( "MATLAB:timestwo:maxlhs",
//             "Too many output arguments.");
//   }
//           
//     
//   double *x,*y;
// 
//   
//  
//   
// 
//   
//   //B OUT = mxCreateDoubleMatrix(M, N, mxREAL); /* Create the output matrix */
//   //B = mxGetPr(B OUT); /* Get the pointer to the data of B */
//   if( !mxIsChar(prhs[0]) )
//         mexErrMsgIdAndTxt( "MATLAB:timestwo:inputNotRealScalarDouble",
//             "Input must be a string");  
//   
//   if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
//       !(mrows==1 && ncols==1) ) {
//     mexErrMsgIdAndTxt( "MATLAB:timestwo:inputNotRealScalarDouble",
//             "Input must be a noncomplex scalar double.");
//   }
//   
//   /* Create matrix for the return argument. */
//   plhs[0] = mxCreateDoubleMatrix((mwSize)mrows, (mwSize)ncols, mxREAL);
//   
//   /* Assign pointers to each input and output. */
//   x = mxGetPr(prhs[0]);
//   y = mxGetPr(plhs[0]);
//   
//   /* Call the timestwo subroutine. */
//   timestwo(y,x);
}