#ifndef _BETA_DENSEBKGDBSMETRIC_

#include <Eigen/Cholesky>

#include "denseDynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with a BS metric induced from a dense background metric, 
/// \f$ \Sigma_{ij} = \sigma_{ij} + \alpha \partial_{i} V \partial_{j} V \f$

class denseBkgdBsMetric: public denseDynamMetric
{
    
    public:
        
        explicit denseBkgdBsMetric(int dim, MatrixXd& backLamda);
        virtual ~denseBkgdBsMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
        
        void evolveQ(double epsilon);
        
        void beginEvolveP(double epsilon);
        
        void finishEvolveP(double epsilon);  
        
        void bounceP(const VectorXd& normal);
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);
    
        void setBackLambda(MatrixXd& backLambda);
    
        void setAlpha(double a) { mAlpha = a; } ///< Set regularization coefficient
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void checkEvolution(double epsilon = 1e-6);
        
        void displayState();
        
        void prepareEvolution() { fComputeLambda(); fComputeGradLogDetLambda(); }
        
    protected:

        LLT<MatrixXd> mSigmaL; ///< Cholesky factorization of background metric
    
        MatrixXd mBackLambda;  ///< Inverse background metric
        MatrixXd mLambda;      ///< Inverse induced metric
        MatrixXd mHessianV;    ///< Hessian of the potential energy
        
        VectorXd mGradLogDetLambda; ///< Gradient of the log determinant of Lambda
        
        double mDetLambda;          ///< Determinant of the inverse induced metric
    
        double mAlpha;              ///< Regularization coefficient
        
        void fComputeDetLambda();
        void fComputeLambda();
        void fComputeGradLogDetLambda();
        
        virtual void fComputeHessianV() = 0; ///< Compute the Hessian matrix of the potential
        
        void fHatA(double epsilon);
        void fHatF(double epsilon);

};

#define _BETA_DENSEBKGDBSMETRIC_
#endif