#ifndef _BETA_DIAGBKGDBSMETRIC_

#include "denseDynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with a BS metric induced from a diagonal background metric, 
/// \f$\Sigma_{ij} = \sigma_{i} \delta_{ij} + 
/// \alpha \partial_{i} V \partial_{j} V\f$

class diagBkgdBsMetric: public denseDynamMetric
{
    
    public:
        
        explicit diagBkgdBsMetric(int dim);
        virtual ~diagBkgdBsMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
        
        void evolveQ(double epsilon);
        
        void beginEvolveP(double epsilon);
        
        void finishEvolveP(double epsilon);  
        
        void bounceP(const VectorXd& normal);
    
        VectorXd& lambda() { return mBackLambda; } ///< Return inverse background metric
    
        MatrixXd& Lambda() { return mLambda; }     ///< Return inverse metric
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);
    
        void setAlpha(double a) { mAlpha = a; } ///< Set regularization coefficient
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void checkEvolution(double epsilon = 1e-6);
    
        void displayState();

        void prepareEvolution() { fComputeLambda(); fComputeGradLogDetLambda(); }
        
    protected:
        
        VectorXd mBackLambda;       ///< Inverse background metric
        MatrixXd mLambda;           ///< Inverse induced metric
        MatrixXd mHessianV;         ///< Hessian of the potential energy
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

#define _BETA_DIAGBKGDBSMETRIC_
#endif