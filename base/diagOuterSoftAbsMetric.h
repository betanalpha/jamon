#ifndef _BETA_DIAGOUTERSOFTABSMETRIC_

#include <Eigen/Dense>

#include "dynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with a diagonal outer-product approximation to the SoftAbs Metric

class diagOuterSoftAbsMetric: public dynamMetric
{
    
    public:
        
        explicit diagOuterSoftAbsMetric(int dim);
        virtual ~diagOuterSoftAbsMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
        double tau();
        
        void bounceP(const VectorXd& normal);
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);
    
        /// Set the SoftAbs regularization coefficient
        void setSoftAbsAlpha(double a) { mSoftAbsAlpha = a; }
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void checkEvolution(const double epsilon = 1e-6);
        
        void displayState();
        
    protected:
        
        double mSoftAbsAlpha;   ///< Regularization coefficient
    
        MatrixXd mH;            ///< Gradient of the diagonal Hessian
        
        VectorXd mLambda;       ///< Inverse metric
        VectorXd mGradHelper;   ///< Auxiliary vector for intermeidate matrix calculations

        /// Compute the Hessian at the current position
        virtual void fComputeH() = 0;
        
        void fComputeMetric();
        void fPrepareSpatialGradients() { fComputeH(); gradV(); }
        
        VectorXd& dTaudp();
        VectorXd& dTaudq();
        VectorXd& dPhidq();
    
};

#define _BETA_DIAGOUTERSOFTABSMETRIC_
#endif