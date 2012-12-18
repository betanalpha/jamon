#ifndef _BETA_DIAGSOFTABSMETRIC_

#include <Eigen/Dense>

#include "dynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with a diagonal approximation to the SoftAbs Metric

class diagSoftAbsMetric: public dynamMetric
{
    
    public:
        
        explicit diagSoftAbsMetric(int dim);
        virtual ~diagSoftAbsMetric() {};
        
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
        
        VectorXd mDiagH;        ///< Diagonal components of the Hessian
        MatrixXd mGradDiagH;    ///< Gradient of the diagonal Hessian
        
        VectorXd mLambda;       ///< Inverse metric
        VectorXd mGradHelper;   ///< Auxiliary vector for intermeidate matrix calculations
        
        /// Compute the diagonal components of the Hessian at the current position
        virtual void fComputeDiagH() = 0;
        
        /// Compute the gradient of the diagonal Hessian
        virtual void fComputeGradDiagH() = 0;
        
        void fComputeMetric();
        void fPrepareSpatialGradients() { fComputeGradDiagH(); gradV(); }

        VectorXd& dTaudp();
        VectorXd& dTaudq();
        VectorXd& dPhidq();
    
};

#define _BETA_DIAGSOFTABSMETRIC_
#endif