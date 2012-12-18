#ifndef _BETA_SOFTABSMETRIC_

#include <Eigen/Dense>

#include "dynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with a SoftAbs Metric

class softAbsMetric: public dynamMetric
{
    
    public:
        
        explicit softAbsMetric(int dim);
        virtual ~softAbsMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
        double tau();
    
        /// Return the inverse SoftAbs metric
        MatrixXd Lambda() 
        { 
            return mEigenDeco.eigenvectors() * mSoftAbsLambdaInv.asDiagonal() * mEigenDeco.eigenvectors().transpose(); 
        }
        
        void bounceP(const VectorXd& normal);
        
        /// Return the eigendecomposition of the metric
        SelfAdjointEigenSolver<MatrixXd>& eigenDeco() { return mEigenDeco; }
        
        /// Calculate the product of the metric with the current momentum
        VectorXd& lambdaDotP() { return dTaudp(); }
    
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);
        
        /// Set the SoftAbs regularization coefficient
        void setSoftAbsAlpha(double alpha) { mSoftAbsAlpha = alpha; }
    
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void checkEvolution(const double epsilon = 1e-6);
        
        void displayState();
        
    protected:
        
        double mSoftAbsAlpha;   ///< Regularization coefficient
    
        MatrixXd mH;            ///< Hessian
        MatrixXd mGradH;        ///< Gradient of the Hessian
        
        MatrixXd mPseudoJ;      ///< Discrete "pseudo-Jacobian" of the SoftAbs transform
    
        MatrixXd mAuxMatrixOne; ///< Auxiliary matrix for intermediate matrix calcs
        MatrixXd mAuxMatrixTwo; ///< Auxiliary matrix for intermediate matrix calcs
        MatrixXd mCacheMatrix;  ///< Auxiliary matrix for caching spatial gradient terms
    
        SelfAdjointEigenSolver<MatrixXd> mEigenDeco;  ///< Eigendecomposition of the metric

        VectorXd mSoftAbsLambda;    ///< Transformed eigenvalues
        VectorXd mSoftAbsLambdaInv; ///< Repciprocal of the transformed eigenvalues
        VectorXd mQp;               ///< Auxiliary vector for intermediate matrix calcs
        VectorXd mLambdaQp;         ///< Auxiliary vector for intermediate matrix calcs

        /// Compute the Hessian at the current position
        virtual void fComputeH() = 0;
        
        /// Compute the ith component of the gradient of the Hessian
        virtual void fComputeGradH(int i) = 0;
        
        void fComputeMetric();
        void fPrepareSpatialGradients();
        void fUpdateP();
    
        VectorXd& dTaudp();
        VectorXd& dTaudq();
        VectorXd& dPhidq();
    
};

#define _BETA_SOFTABSMETRIC_
#endif