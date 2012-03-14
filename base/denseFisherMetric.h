#ifndef _BETA_DENSEMETRIC_

#include <Eigen/Cholesky>

#include "denseDynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with a dense Fisher-Rao metric.

class denseFisherMetric: public denseDynamMetric
{
    
    public:
        
        explicit denseFisherMetric(int dim);
        virtual ~denseFisherMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();

        void evolveQ(double epsilon);

        void beginEvolveP(double epsilon);

        void finishEvolveP(double epsilon);  

        void bounceP(const VectorXd& normal);
    
        /// Return the Fisher-Rao metric
        MatrixXd& G() { return mG; }
    
        /// Return the Cholesky decompositio of the Fisher-Rao metric
        LLT<MatrixXd>& GL() { return mGL; }
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void checkEvolution(double epsilon = 1e-6);
        
        void displayState();
        
        void prepareEvolution() { fComputeCholeskyG(); }
        
    protected:
    
        MatrixXd mG;          ///< denseFisher-Rao metric
        MatrixXd mGradG;      ///< Component of the gradient of the denseFisher-Rao metric
        LLT<MatrixXd> mGL;    ///< Cholesky decomposition of the denseFisher-Rao metric
        VectorXd mC;          ///< Additional auxiliary vector for efficient matrix computations
        
        double mLogDetG;      ///< Log determinant of the denseFisher-Rao metric
    
        /// Compute the denseFisher-Rao metric at the current position
        virtual void fComputeG() = 0;

        /// Compute the ith component of the gradient of the denseFisher-Rao metric
        virtual void fComputeGradG(int i) = 0;
    
        void fComputeCholeskyG();

        void fHatA(double epsilon);
        void fHatF(double epsilon);
    
};
 
#define _BETA_DENSEFISHERMETRIC_
#endif