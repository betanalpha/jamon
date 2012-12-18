#ifndef _BETA_BASEHAMILTONIAN_

#include <Eigen/Core>
#include <RandomLib/Random.hpp>

// Disable Eigen runtime checks
#define NDEBUG

using namespace Eigen;
using RandomLib::Random;

///
///  \mainpage Jamón: An Implementation of Constrained Hamiltonian Monte Carlo
///
///  A suite of classes facilitating generalized Hamiltonian 
///  Monte Carlo (both constrained and unconstrainted).
///  For details see http://arxiv.org/abs/1112.4118 
/// 
///  baseHamiltonian defines the interface for the
///  target distribution, in particular the kinetic
///  energy, the potential energy, the evolution
///  of the position and momentum, and any constraints.
///
///  chainBundle provides a container of baseHamiltonians,
///  including the functionality to perform HMC transitions,
///  diagnose convergence, and analyse samples.
///
///  Specifically, the code implements Hamiltonians defined
///  on Euclidean and Riemannian manifolds (i.e. with
///  kinetic energies given by quadratic forms):
///
///     - constMetric (Euclidean Metrics)
///         - diagConstMetric  (Diagonal)
///         - denseConstMetric (Dense)
///
///     - dynamMetric (Riemannian Metrics)
///
///         - denseFisherMetric (Dense Fisher-Rao)
///         - softAbsMetric (Dense SoftAbs)
///
///             - Approximations to the SoftAbs Metric
///                 - diagSoftAbsMetric (Diagonal)
///                 - outerSoftAbsMetric (Outer-Product)
///                 - diagOuterSoftAbsMetric (Diagonal Outer-Product)
///
///  Example implementations include a multivariate Gaussian
///  and Neal's "funnel" distribution.
///
///
///  Matrix operations are performed with the Eigen
///  template library: http://eigen.tuxfamily.org/
///
///  Random number generation is handled with the
///  randomlib library: http://randomlib.sourceforge.net/
///
///  Option plotting available with gnuplot:
///  http://www.gnuplot.info/
///
///  Copyright Michael Betancourt 2012
///  betanalpha@gmail.com
///
///  Permission is hereby granted, free of charge, to any person obtaining a 
///  copy of this software and associated documentation files (the "Software"), 
///  to deal in the Software without restriction, including without limitation 
///  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
///  and/or sell copies of the Software, and to permit persons to whom the 
///  Software is furnished to do so, subject to the following conditions:
///  
///  The above copyright notice and this permission notice shall be 
///  included in all copies or substantial portions of the Software.
///  
///  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
///  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
///  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
///  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
///  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
///  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
///  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
///

///  \example examples/main.cpp
/// Neal's "funnel" distribution and logistic regression.

///  \author Michael Betancourt
///
///  Abstract base class defining the interface
///  for a generic Hamiltonian Monte Carlo chain,
///  with the potential energy given by a 
///  target distribution and the kinetic energy 
///  left free for optimization.

class baseHamiltonian
{
  
    public:
    
        explicit baseHamiltonian(int dim);
        virtual ~baseHamiltonian() {};
    
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
    
        double dim() { return mDim; }              ///< Return the dimensionality of the target space

        virtual double T() = 0;                    ///< Return the kinetic energy
        virtual double V() = 0;                    ///< Return the potential energy
        virtual const VectorXd& gradV() = 0;       ///< Return the gradient of the potential energy
    
        double H() { return T() + V(); }           ///< Return the Hamiltonian
    
        VectorXd& q() { return mQ; }               ///< Return the current position
        double q(int i) { return mQ(i); }          ///< Return the ith component of the current position
        VectorXd& p() { return mP; }               ///< Return the current momentum
        double p(int i) { return mP(i); }          ///< Return the ith component of the current momentum
    
        VectorXd& acceptQ() { return mAcceptQ; }   ///< Return the currect accept window sample
        VectorXd& rejectQ() { return mRejectQ; }   ///< Return the currect reject window sample
    
        virtual void evolveQ(const double epsilon) = 0;  ///< Evolve the position through some time epsilon
    
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////

        /// Perform any necessary calculations before
        /// simulation Hamiltonian dynamics
        virtual void prepareEvolution() {};
    
        /// Evolve the momenta through an initial half step of time epsilon
        virtual void beginEvolveP(const double epsilon) = 0;
        
        /// Evolve the momenta through a final half step of time epsilon
        virtual void finishEvolveP(const double epsilon) = 0;  
        
        /// Evolve the momentum through a bounce off of a constraint surface
        /// \param normal Vector normal to constraint surface
        virtual void bounceP(const VectorXd& normal) = 0;
    
        /// Sample the momentum from a conditional distribution of
        /// \f$ \pi \left( \mathbf{p}, \mathbf{q} \right) \propto \exp \left( - H \right) \f$
        /// \param random External RandomLib generator
        virtual void sampleP(Random& random) = 0;
    
        /// Save current position and momentum
        void saveCurrentPoint() { mStoreQ = mQ, mStoreP = mP; }
    
        /// Restore saved position and momentum
        void restoreStoredPoint() { mQ = mStoreQ, mP = mStoreP; }
        
        /// Save current position as a sample from the reject window
        void saveAsRejectSample() { mRejectQ = mQ; }
    
        /// Save current position as a sample from the accept window
        void saveAsAcceptSample() { mAcceptQ = mQ; }
        
        /// Select between the reject and accept windows
        void sampleWindows(const bool accept) { accept ? mQ = mAcceptQ : mQ = mRejectQ; }
    
        /// Set moving average decay rate
        void setAverageDecay(const double alpha) { mMovingAlpha = alpha; }
    
        void updateMetroStats(const bool b, const double a);
    
        void clearHistory();
    
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
    
        bool isNaN();
    
        /// Is the chain within the support of the distribution?
        virtual bool supportViolated() { return false; }
        
        /// Return the normal to the support boundary
        virtual VectorXd& supportNormal() { return mN; }
    
        /// Comparing the evolution implementations with finite differences
        virtual void checkEvolution(const double epsilon = 1e-6);
    
        /// Metropolis accept rate over the history of the chains
        double acceptRate() { return mNumAccept / (mNumAccept + mNumReject); }
        
        /// Moving expectation of the Metropolis accept rate
        double movingAcceptRate() { return mAcceptRateBar; }
    
        /// Display current state
        virtual void displayState();
    
    protected:
    
        const int mDim;       ///< Dimension of target space
    
        VectorXd mQ;          ///< Position in target space
        VectorXd mStoreQ;     ///< Stored position
        VectorXd mRejectQ;    ///< Stored position in reject window
        VectorXd mAcceptQ;    ///< Stored position in accept window
    
        VectorXd mP;          ///< Momentum
        VectorXd mStoreP;     ///< Stored momentum
    
        VectorXd mN;          ///< Normal to any surface of constrained support
    
        VectorXd mGradV;      ///< Gradient of the potential energy
    
        // Metropolis parameters
        double mNumAccept;        ///< Total number of accepted samples
        double mNumReject;        ///< Total number of rejected samples
        
        double mAcceptRateBar;    ///< Moving average accept rate
        double mEffN;             ///< Moving average effective samples
        double mMovingAlpha;      ///< Moving average decay rate
    
};

#define _BETA_BASEHAMILTONIAN_
#endif