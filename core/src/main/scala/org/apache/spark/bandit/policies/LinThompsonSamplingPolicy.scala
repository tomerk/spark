/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.bandit.policies

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, eigSym, inv, max, sum}
import breeze.numerics.log
import breeze.stats.distributions._
import org.apache.spark.bandit.WeightedStats
import org.apache.spark.util.StatCounter

import scala.math.log1p
import scala.runtime.ScalaRunTime

/**
 * Linear Thompson Sampling,
 * Thompson Sampling for Contextual Bandits with Linear Payoffs
 * Agrawal et al. JMLR
 *
 * http://jmlr.csail.mit.edu/proceedings/papers/v28/agrawal13.pdf
 *
 * @param numArms
 * @param numFeatures
 * @param v Assuming rewards are R-sub-gaussian,
 *          set v = R*sqrt((24/epsilon)*numFeatures*ln(1/delta))
 *
 *          Practically speaking I'm not really sure what values to set,
 *          so I'll default to 5? Larger v means larger variance & more
 *          weight on sampling arms w/o the highest expectation
 * @param useCholesky Use cholesky as opposed to eigendecomposition. Much faster,
 *                    but risks erroring for some matrices.
 */
private[spark] class LinThompsonSamplingPolicy(numArms: Int,
                                               numFeatures: Int,
                                               v: Double,
                                               useCholesky: Boolean,
                                               usingBias: Boolean,
                                               regParam: Double = 1e-3)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armRewardsAcc: DenseVector[Double],
                                         armRewardsStats: WeightedStats): Double = {
    // TODO: Should be able to optimize code by only computing coefficientMean after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    if (armRewardsStats.totalWeights >= 2) {

      val currArmFeaturesAcc = armFeaturesAcc - DenseMatrix.eye[Double](numFeatures)
      val regValue = sum(diag(currArmFeaturesAcc))*regParam

      val regVec = DenseVector.fill(numFeatures)(regValue) / numFeatures.toDouble

      if (usingBias) {
        regVec(0) = 0.0
      }

      currArmFeaturesAcc += diag(regVec)


      val coefficientMean = currArmFeaturesAcc \ armRewardsAcc
      val coefficientDist = InverseCovarianceMultivariateGaussian(
        coefficientMean,
        // We divide because this is the inverse covariance
        currArmFeaturesAcc / (v * numFeatures * armRewardsStats.variance),
        useCholesky = useCholesky)
      val coefficientSample = coefficientDist.draw()

      coefficientSample.t * features
    } else {
      Double.PositiveInfinity
    }
  }
}

/**
 * Represents a Gaussian distribution over a single real variable.
 *
 * Combination of Breeze multivariate gaussian draw code & spark multivariate gaussian
 */
case class InverseCovarianceMultivariateGaussian(
                                 mean: DenseVector[Double],
                                 inverseCovariance : DenseMatrix[Double],
                                 useCholesky: Boolean
                               )(implicit rand: RandBasis = Rand) {
  def draw(): DenseVector[Double] = {
    val z: DenseVector[Double] = DenseVector.rand(mean.length, rand.gaussian(0, 1))
    root * z += mean
  }

  private val root: DenseMatrix[Double] = {
    if (useCholesky) {
      inv(cholesky(inverseCovariance)).t
    } else {
      val eigSym.EigSym(d, u) = eigSym(inverseCovariance) // sigma = u * diag(d) * u.t

      // For numerical stability, values are considered to be non-zero only if they exceed tol.
      // This prevents any inverted value from exceeding (eps * n * max(d))^-1
      val tol = MLToleranceUtilsCopy.EPSILON * max(d) * d.length

      try {
        // log(pseudo-determinant) is sum of the logs of all non-zero singular values
        //val logPseudoDetSigma = d.activeValuesIterator.filter(_ > tol).map(math.log).sum

        // calculate the root-pseudo-inverse of the diagonal matrix of singular values
        // by inverting the square root of all non-zero values
        val pinvS = diag(d.map(v => if (v > tol) math.sqrt(1.0 / v) else 0.0))

        pinvS * u.t
      } catch {
        case uex: UnsupportedOperationException =>
          throw new IllegalArgumentException("Covariance matrix has no non-zero singular values")
      }
    }
  }
}

private[spark] object MLToleranceUtilsCopy {

  lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }
}



