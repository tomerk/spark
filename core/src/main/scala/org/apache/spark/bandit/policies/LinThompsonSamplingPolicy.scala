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

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, inv, sum}
import breeze.numerics.log
import breeze.stats.distributions._
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
 */
private[spark] class LinThompsonSamplingPolicy(numArms: Int, numFeatures: Int, v: Double)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armRewardsAcc: DenseVector[Double],
                                         armRewardsStats: StatCounter): Double = {
    // TODO: Should be able to optimize code by only computing coefficientMean after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    if (armRewardsStats.count > 2) {

      val coefficientMean = armFeaturesAcc \ armRewardsAcc
      val coefficientDist = InverseCovarianceMultivariateGaussian(
        coefficientMean,
        // We divide because this is the inverse covariance
         armFeaturesAcc / (v * armRewardsStats.sampleVariance))
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
 * @author dlwh, modified by tomerk11 to take inverse covariance as input
 */
case class InverseCovarianceMultivariateGaussian(
                                 mean: DenseVector[Double],
                                 inverseCovariance : DenseMatrix[Double]
                               )(implicit rand: RandBasis = Rand)
  extends ContinuousDistr[DenseVector[Double]] with
    Moments[DenseVector[Double], DenseMatrix[Double]] {
  def draw(): DenseVector[Double] = {
    val z: DenseVector[Double] = DenseVector.rand(mean.length, rand.gaussian(0, 1))
    root * z += mean
  }

  private val root: DenseMatrix[Double] = inv(cholesky(inverseCovariance)).t

  override def toString(): String = ScalaRunTime._toString(this)

  override def unnormalizedLogPdf(t: DenseVector[Double]): Double = {
    val centered = t - mean
    val slv = inverseCovariance * centered

    -(slv dot centered) / 2.0
  }

  override lazy val logNormalizer: Double = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = sum(log(diag(root)))
    mean.length/2 *  log(2 * math.Pi) + det
  }

  def variance: DenseMatrix[Double] = root * root.t
  def mode: DenseVector[Double] = mean
  lazy val entropy: Double = {
    mean.length * log1p(2 * math.Pi) + sum(log(diag(root)))
  }
}


