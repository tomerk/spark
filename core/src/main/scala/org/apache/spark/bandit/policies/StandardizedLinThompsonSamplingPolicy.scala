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

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, eigSym, inv, max, min, sum}
import breeze.numerics.sqrt
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
class StandardizedLinThompsonSamplingPolicy(numArms: Int,
                                            numFeatures: Int,
                                            v: Double,
                                            useCholesky: Boolean,
                                            regParam: Double = 1e-2)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armFeatureSumAcc: DenseVector[Double],
                                         armRewardsAcc: DenseVector[Double],
                                         armRewardsStats: WeightedStats): Double = {
    // TODO: Should be able to optimize code by only computing coefficientMean after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    val n = armRewardsStats.totalWeights
    if (n >= 2) {

      val currArmFeaturesAcc = armFeaturesAcc - DenseMatrix.eye[Double](numFeatures)

      val featureMeans = armFeatureSumAcc / n
      val featureCov = (currArmFeaturesAcc / n) - (featureMeans * featureMeans.t)

      val featureStdDev = sqrt(diag(featureCov)).map(x =>
        if (x <= 1e-10 || x.isNaN) Double.PositiveInfinity else x)
      val featureCorr = n * featureCov :/ (featureStdDev * featureStdDev.t)

      min.inPlace(max.inPlace(featureCorr, -n), n)

      val rewardStdDev = math.sqrt(armRewardsStats.variance)
      val scaledRewards = (armRewardsAcc - (armFeatureSumAcc * armRewardsStats.mean)
        ) :/ (featureStdDev * rewardStdDev)


      val regVec = DenseVector.fill(numFeatures)((regParam + 1.0 * n)) - diag(featureCorr)


      //logError("About to print unregged matrix: \n" +featureCorr.toString(1000, 1000))

      featureCorr += diag(regVec)

      //logError("About to print regged matrix: \n" +featureCorr.toString(1000, 1000))

      val coefficientMean = featureCorr \ scaledRewards
      val coefficientDist = InverseCovarianceMultivariateGaussian(
        coefficientMean,
        // We divide because this is the inverse covariance
        featureCorr / (v * numFeatures * 1.0),
        useCholesky = useCholesky)
      val coefficientSample = coefficientDist.draw()

      val normFeatures = (features - featureMeans) :/ featureStdDev
      coefficientSample.t * normFeatures * rewardStdDev + armRewardsStats.mean
    } else {
      Double.PositiveInfinity
    }
  }
}

class BaseLinThompsonSamplingPolicy(numArms: Int,
                                    numFeatures: Int,
                                    v: Double,
                                    useCholesky: Boolean,
                                    usingBias: Boolean,
                                    regParam: Double = 1e-3)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armFeatureSumAcc: DenseVector[Double],
                                         armRewardsAcc: DenseVector[Double],
                                         armRewardsStats: WeightedStats): Double = {
    // TODO: Should be able to optimize code by only computing coefficientMean after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    if (armRewardsStats.totalWeights >= 2) {

      val currArmFeaturesAcc = armFeaturesAcc

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

object MLToleranceUtilsCopyCopy extends Serializable {

  @transient lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }
}

