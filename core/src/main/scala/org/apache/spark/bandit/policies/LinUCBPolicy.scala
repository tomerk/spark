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

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import org.apache.spark.bandit.WeightedStats
import org.apache.spark.util.StatCounter

/**
 * LinUCB, from:
 * A Contextual-Bandit Approach to Personalized News Article Recommendation
 * (Li et al, 2010)
 * http://research.cs.rutgers.edu/~lihong/pub/Li10Contextual.pdf
 *
 *
 * TODO: Optimize code for the case where numFeatures is small (<= 5 to 10 or so)
 *
 * @param numArms
 * @param numFeatures
 * @param alpha hyperparameter that corresponds to how much uncertainty is valued.
 *              Set according to 1 + sqrt(ln(2/delta)/2)
 *              where 1-delta roughly corresponds to the probability that the
 *              estimated value has the true value within it's confidence bound
 */
private[spark] class LinUCBPolicy(numArms: Int, numFeatures: Int, alpha: Double)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armFeatureSumAcc: DenseVector[Double],
                                         armRewardsAcc: DenseVector[Double],
                                         armRewardsStats: WeightedStats): Double = {
    // TODO: Should be able to optimize code by only computing coefficientEstimate after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    if (armRewardsStats.totalWeights >= 2) {
      val coefficientEstimate = armFeaturesAcc \ armRewardsAcc
      coefficientEstimate.t * features + alpha * sqrt(
        features.t * armRewardsStats.variance * (armFeaturesAcc \ features)
      )
    } else {
      Double.PositiveInfinity
    }
  }
}
