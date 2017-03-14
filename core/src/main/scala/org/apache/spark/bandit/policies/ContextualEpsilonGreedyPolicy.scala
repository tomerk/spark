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
import org.apache.spark.util.StatCounter

/**
 * Note: Unplayed arms will default to zero reward. Fine in the case where our rewards
 * are strictly negative (e.g. reward = -1*runtime), but otherwise could cause issues.
 *
 * @param numArms
 * @param numFeatures
 * @param epsilon
 */
private[spark] class ContextualEpsilonGreedyPolicy(numArms: Int, numFeatures: Int, epsilon: Double)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override def chooseArm(features: DenseVector[Double]): Int = {
    if (scala.util.Random.nextFloat() >= epsilon) {
      super.chooseArm(features)
    } else {
      scala.util.Random.nextInt(numArms)
    }
  }

  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armRewardsAcc: DenseVector[Double],
                                         armRewardStatsAcc: StatCounter): Double = {
    // Note: Unplayed arms will default to zero reward. Fine
    val coefficientEstimate = armFeaturesAcc \ armRewardsAcc
    coefficientEstimate.t*features
  }
}
