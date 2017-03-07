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

import breeze.stats.distributions.Gaussian

/**
 * Thompson Sampling with Gaussian priors that have a variance of 1,
 * from:
 * https://bandits.wikischolars.columbia.edu/file/view/Lecture+4.pdf
 *
 * Fixme: This prior is ridiculous. We should be observing the running
 * variance of the rewards then using that.
 * @param numArms
 */
private[spark] class GaussianThompsonSamplingPolicy(
                                                     numArms: Int,
                                                     varianceMultiplier: Double
                                                   ) extends BanditPolicy(numArms) {
  override protected def estimateRewards(playsToMake: Int,
                                         totalPlays: Array[Long],
                                         totalRewards: Array[Double],
                                         totalRewardsSquared: Array[Double]): Seq[Double] = {
    (0 until numArms).map { arm =>
      val numPlays = totalPlays(arm)
      if (numPlays > 0) {
        // Variance Computed using naive algorithm from:
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        val varNumerator = totalRewardsSquared(arm) -
          (totalRewards(arm) * totalRewards(arm)) / totalPlays(arm)
        val runningVariance = varNumerator / (numPlays - 1)

        new Gaussian(
          totalRewards(arm) / (totalPlays(arm) + 1.0),
          runningVariance / totalPlays(arm)
        ).draw()
      } else {
        Double.PositiveInfinity
      }
    }
  }
}
