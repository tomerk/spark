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
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.bandit.WeightedStats

/**
 * Thompson Sampling with Gaussian priors (using sample variance).
 *
 * Modified from gaussian priorsthat have a variance of 1, from:
 * https://bandits.wikischolars.columbia.edu/file/view/Lecture+4.pdf
 *
 * @param numArms
 */
private[spark] class GaussianThompsonSamplingPolicy(
                                                     numArms: Int,
                                                     varianceMultiplier: Double
                                                   ) extends BanditPolicy(numArms) {
  @transient private lazy val tDists = (1 until 20).map(x => new TDistribution(x)).toArray
  override protected def estimateRewards(playsToMake: Int,
                                         totalRewards: Array[WeightedStats]): Seq[Double] = {
    (0 until numArms).map { arm =>
      val numPlays = totalRewards(arm).totalWeights
      if (numPlays >= 2) {
        val runningVariance = totalRewards(arm).variance

        // Breeze expects sigma as the input to the gaussian, not the variance.
        new Gaussian(
          totalRewards(arm).mean,
          math.sqrt(runningVariance / numPlays) * varianceMultiplier
        ).draw()
      } else if (numPlays >= 2) {
        val runningVariance = totalRewards(arm).variance

        (tDists(numPlays.toInt - 1).sample()) *
          math.sqrt(runningVariance / numPlays) * varianceMultiplier +
          totalRewards(arm).mean
      } else {
        Double.PositiveInfinity
      }
    }
  }
}
