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

/**
 * Like UCB-tuned, but without the min(1/4, std dev), just std dev.
 *
 * @param numArms
 */
private[spark] class UCBPsuedoTunedPolicy(
                                           numArms: Int,
                                           boundsConst: Double
                                         ) extends BanditPolicy(numArms) {
  override protected def estimateRewards(playsToMake: Int,
                                         totalPlays: Array[Long],
                                         totalRewards: Array[Double],
                                         totalRewardsSquared: Array[Double]): Seq[Double] = {
    val n = totalPlays.sum
    (0 until numArms).map { arm =>
      val numPlays = totalPlays(arm)
      if (numPlays > 2) {
        // Variance Computed using naive algorithm from:
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        val varNumerator = totalRewardsSquared(arm) -
          (totalRewards(arm) * totalRewards(arm)) / totalPlays(arm)
        val runningVariance = varNumerator / (numPlays - 1)

        (totalRewards(arm) / totalPlays(arm)) +
          boundsConst * math.sqrt(runningVariance * 2.0*math.log(n)/totalPlays(arm))
      } else {
        Double.PositiveInfinity
      }
    }
  }
}
