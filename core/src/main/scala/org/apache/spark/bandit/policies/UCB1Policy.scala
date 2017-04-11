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

import org.apache.spark.bandit.WeightedStats

/**
 * UCB1, algorithm from:
 * https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
 *
 * Just like with GaussianThompsonSampling this has issues.
 * UCB1 was designed in a context where the reward is in [0,1]
 * and for us it clearly is not.
 *
 * Either multiply that factor by a hyperparameter times the range
 * in rewards of the individual arm's expectations, or use UCB tuned
 * (again slightly modified because 1/4 will be poorly scaled)
 *
 * @param numArms
 */
private[spark] class UCB1Policy(numArms: Int, boundsConst: Double) extends BanditPolicy(numArms) {
  override protected def estimateRewards(playsToMake: Int,
                                         totalRewards: Array[WeightedStats]): Seq[Double] = {
    val n = totalRewards.map(_.totalWeights).sum
    (0 until numArms).map { arm =>
      val numPlays = totalRewards(arm).totalWeights
      if (numPlays > 0) {
        totalRewards(arm).mean +
          boundsConst * math.sqrt(2.0*math.log(n)/numPlays)
      } else {
        Double.PositiveInfinity
      }
    }
  }
}
