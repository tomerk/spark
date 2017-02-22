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
 * UCB1, algorithm from:
 * https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
 *
 * @param numArms
 */
private[spark] class UCB1Policy(numArms: Int) extends BanditPolicy(numArms) {
  override protected def estimateRewards(playsToMake: Int,
                                         totalPlays: Array[Long],
                                         totalRewards: Array[Double]): Seq[Double] = {
    val n = totalPlays.sum
    (0 until numArms).map { arm =>
      if (totalPlays(arm) > 0) {
        (totalRewards(arm) / totalPlays(arm)) + math.sqrt(2.0*math.log(n)/totalPlays(arm))
      } else {
        Double.PositiveInfinity
      }
    }
  }
}
