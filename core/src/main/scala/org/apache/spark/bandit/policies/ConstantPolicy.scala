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
 * Constant Policy that always selects the same arm
 *
 * @param numArms
 * @param arm The arm to always select
 */
private[spark] class ConstantPolicy(numArms: Int, arm: Int)
  extends BanditPolicy(numArms) {
  override def chooseArm(plays: Int): Int = {
    arm
  }

  override protected def estimateRewards(playsToMake: Int,
                                         totalRewards: Array[WeightedStats]): Seq[Double] = {
    (0 until numArms).map { curArm =>
      if (curArm != arm) {
        0
      } else {
        Double.PositiveInfinity
      }
    }
  }
}
