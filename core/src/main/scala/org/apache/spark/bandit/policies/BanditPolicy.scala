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

import java.io.ObjectOutputStream

sealed trait BanditPolicyParams
case class EpsilonGreedyPolicyParams(epsilon: Double = 0.2) extends BanditPolicyParams
case class UCB1PolicyParams() extends BanditPolicyParams
case class GaussianThompsonSamplingPolicyParams() extends BanditPolicyParams

abstract class BanditPolicy(val numArms: Int) extends Serializable {
  @transient lazy private[spark] val stateLock = this
  private val totalPlays: Array[Long] = Array.fill(numArms)(0L)
  private val totalRewards: Array[Double] = Array.fill(numArms)(0.0)

  def chooseArm(plays: Int): Int = {
    val rewards = estimateRewards(plays)
    val maxReward = rewards.max
    val bestArms = rewards.zipWithIndex.filter(_._1 == maxReward)
    bestArms(scala.util.Random.nextInt(bestArms.length))._2
  }

  private def estimateRewards(plays: Int): Seq[Double] = {
    val (playsCopy, rewardsCopy) = stateLock.synchronized {
      (totalPlays.clone(), totalRewards.clone())
    }

    estimateRewards(plays, playsCopy, rewardsCopy)
  }

  protected def estimateRewards(playsToMake: Int,
                                totalPlays: Array[Long],
                                totalRewards: Array[Double]): Seq[Double]

  def provideFeedback(arm: Int, plays: Int, reward: Double): Unit = stateLock.synchronized {
    totalPlays(arm) += plays
    totalRewards(arm) += reward
  }

  def setState(plays: Array[Long], rewards: Array[Double]): Unit = stateLock.synchronized {
    for (i <- 0 until numArms) {
      totalPlays(i) = plays(i)
      totalRewards(i) = rewards(i)
    }
  }

  /**
   * We make sure to capture the state lock before serialization
   * @param out
   */
  private def writeObject(out: ObjectOutputStream): Unit = stateLock.synchronized {
    out.defaultWriteObject()
  }
}
