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

import org.apache.spark.bandit.WeightedStats
import org.apache.spark.internal.Logging

sealed trait BanditPolicyParams
case class ConstantPolicyParams(arm: Int) extends BanditPolicyParams
case class EpsilonGreedyPolicyParams(epsilon: Double = 0.2) extends BanditPolicyParams
case class EpsilonFirstPolicyParams(epsilon: Double) extends BanditPolicyParams
case class UCB1PolicyParams(rewardRange: Double = 1.0) extends BanditPolicyParams
case class UCB1NormalPolicyParams(rewardRange: Double = 1.0) extends BanditPolicyParams
case class GaussianBayesUCBPolicyParams(rewardRange: Double = 1.0) extends BanditPolicyParams
case class GaussianThompsonSamplingPolicyParams(
                                                 varMultiplier: Double = 1.0
                                               ) extends BanditPolicyParams

abstract class BanditPolicy(val numArms: Int) extends Logging with Serializable {
  @transient lazy private[spark] val stateLock = this
  private val rewards: Array[WeightedStats] =
    Array.fill(numArms)(new WeightedStats())

  def totalPlays(): Double = {
    stateLock.synchronized {
      var sum = 0.0
      var i = 0
      while (i < numArms) {
        sum += rewards(i).totalWeights
        i += 1
      }
      sum
    }
  }

  def chooseArm(plays: Int): Int = {
    val rewardEstimates = estimateRewards(plays)
    val maxReward = rewardEstimates.max
    val bestArms = rewardEstimates.zipWithIndex.filter(_._1 == maxReward)

    bestArms(scala.util.Random.nextInt(bestArms.length))._2
  }

  private def estimateRewards(plays: Int): Seq[Double] = {
    val rewardsCopy = stateLock.synchronized {
      rewards.map(_.copy())
    }

    estimateRewards(plays, rewardsCopy)
  }

  protected def estimateRewards(playsToMake: Int,
                                observedRewards: Array[WeightedStats]): Seq[Double]

  def provideFeedback(arm: Int,
                      plays: Long,
                      reward: Double): Unit = stateLock.synchronized {
    rewards(arm).addMultiple(reward, 1.0, plays)
  }

  def setState(newRewards: Array[WeightedStats]): Unit = stateLock.synchronized {
    for (i <- 0 until numArms) {
      setState(i, newRewards(i))
    }
  }

  def setState(arm: Int, rewardObservations: WeightedStats): Unit =
    stateLock.synchronized {
      rewards(arm) = rewardObservations
  }

  /**
   * We make sure to capture the state lock before serialization
   * @param out
   */
  private def writeObject(out: ObjectOutputStream): Unit = stateLock.synchronized {
    out.defaultWriteObject()
  }
}
