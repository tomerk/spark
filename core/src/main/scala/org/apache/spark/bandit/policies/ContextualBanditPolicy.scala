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

abstract class ContextualBanditPolicy(val numArms: Int, val numFeatures: Int) extends Serializable {
  @transient lazy private val stateLock = this
  val featuresAccumulator: Array[DenseMatrix[Double]] = {
    Array.fill(numArms)(DenseMatrix.eye(numFeatures))
  }
  val rewardAccumulator: Array[DenseVector[Double]] = {
    Array.fill(numArms)(DenseVector.zeros(numFeatures))
  }

  def chooseArm(features: DenseVector[Double]): Int = {
    val rewards = estimateRewards(features)
    val maxReward = rewards.max
    val bestArms = rewards.zipWithIndex.filter(_._1 == maxReward)
    bestArms(scala.util.Random.nextInt(bestArms.length))._2
  }

  private def estimateRewards(features: DenseVector[Double]): Seq[Double] = {
    (0 until numArms).map { arm =>
      val (armFeaturesAcc, armRewardsAcc) = stateLock.synchronized {
        (featuresAccumulator(arm), rewardAccumulator(arm))
      }
      estimateRewards(features, armFeaturesAcc, armRewardsAcc)
    }
  }

  protected def estimateRewards(features: DenseVector[Double],
                                armFeaturesAcc: DenseMatrix[Double],
                                armRewardsAcc: DenseVector[Double]): Double

  def provideFeedback(arm: Int, features: DenseVector[Double], reward: Double): Unit = {
    val xxT = features * features.t
    val rx = reward * features

    stateLock.synchronized {
      // Not an in-place update for the matrices, leaves them valid when used elsewhere!
      featuresAccumulator(arm) = featuresAccumulator(arm) + xxT
      rewardAccumulator(arm) = rewardAccumulator(arm) + rx
    }
  }

  /**
   * Warning: Does not lock the state of the other policy!
   * @param otherPolicy
   */
  def subtractState(otherPolicy: ContextualBanditPolicy): Unit = {
    for (arm <- 0 until numArms) stateLock.synchronized {
      // Not an in-place update for the matrices, leaves them valid when used elsewhere!
      featuresAccumulator(arm) = featuresAccumulator(arm) - otherPolicy.featuresAccumulator(arm)
      rewardAccumulator(arm) = rewardAccumulator(arm) - otherPolicy.rewardAccumulator(arm)
    }
  }

  /**
   * Warning: Does not lock the state of the other policy!
   * @param otherPolicy
   */
  def addState(otherPolicy: ContextualBanditPolicy): Unit = {
    for (arm <- 0 until numArms) stateLock.synchronized {
      // Not an in-place update for the matrices, leaves them valid when used elsewhere!
      featuresAccumulator(arm) = featuresAccumulator(arm) + otherPolicy.featuresAccumulator(arm)
      rewardAccumulator(arm) = rewardAccumulator(arm) + otherPolicy.rewardAccumulator(arm)
    }
  }
}
