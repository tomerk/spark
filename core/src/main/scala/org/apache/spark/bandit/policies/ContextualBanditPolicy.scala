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

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.bandit.WeightedStats
import org.apache.spark.util.StatCounter

sealed trait ContextualBanditPolicyParams
case class ContextualEpsilonGreedyPolicyParams(numFeatures: Int, epsilon: Double = 0.2)
  extends ContextualBanditPolicyParams
case class ContextualEpsilonFirstPolicyParams(numFeatures: Int, epsilon: Double)
  extends ContextualBanditPolicyParams
case class LinUCBPolicyParams(numFeatures: Int, alpha: Double = 2.36)
  extends ContextualBanditPolicyParams
case class LinThompsonSamplingPolicyParams(numFeatures: Int,
                                           v: Double = 5.0,
                                           useCholesky: Boolean = false)
  extends ContextualBanditPolicyParams

abstract class ContextualBanditPolicy(val numArms: Int, val numFeatures: Int) extends Serializable {
  @transient lazy private[spark] val stateLock = this
  val featuresAccumulator: Array[DenseMatrix[Double]] = {
    Array.fill(numArms)(DenseMatrix.eye(numFeatures))
  }
  val rewardAccumulator: Array[DenseVector[Double]] = {
    Array.fill(numArms)(DenseVector.zeros(numFeatures))
  }
  val rewardStatsAccumulator: Array[WeightedStats] = {
    Array.fill(numArms)(new WeightedStats())
  }

  def totalPlays(): Double = {
    stateLock.synchronized {
      var sum = 0.0
      var i = 0
      while (i < numArms) {
        sum += rewardStatsAccumulator(i).totalWeights
        i += 1
      }
      sum
    }
  }

  def chooseArm(features: DenseVector[Double]): Int = {
    val rewards = estimateRewards(features)
    val maxReward = rewards.max
    val bestArms = rewards.zipWithIndex.filter(_._1 == maxReward)
    bestArms(scala.util.Random.nextInt(bestArms.length))._2
  }

  private def estimateRewards(features: DenseVector[Double]): Seq[Double] = {
    (0 until numArms).map { arm =>
      val (armFeaturesAcc, armRewardsAcc, rewardStatsAcc) = stateLock.synchronized {
        (featuresAccumulator(arm), rewardAccumulator(arm), rewardStatsAccumulator(arm))
      }
      estimateRewards(features, armFeaturesAcc, armRewardsAcc, rewardStatsAcc)
    }
  }

  protected def estimateRewards(features: DenseVector[Double],
                                armFeaturesAcc: DenseMatrix[Double],
                                armRewardsAcc: DenseVector[Double],
                                rewardStatsAcc: WeightedStats): Double

  def provideFeedback(arm: Int,
                      features: DenseVector[Double],
                      rewardStats: WeightedStats): Unit = {
    // Note that this assumes all weights are 1
    val xxT = features * features.t
    val rx = rewardStats.count * rewardStats.mean * features

    provideFeedback(arm, xxT, rx, rewardStats)
  }

  def provideFeedback(arm: Int,
                      features: DenseMatrix[Double],
                      rewardVec: DenseVector[Double],
                      rewardStats: WeightedStats): Unit = {
    stateLock.synchronized {
      // Not an in-place update for the matrices, leaves them valid when used elsewhere!
      featuresAccumulator(arm) = featuresAccumulator(arm) + features
      rewardAccumulator(arm) = rewardAccumulator(arm) + rewardVec
      rewardStatsAccumulator(arm).merge(rewardStats)
    }
  }

  def setState(features: Array[DenseMatrix[Double]],
               rewards: Array[DenseVector[Double]],
               rewardStats: Array[WeightedStats]): Unit = {
    for (arm <- 0 until numArms) stateLock.synchronized {
      setState(arm, features(arm), rewards(arm), rewardStats(arm))
    }
  }

  def setState(arm: Int,
               features: DenseMatrix[Double],
               rewards: DenseVector[Double],
               rewardStats: WeightedStats): Unit = {
    stateLock.synchronized {
      featuresAccumulator(arm) = features
      rewardAccumulator(arm) = rewards
      rewardStatsAccumulator(arm) = rewardStats
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
