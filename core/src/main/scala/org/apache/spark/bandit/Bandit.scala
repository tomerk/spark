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

package org.apache.spark.bandit

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt

import scala.reflect.ClassTag

/**
 * The bandit class is used for dynamically tuned methods appearing in spark tasks.
 */
abstract class Bandit[A: ClassTag, B: ClassTag] {
  def apply(in: A): B

  /**
   * A vectorized bandit strategy. Given a sequence of input, choose a single arm
   * to apply to all of the input. The learning will treat this as the reward
   * averaged over all the input items, given this decision.
   *
   * @param in The vector of input
   */
  def vectorizedApply(in: Seq[A]): Seq[B]

  def saveTunedSettings(file: String): Unit
  def loadTunedSettings(file: String): Unit
}

/**
 * LinUCB, from:
 * A Contextual-Bandit Approach to Personalized News Article Recommendation
 * (Li et al, 2010)
 * http://research.cs.rutgers.edu/~lihong/pub/Li10Contextual.pdf
 *
 * FIXME: The first round should choose totally randomly! Or else will wayy overweight
 * the first thing.
 *
 * TODO: Optimize code for the case where numFeatures is small (<= 5 to 10 or so)
 *
 * @param numArms
 * @param numFeatures
 * @param alpha hyperparameter that corresponds to how much uncertainty is valued.
 *              Set according to 1 + sqrt(ln(2/delta)/2)
 *              where 1-delta roughly corresponds to the probability that the
 *              estimated value has the true value within it's confidence bound
 */
class LinUCBState(numArms: Int, numFeatures: Int, alpha: Double = 2.36) {
  val as: Array[DenseMatrix[Double]] = Array.fill(numArms)(DenseMatrix.eye(numFeatures))
  val bs: Array[DenseVector[Double]] = Array.fill(numArms)(DenseVector.zeros(numFeatures))

  def chooseArm(features: DenseVector[Double]): Int = {
    var arm = 0
    var bestArm = -1
    var bestArmEstimate = Double.NegativeInfinity
    while (arm < numArms) {
      val (a, b) = getState(arm)
      val coefficientEstimate = a \ b
      val rewardEstimate = coefficientEstimate.t*features + alpha*sqrt(features.t*(a \ features))

      if (rewardEstimate > bestArmEstimate) {
        bestArm = arm
        bestArmEstimate = rewardEstimate
      }
      arm += 1
    }

    bestArm
  }

  protected def getState(arm: Int): (DenseMatrix[Double], DenseVector[Double]) = this.synchronized {
    (as(arm), bs(arm))
  }

  protected def addToState(arm: Int, a: DenseMatrix[Double], b: DenseVector[Double]): Unit =
    this.synchronized {
    as(arm) = as(arm) + a
    bs(arm) = bs(arm) + b
  }

  def updateState(features: DenseVector[Double], arm: Int, reward: Double): Unit = {
    val xxT = features * features.t
    val rx = reward * features
    addToState(arm, xxT, rx)
  }
}

/**
 * UCB1, algorithm from:
 * https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
 *
 * @param numArms
 */
class UCB1State(numArms: Int) {
  var totalPlays = 0L
  val plays: Array[Long] = Array.fill(numArms)(0L)
  val rewards: Array[Double] = Array.fill(numArms)(0.0)

  def chooseArm(): Int = {
    var arm = 0
    var bestArm = -1
    var bestArmEstimate = Double.NegativeInfinity
    val n = getTotalPlays()
    if (n >= numArms) {
      while (arm < numArms) {
        val (playsForThisArm, rewardsForThisArm) = getState(arm)
        val rewardEstimate = if (playsForThisArm > 0) {
          // If the arm has been played, compute a reward.
          (rewardsForThisArm / playsForThisArm) + math.sqrt(2.0*math.log(n)/playsForThisArm)

        } else {
          Double.PositiveInfinity
        }

        if (rewardEstimate > bestArmEstimate) {
          bestArm = arm
          bestArmEstimate = rewardEstimate
        }

        arm += 1
      }
    } else {
      // Choose a random unplayed arm
      val armsWithNoPlays = plays.zipWithIndex.filter(_._1==0).map(_._2)
      if (armsWithNoPlays.length > 0) {
        bestArm = armsWithNoPlays(scala.util.Random.nextInt(armsWithNoPlays.length))
      } else {
        // All the arms have been played & feedback has been provided while this has been happening
        bestArm = chooseArm()
      }
    }

    bestArm
  }

  protected def getState(arm: Int): (Long, Double) = this.synchronized {
    (plays(arm), rewards(arm))
  }

  protected def getTotalPlays(): Long = this.synchronized {
    totalPlays
  }

  protected def addToState(arm: Int, numPlays: Long, totalRewards: Double): Unit =
    this.synchronized {
      plays(arm) += numPlays
      rewards(arm) += totalRewards
      totalPlays += numPlays
    }

  def updateState(arm: Int, reward: Double): Unit = {
    addToState(arm, 1, reward)
  }
}

/**
 * Epsilon-greedy
 *
 * Fixme: The first action made by each thread will be to select the last arm
 *
 * @param numArms
 * @param epsilon Percent of the time to explore as opposed to exploit. Value between 0 and 1.
 */
class EpsilonGreedyState(numArms: Int, epsilon: Double) {
  val plays: Array[Long] = Array.fill(numArms)(0L)
  val rewards: Array[Double] = Array.fill(numArms)(0.0)

  def chooseArm(): Int = {
    var arm = 0
    var bestArm = -1
    var bestArmEstimate = Double.NegativeInfinity
    val rand = scala.util.Random.nextFloat()
    if (rand > epsilon) {
      // Exploit: Choose the best arm
      while (arm < numArms) {
        val (playsForThisArm, rewardsForThisArm) = getState(arm)
        val rewardEstimate = if (playsForThisArm > 0) {
          // If the arm has been played, compute a reward.
          rewardsForThisArm / playsForThisArm
        } else {
          Double.PositiveInfinity
        }

        if (rewardEstimate > bestArmEstimate) {
          bestArm = arm
          bestArmEstimate = rewardEstimate
        }

        arm += 1
      }
    } else {
      // Explore: Choose a random arm
      bestArm = scala.util.Random.nextInt(numArms)
    }

    bestArm
  }

  protected def getState(arm: Int): (Long, Double) = this.synchronized {
    (plays(arm), rewards(arm))
  }

  protected def addToState(arm: Int, numPlays: Long, totalRewards: Double): Unit =
    this.synchronized {
      plays(arm) += numPlays
      rewards(arm) += totalRewards
    }

  def updateState(arm: Int, reward: Double): Unit = {
    addToState(arm, 1, reward)
  }
}
