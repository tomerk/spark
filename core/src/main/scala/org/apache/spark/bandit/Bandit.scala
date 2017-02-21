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
      val (a, b) = this.synchronized {
        (as(arm), bs(arm))
      }

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

  def updateState(features: DenseVector[Double], arm: Int, reward: Double): Unit = {
    val xxT = features * features.t
    val rx = reward * features
    this.synchronized {
      as(arm) = as(arm) + xxT
      bs(arm) = bs(arm) + rx
    }
  }
}
