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

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.numerics.sqrt
import breeze.stats.distributions.MultivariateGaussian

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

// FIXME!
// TODO! FIXME! Write these w/ a method to just compute rewards, that way
// random-tiebreakers & multi-objective comparators can easily be used,
// and the code for synchronization & reward computation can be much simplified

// Arm choosing logic: arg-max on rewards, filter seq to only
// include max-valued arms, select random index in filtered list

// Also need state access & merging code that is complete, not just per-arm??!!
// (ALTHOUGH if we can send state updates for each arm separately it could lower
// the total required communication, but that's a premature optimization)


// The state & state updates are basically the same for thompson's sampling and LinUCB
// (will need to note that the thompson sampling code was modified to be on disjoint models)
// The only difference is the reward computation (and it may be worth pre-computing the inverse?)

// Future work: lower-overhead, sgd based updates!

abstract class ContextualBanditPolicy(numArms: Int, numFeatures: Int) extends Serializable {
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


/**
 * Linear Thompson Sampling,
 * Thompson Sampling for Contextual Bandits with Linear Payoffs
 * Agrawal et al. JMLR
 *
 * http://jmlr.csail.mit.edu/proceedings/papers/v28/agrawal13.pdf
 *
 * @param numArms
 * @param numFeatures
 * @param v Assuming rewards are R-sub-gaussian,
 *          set v = R*sqrt((24/epsilon)*numFeatures*ln(1/delta))
 *
 *          Practically speaking I'm not really sure what values to set,
 *          so I'll default to 5? Larger v means larger variance & more
 *          weight on sampling arms w/o the highest expectation
 */
class LinThompsonSamplingPolicy(numArms: Int, numFeatures: Int, v: Double = 5.0)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armRewardsAcc: DenseVector[Double]): Double = {
    // TODO: Should be able to optimize code by only computing coefficientMean after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    val coefficientMean = armFeaturesAcc \ armRewardsAcc
    val coefficientDist = MultivariateGaussian(coefficientMean, v*v*inv(armFeaturesAcc))
    val coefficientSample = coefficientDist.draw()

    coefficientSample.t*features
  }
}

/**
 * LinUCB, from:
 * A Contextual-Bandit Approach to Personalized News Article Recommendation
 * (Li et al, 2010)
 * http://research.cs.rutgers.edu/~lihong/pub/Li10Contextual.pdf
 *
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
class LinUCBPolicy(numArms: Int, numFeatures: Int, alpha: Double = 2.36)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armRewardsAcc: DenseVector[Double]): Double = {
    // TODO: Should be able to optimize code by only computing coefficientEstimate after
    // updates. Would require an update to ContextualBanditPolicy to apply optimization
    // to all contextual bandits.
    val coefficientEstimate = armFeaturesAcc \ armRewardsAcc
    coefficientEstimate.t*features + alpha*sqrt(features.t*(armFeaturesAcc \ features))
  }
}

/**
 * Note: Unplayed arms will default to zero reward. Fine in the case where our rewards
 * are strictly negative (e.g. reward = -1*runtime), but otherwise could cause issues.
 *
 * @param numArms
 * @param numFeatures
 * @param epsilon
 */
class ContextualEpsilonGreedyPolicy(numArms: Int, numFeatures: Int, epsilon: Double = 0.2)
  extends ContextualBanditPolicy(numArms, numFeatures) {
  override def chooseArm(features: DenseVector[Double]): Int = {
    if (scala.util.Random.nextFloat() >= epsilon) {
      super.chooseArm(features)
    } else {
      scala.util.Random.nextInt(numArms)
    }
  }

  override protected def estimateRewards(features: DenseVector[Double],
                                         armFeaturesAcc: DenseMatrix[Double],
                                         armRewardsAcc: DenseVector[Double]): Double = {
    // Note: Unplayed arms will default to zero reward. Fine
    val coefficientEstimate = armFeaturesAcc \ armRewardsAcc
    coefficientEstimate.t*features
  }
}

abstract class BanditPolicy(numArms: Int) extends Serializable {
  @transient lazy private val stateLock = this
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

  /**
   * Warning: Does not lock the state of the other policy!
   * @param otherPolicy
   */
  def subtractState(otherPolicy: BanditPolicy): Unit = stateLock.synchronized {
    for (arm <- 0 until numArms) {
      totalPlays(arm) -= otherPolicy.totalPlays(arm)
      totalRewards(arm) -= otherPolicy.totalRewards(arm)
    }
  }

  /**
   * Warning: Does not lock the state of the other policy!
   * @param otherPolicy
   */
  def addState(otherPolicy: BanditPolicy): Unit = stateLock.synchronized {
    for (arm <- 0 until numArms) {
      totalPlays(arm) += otherPolicy.totalPlays(arm)
      totalRewards(arm) += otherPolicy.totalRewards(arm)
    }
  }
}

/**
 * UCB1, algorithm from:
 * https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
 *
 * @param numArms
 */
class UCB1Policy(numArms: Int) extends BanditPolicy(numArms) {
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

/**
 * Epsilon-greedy
 *
 * @param numArms
 * @param epsilon Percent of the time to explore as opposed to exploit. Value between 0 and 1.
 */
class EpsilonGreedyPolicy(numArms: Int, epsilon: Double) extends BanditPolicy(numArms) {
  override def chooseArm(plays: Int): Int = {
    if (scala.util.Random.nextFloat() >= epsilon) {
      super.chooseArm(plays)
    } else {
      scala.util.Random.nextInt(numArms)
    }
  }

  override protected def estimateRewards(playsToMake: Int,
                                         totalPlays: Array[Long],
                                         totalRewards: Array[Double]): Seq[Double] = {
    (0 until numArms).map { arm =>
      if (totalPlays(arm) > 0) {
        totalRewards(arm) / totalPlays(arm)
      } else {
        Double.PositiveInfinity
      }
    }
  }
}

