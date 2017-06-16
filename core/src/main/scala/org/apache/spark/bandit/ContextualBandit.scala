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

import scala.reflect.ClassTag
import breeze.linalg.DenseVector
import org.apache.spark.SparkEnv
import org.apache.spark.bandit.policies.ContextualBanditPolicy
import org.apache.spark.internal.Logging
import org.apache.spark.util.StatCounter

case class Action(arm: Int, reward: Double)

class DelayedContextualBanditFeedbackProvider(bandit: ContextualBandit[_, _],
                                    arm: Int,
                                    features: DenseVector[Double],
                                    initRuntime: Long,
                                    threadId: Long
                                             ) extends DelayedFeedbackProvider with Logging {
  var totalTime = initRuntime
  def banditId: Long = bandit.id
  def getRuntime: Long = totalTime
  def provide(reward: Double): Unit = {
    //logError(s"Join ${bandit.id} by $arm took $reward")
    bandit.provideDelayedContextualFeedback(threadId, arm, reward, features)
  }

  override def getArm: Int = arm
}

/**
 * The contextual bandit class is used for dynamically tuned methods appearing in spark tasks.
 */
class ContextualBandit[A: ClassTag, B: ClassTag] private[spark] (val id: Long,
                                                 val arms: Seq[A => B],
                                                 val featureExtractor: A => DenseVector[Double],
                                                 private var initPolicy: ContextualBanditPolicy
                                                ) extends BanditTrait[A, B] with Logging {

  @transient private lazy val banditManager = SparkEnv.get.banditManager

  /**
   * Given a single input, choose a single arm to apply to that input, and
   * update the policy with the observed runtime.
   *
   * @param in The input item
   */
  def apply(in: A): B = {
    val threadId = if (banditManager.alwaysShare) {
      0
    } else {
      java.lang.Thread.currentThread().getId
    }

    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)

    val features = featureExtractor(in)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    val reward = new WeightedStats().add(startTime - endTime)
    banditManager.provideContextualFeedback(id, threadId, arm, features, reward)
    result
  }

  def applyAndDelayFeedback(in: A): (B, DelayedContextualBanditFeedbackProvider) = {
    val threadId = if (banditManager.alwaysShare) {
      0
    } else {
      java.lang.Thread.currentThread().getId
    }

    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)

    val features = featureExtractor(in)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    val delayedProvider = new DelayedContextualBanditFeedbackProvider(
      this, arm, features, endTime - startTime, threadId)
    (result, delayedProvider)
  }

  def applyAndOutputReward(in: A): (B, Action) = {
    val threadId = if (banditManager.alwaysShare) {
      0
    } else {
      java.lang.Thread.currentThread().getId
    }

    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)

    val features = featureExtractor(in)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    val reward = new WeightedStats().add(startTime - endTime)
    banditManager.provideContextualFeedback(id, threadId, arm, features, reward)
    (result, Action(arm, startTime - endTime))
  }

  /**
   * A vectorized bandit strategy. Given a sequence of input, choose a single arm
   * to apply to all of the input. The learning for all the items will be batched
   * given that one arm was selected.
   *
   * WARNING: Do not pass in a lazy seq (stream).
   * @param in The vector of input
   */
  def vectorizedApply(in: Seq[A]): Seq[B] = {
    val threadId = if (banditManager.alwaysShare) {
      0
    } else {
      java.lang.Thread.currentThread().getId
    }

    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)

    // Because our contextual models our linear, the features is just over the individual inputs
    val features = in.map(featureExtractor).reduce(_ + _) / in.length.toDouble
    val arm = policy.chooseArm(features)

    var sumRewards: Long = 0L
    val seqResult = in.map { x =>
      val startTime = System.nanoTime()
      val result = arms(arm).apply(x)
      val endTime = System.nanoTime()
      sumRewards += startTime - endTime
      result
    }

    val rewards = new WeightedStats().add(sumRewards / in.length.toDouble)

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    banditManager.provideContextualFeedback(id, threadId, arm, features, rewards)
    seqResult
  }

  protected[bandit] def provideDelayedContextualFeedback(
      threadId: Long,
      arm: Int,
      reward: Double,
      features: DenseVector[Double]
    ): Unit = {
    val rewardStats = new WeightedStats().add(reward)

    banditManager.provideContextualFeedback(id, threadId, arm, features, rewardStats)
  }


  def saveTunedSettings(file: String): Unit = {
    throw new NotImplementedError()
  }
  def loadTunedSettings(file: String): Unit = {
    throw new NotImplementedError()
  }
}
