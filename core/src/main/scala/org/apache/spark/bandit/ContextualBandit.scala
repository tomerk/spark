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

case class Action(arm: Int, reward: Double)

/**
 * The contextual bandit class is used for dynamically tuned methods appearing in spark tasks.
 */
class ContextualBandit[A: ClassTag, B: ClassTag] private[spark] (val id: Long,
                                                 val arms: Seq[A => B],
                                                 val featureExtractor: A => DenseVector[Double],
                                                 private var initPolicy: ContextualBanditPolicy
                                                ) extends BanditTrait[A, B] with Logging {

  @transient private lazy val banditManager = SparkEnv.get.banditManager
  @transient private lazy val policy = {
    initPolicy = banditManager.registerOrLoadPolicy(id, initPolicy)
    initPolicy
  }

  /**
   * Given a single input, choose a single arm to apply to that input, and
   * update the policy with the observed runtime.
   *
   * @param in The input item
   */
  def apply(in: A): B = {
    val features = featureExtractor(in)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    banditManager.provideContextualFeedback(id, arm, features, startTime - endTime)
    result
  }

  def applyAndOutputReward(in: A): (B, Action) = {
    val features = featureExtractor(in)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    banditManager.provideContextualFeedback(id, arm, features, startTime - endTime)
    (result, Action(arm, startTime - endTime))
  }

  def applyAndOutputRewardAndModelUsed(in: A): (B, Action) = {
    val features = featureExtractor(in)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    banditManager.provideContextualFeedback(id, arm, features, startTime - endTime)
    (result, Action(arm, startTime - endTime))
  }


  /**
   * A vectorized bandit strategy. Given a sequence of input, choose a single arm
   * to apply to all of the input. The learning for all the items will be batched
   * given that one arm was selected.
   *
   * @param in The vector of input
   */
  def vectorizedApply(in: Seq[A]): Seq[B] = {
    // Because our contextual models our linear, the features is just over the individal inputs
    val features = in.map(featureExtractor).reduce(_ + _)
    val arm = policy.chooseArm(features)
    val startTime = System.nanoTime()
    val result = in.map(arms(arm))
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    banditManager.provideContextualFeedback(id, arm, features, startTime - endTime)
    result
  }

  def saveTunedSettings(file: String): Unit = {
    throw new NotImplementedError()
  }
  def loadTunedSettings(file: String): Unit = {
    throw new NotImplementedError()
  }
}
