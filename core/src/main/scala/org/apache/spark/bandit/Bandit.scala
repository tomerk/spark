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
import org.apache.spark.SparkEnv
import org.apache.spark.bandit.policies.BanditPolicy
import org.apache.spark.internal.Logging

trait BanditTrait[A, B] extends Serializable {
  def apply(in: A): B
  def applyAndOutputReward(in: A): (B, Action)
  def vectorizedApply(in: Seq[A]): Seq[B]
}

class DelayedBanditFeedbackProvider(bandit: Bandit[_, _],
                                    arm: Int,
                                    plays: Int,
                                    initRuntime: Long,
                                    threadId: Long) {
  var totalTime = initRuntime
  def getRuntime: Long = totalTime
  def provide(reward: Double): Unit = {
    bandit.provideDelayedFeedback(threadId, arm, reward, plays)
  }
}

/**
 * The bandit class is used for dynamically tuned methods appearing in spark tasks.
 */
class Bandit[A: ClassTag, B: ClassTag] private[spark] (val id: Long,
                                                       val arms: Seq[A => B],
                                                       private var initPolicy: BanditPolicy
                                                      ) extends BanditTrait[A, B] with Logging {

  @transient private lazy val banditManager = SparkEnv.get.banditManager

  /**
   * Given a single input, choose a single arm to apply to that input, and
   * update the policy with the observed runtime.
   *
   * @param in The input item
   */
  def apply(in: A): B = {
    val threadId = java.lang.Thread.currentThread().getId
    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)
    val arm = policy.chooseArm(1)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    val reward: Double = startTime - endTime
    banditManager.provideFeedback(id, threadId, arm, 1, reward)
    result
  }

  def applyAndDelayFeedback(in: A): (B, DelayedBanditFeedbackProvider) = {
    val threadId = java.lang.Thread.currentThread().getId
    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)
    val arm = policy.chooseArm(1)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    val reward = startTime - endTime
    val delayedProvider = new DelayedBanditFeedbackProvider(this, arm, 1, reward, threadId)
    (result, delayedProvider)
  }

  def applyAndOutputReward(in: A): (B, Action) = {
    val threadId = java.lang.Thread.currentThread().getId
    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)

    val arm = policy.chooseArm(1)
    val startTime = System.nanoTime()
    val result = arms(arm).apply(in)
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    val reward: Double = startTime - endTime

    banditManager.provideFeedback(id, threadId, arm, 1, reward)
    (result, Action(arm, startTime - endTime))
  }

  protected[bandit] def provideDelayedFeedback(
    threadId: Long,
    arm: Int,
    reward: Double,
    plays: Int
  ): Unit = {
    banditManager.provideFeedback(id, threadId, arm, plays, reward)
  }


  /**
   * A vectorized bandit strategy. Given a sequence of input, choose a single arm
   * to apply to all of the input. The learning for all the items will be batched
   * given that one arm was selected.
   *
   * @param in The vector of input
   */
  def vectorizedApply(in: Seq[A]): Seq[B] = {
    val threadId = java.lang.Thread.currentThread().getId
    val policy = banditManager.registerOrLoadPolicy(id, threadId, initPolicy)

    val arm = policy.chooseArm(in.length)
    val startTime = System.nanoTime()
    val result = in.map(arms(arm))
    val endTime = System.nanoTime()

    // Intentionally provide -1 * elapsed time as the reward, so it's better to be faster
    val reward: Double = startTime - endTime

    banditManager.provideFeedback(id, threadId, arm, in.length, reward / in.length)
    result
  }

  def saveTunedSettings(file: String): Unit = {
    throw new NotImplementedError()
  }
  def loadTunedSettings(file: String): Unit = {
    throw new NotImplementedError()
  }
}



// Also need state access & merging code that is complete, not just per-arm??!!
// (ALTHOUGH if we can send state updates for each arm separately it could lower
// the total required communication, but that's a premature optimization)


// The state & state updates are basically the same for thompson's sampling and LinUCB
// (will need to note that the thompson sampling code was modified to be on disjoint models)
// The only difference is the reward computation (and it may be worth pre-computing the inverse?)

// Future work: lower-overhead, sgd based updates!?
