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

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

import scala.reflect.ClassTag
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.bandit.policies._
import org.apache.spark.internal.Logging
import org.apache.spark.{SecurityManager, SparkConf}


private[spark] class BanditManager(
    val isDriver: Boolean,
    conf: SparkConf,
    securityManager: SecurityManager)
  extends Logging {

  private var initialized = false
  private val policies = new ConcurrentHashMap[Long, (BanditPolicy, (Array[Long], Array[Double]))]()
  private val contextualPolicies = new ConcurrentHashMap[Long,
      (ContextualBanditPolicy, (Array[DenseMatrix[Double]], Array[DenseVector[Double]]))]()

  initialize()

  // Called by SparkContext or Executor before using Broadcast
  private def initialize() {
    synchronized {
      if (!initialized) {
        initialized = true
      }
    }
  }

  def stop() {

  }

  def registerOrLoadPolicy(id: Long, policy: BanditPolicy): BanditPolicy = {
    policies.putIfAbsent(id, (policy,
      (Array.fill(policy.numArms)(0), Array.fill(policy.numArms)(0))))
    policies.get(id)._1
  }

  def registerOrLoadPolicy(id: Long, policy: ContextualBanditPolicy): ContextualBanditPolicy = {
    val initFeatures: Array[DenseMatrix[Double]] = Array.fill(policy.numArms) {
      DenseMatrix.zeros(policy.numFeatures, policy.numFeatures)
    }

    val initRewards: Array[DenseVector[Double]] = Array.fill(policy.numArms) {
      DenseVector.zeros(policy.numFeatures)
    }

    contextualPolicies.putIfAbsent(id, (policy,
      (initFeatures, initRewards)))
    contextualPolicies.get(id)._1
  }

  def provideFeedback(id: Long, arm: Int, plays: Long, reward: Double): Unit = {
    val (policy, (localPlays, localRewards)) = policies.get(id)
    policy.stateLock.synchronized {
      policy.provideFeedback(arm, plays, reward)
      localPlays(arm) += plays
      localRewards(arm) += reward
    }
  }

  def provideContextualFeedback(id: Long,
                                arm: Int,
                                features: DenseVector[Double],
                                reward: Double): Unit = {
    val (policy, (localFeatures, localRewards)) = contextualPolicies.get(id)
    val xxT = features * features.t
    val rx = reward * features

    policy.stateLock.synchronized {
      policy.provideFeedback(arm, xxT, rx)
      localFeatures(arm) = localFeatures(arm) + xxT
      localRewards(arm) = localRewards(arm) + rx
    }

  }

  private val nextBanditId = new AtomicLong(0)

  /**
   * Creates a new bandit.
   *
   * @param arms The arms to choose between
   * @param policyParams The learning policy to use
   * @param isLocal whether we are in local mode (single JVM process)
   */
  def newBandit[A: ClassTag, B: ClassTag](arms: Seq[A => B],
                                          policyParams: BanditPolicyParams,
                                          isLocal: Boolean): Bandit[A, B] = {
    val policy = policyParams match {
      case EpsilonGreedyPolicyParams(epsilon) =>
        new EpsilonGreedyPolicy(numArms = arms.length, epsilon)
      case GaussianThompsonSamplingPolicyParams() =>
        new GaussianThompsonSamplingPolicy(numArms = arms.length)
      case UCB1PolicyParams() =>
        new UCB1Policy(numArms = arms.length)
    }

    val id = nextBanditId.getAndIncrement()
    new Bandit(id, arms, policy)
  }

  /**
   * Creates a new contextual bandit.
   *
   * @param arms The arms to choose between
   * @param features The features computation to use for the bandit
   * @param policyParams The learning policy to use
   * @param isLocal whether we are in local mode (single JVM process)
   */
  def newContextualBandit[A: ClassTag, B: ClassTag](arms: Seq[A => B],
                                                    features: A => DenseVector[Double],
                                                    policyParams: ContextualBanditPolicyParams,
                                                    isLocal: Boolean): ContextualBandit[A, B] = {
    val policy = policyParams match {
      case ContextualEpsilonGreedyPolicyParams(numFeatures, epsilon) =>
        new ContextualEpsilonGreedyPolicy(numArms = arms.length, numFeatures, epsilon)
      case LinThompsonSamplingPolicyParams(numFeatures, v) =>
        new LinThompsonSamplingPolicy(numArms = arms.length, numFeatures, v)
      case LinUCBPolicyParams(numFeatures, alpha) =>
        new LinUCBPolicy(numArms = arms.length, numFeatures, alpha)
    }

    val id = nextBanditId.getAndIncrement()
    new ContextualBandit(id, arms, features, policy)
  }

  def removeBandit(id: Long, removeFromDriver: Boolean, blocking: Boolean): Unit = {
    policies.remove(id)
    contextualPolicies.remove(id)

    // TODO: Add distributed delete also
  }
}
