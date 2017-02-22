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

import java.util.concurrent.atomic.AtomicLong

import scala.reflect.ClassTag

import breeze.linalg.DenseVector

import org.apache.spark.bandit.policies._
import org.apache.spark.internal.Logging
import org.apache.spark.{SecurityManager, SparkConf}


private[spark] class BanditManager(
    val isDriver: Boolean,
    conf: SparkConf,
    securityManager: SecurityManager)
  extends Logging {

  private var initialized = false

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

  def registerOrLoadPolicy(id: Long, policy: BanditPolicy): BanditPolicy = ???

  def registerOrLoadPolicy(id: Long, policy: ContextualBanditPolicy): ContextualBanditPolicy = ???

  def provideFeedback(id: Long, policy: BanditPolicy, arm: Int, numPlays: Long, reward: Double): Unit = {

  }

  def provideContextualFeedback(id: Long,
                                policy: ContextualBanditPolicy,
                                arm: Int,
                                features: DenseVector[Double],
                                reward: Double): Unit = {

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

  def removeBandit(id: Long, removeFromDriver: Boolean, blocking: Boolean) = ???
}
