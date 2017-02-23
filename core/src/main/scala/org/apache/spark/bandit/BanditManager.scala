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

import java.util.concurrent.{ConcurrentHashMap, ScheduledExecutorService, ScheduledFuture, TimeUnit}
import java.util.concurrent.atomic.AtomicLong

import scala.reflect.ClassTag
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.bandit.policies._
import org.apache.spark.internal.Logging
import org.apache.spark.rpc.{RpcEndpointRef, RpcEnv}
import org.apache.spark.util.{SparkExitCode, ThreadUtils, Utils}
import org.apache.spark.{SecurityManager, SparkConf}

import scala.collection.mutable
import scala.concurrent.ExecutionContext


// TODO: FIXME: The bandit manager setup is fairly hacky right now, w/o proper garbage collection!
// Really should have a full setup w/ slave endpoints & the master endpoint, and communication
// Between them both to do learning & to do garbage collection, but for the time being we're not
// mucking around with that just to get this working asap.

// There are likely to be some excessive data sends, but not the end of the world
// Akka ensures ordering messages between sender-reciever pairs, so I believe it
// should receive distributed updates in the correct order
private[spark] class BanditManager(
    val isDriver: Boolean,
    val executorId: String,
    conf: SparkConf,
    val master: RpcEndpointRef,
    securityManager: SecurityManager)
  extends Logging {

  private var initialized = false
  private val policies = new ConcurrentHashMap[Long, (BanditPolicy, (Array[Long], Array[Double]))]()
  private val contextualPolicies = new ConcurrentHashMap[Long,
      (ContextualBanditPolicy, (Array[DenseMatrix[Double]], Array[DenseVector[Double]]))]()
  private val updatedBandits = mutable.Set[Long]()

  initialize()

  // "eventLoopThread" is used to run some pretty fast actions. The actions running in it should not
  // block the thread for a long time.
  private var banditDistributedUpdateTask: ScheduledFuture[_] = null

  private var eventLoopThread: ScheduledExecutorService = null

  // Called by SparkContext or Executor before using Broadcast
  private def initialize() {
    synchronized {
      if (!initialized) {
        eventLoopThread = ThreadUtils.newDaemonSingleThreadScheduledExecutor(
          "bandit-manager-event-loop-thread")

        // Used to provide the implicit parameter of `Future` methods.
        val forwardMessageExecutionContext =
        ExecutionContext.fromExecutor(eventLoopThread,
          t => t match {
            case ie: InterruptedException => // Exit normally
            case e: Throwable =>
              logError(e.getMessage, e)
              System.exit(SparkExitCode.UNCAUGHT_EXCEPTION)
          })

        banditDistributedUpdateTask = eventLoopThread.scheduleAtFixedRate(new Runnable {
          override def run(): Unit = Utils.tryLogNonFatalError {
            // Clone & clear the ids to update
            val ids = updatedBandits.synchronized {
              val idSetClone = updatedBandits.toSeq
              updatedBandits.clear()
              idSetClone
            }

            // Construct the updates, setting locks as necessary
            val updates: Seq[BanditUpdate] = ids.map { id =>
              if (policies.containsKey(id)) {
                val (policy, (localPlays, localRewards)) = policies.get(id)
                policy.stateLock.synchronized {
                  MABBanditUpdate(id, localPlays.clone(), localRewards.clone())
                }
              } else {
                val (policy, (localFeatures, localRewards)) = contextualPolicies.get(id)
                policy.stateLock.synchronized {
                  ContextualBanditUpdate(id, localFeatures.clone(), localRewards.clone())
                }
              }
            }

            if (updates.nonEmpty) {
              master.ask[SendDistributedUpdates](SendLocalUpdates(executorId, updates)).onComplete {
                _.foreach {
                  _.updates.foreach {
                    case ContextualBanditUpdate(id, features, rewards) =>
                      mergeDistributedContextualFeedback(id, features, rewards)
                    case MABBanditUpdate(id, plays, rewards) =>
                      mergeDistributedFeedback(id, plays, rewards)
                  }
                }
              }(forwardMessageExecutionContext)
            }
          }
        }, 0, 5000, TimeUnit.MILLISECONDS)

        initialized = true
      }
    }
  }

  def stop() {
    if (banditDistributedUpdateTask != null) {
      banditDistributedUpdateTask.cancel(true)
    }
    if (eventLoopThread != null) {
      eventLoopThread.shutdownNow()
    }
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

    updatedBandits.synchronized {
      updatedBandits.add(id)
    }
  }

  def mergeDistributedFeedback(id: Long,
                               plays: Array[Long],
                               rewards: Array[Double]): Unit = {
    val (policy, (localPlays, localRewards)) = policies.get(id)

    policy.stateLock.synchronized {
      for (arm <- 0 until policy.numArms) {
        val newPlays = localPlays(arm) + plays(arm)
        val newRewards = localRewards(arm) + rewards(arm)

        policy.setState(arm, newPlays, newRewards)
      }
    }

    updatedBandits.synchronized {
      updatedBandits.add(id)
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

    updatedBandits.synchronized {
      updatedBandits.add(id)
    }
  }

  def mergeDistributedContextualFeedback(id: Long,
                                         features: Array[DenseMatrix[Double]],
                                         rewards: Array[DenseVector[Double]]): Unit = {
    logInfo(s"feedback: $id, ${features.toSeq} ${rewards.toSeq}")

    val (policy, (localFeatures, localRewards)) = contextualPolicies.get(id)

    val eye = DenseMatrix.eye[Double](policy.numFeatures)
    policy.stateLock.synchronized {
      for (arm <- 0 until policy.numArms) {
        val newFeatures = eye + localFeatures(arm) + features(arm)
        val newRewards = localRewards(arm) + rewards(arm)

        policy.setState(arm, newFeatures, newRewards)
      }
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
    logInfo("Removing bandit $id")
    policies.remove(id)
    contextualPolicies.remove(id)

    // TODO: Add distributed delete also
  }
}
