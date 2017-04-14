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
import breeze.linalg.{DenseMatrix, DenseVector, norm}
import org.apache.spark.bandit.policies._
import org.apache.spark.internal.Logging
import org.apache.spark.rpc.{RpcEndpointRef, RpcEnv}
import org.apache.spark.util.{SparkExitCode, StatCounter, ThreadUtils, Utils}
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

  // policy is: (banditId, threadId) -> (policy, recentFeedback, olderfeedback)
  private val policies = {
    new ConcurrentHashMap[(Long, Long),
      (BanditPolicy, Array[WeightedStats], Array[WeightedStats])]()
  }

  private val contextualPolicies = new ConcurrentHashMap[(Long, Long),
      (ContextualBanditPolicy,
        (Array[DenseMatrix[Double]],
          Array[DenseVector[Double]],
          Array[WeightedStats]),
        (Array[DenseMatrix[Double]],
            Array[DenseVector[Double]],
            Array[WeightedStats]))]()
  private val updatedBandits = mutable.Set[(Long, Long)]()
  private val driftUpdatedBandits = mutable.Set[(Long, Long)]()

  initialize()

  // "eventLoopThread" is used to run some pretty fast actions. The actions running in it should not
  // block the thread for a long time.
  private var banditDistributedUpdateTask: ScheduledFuture[_] = null
  private var banditDriftDetectTask: ScheduledFuture[_] = null

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

        val driftDetectRate = conf.getTimeAsMs("spark.bandits.driftDetectionRate", "5s")
        val driftDetectConstant = conf.getDouble("spark.bandits.driftCoefficient", 0.25)
        if (driftDetectRate > 0) {
          banditDriftDetectTask = eventLoopThread.scheduleAtFixedRate(new Runnable {
            override def run(): Unit = Utils.tryLogNonFatalError {
              // Clone & clear the ids to update
              val ids = driftUpdatedBandits.synchronized {
                val idSetClone = driftUpdatedBandits.toSeq
                driftUpdatedBandits.clear()
                idSetClone
              }

              // Prepare any drifted bandits for a local reset upon the next
              // distributed update
              ids.foreach { case (id, threadId) =>
                if (policies.containsKey((id, threadId))) {
                  val (policy, recentRewards, oldRewards) = policies.get((id, threadId))

                  val arms = policy.numArms
                  for (i <- 0 until arms) {
                    policy.stateLock.synchronized {
                      // Identify whether drifting occurred
                      val meanDiff = math.abs(recentRewards(i).mean - oldRewards(i).mean)

                      // confidence bound for the old rewards
                      val oldCb = {
                        driftDetectConstant * math.sqrt(
                          oldRewards(i).variance *
                            (1 + math.log(1 + oldRewards(i).totalWeights)) /
                            (1 + oldRewards(i).totalWeights)
                        )
                      }

                      // confidence bound for recent rewards
                      val cb = {
                        driftDetectConstant * math.sqrt(
                          recentRewards(i).variance *
                            (1 + math.log(1 + recentRewards(i).totalWeights)) /
                            (1 + recentRewards(i).totalWeights)
                        )
                      }

                      // If drifting was detected, reset our knowledge.
                      // Otherwise move the recent observations to the old ones
                      if (meanDiff > cb + oldCb) {
                        oldRewards(i) = recentRewards(i)
                      } else {
                        oldRewards(i).merge(recentRewards(i))
                      }

                      // Then reset the recent observations tracker
                      recentRewards(i) = new WeightedStats()
                    }
                  }
                } else {
                  val (policy,
                  (recentFeatures, recentRewards, recentRewardStats),
                  (oldFeatures, oldRewards, oldRewardStats)) = {
                    contextualPolicies.get((id, threadId))
                  }

                  val arms = policy.numArms
                  for (i <- 0 until arms) {
                    policy.stateLock.synchronized {
                      val numFeatures = recentFeatures(i).rows
                      val eye = DenseMatrix.eye[Double](numFeatures)
                      val recentWeights = (recentFeatures(i) + eye) \ recentRewards(i)
                      val oldWeights = (oldFeatures(i) + eye) \ oldRewards(i)

                      val meanOfNorms = (norm(recentWeights(i)) + norm(oldWeights(i))) / 2.0
                      val meanDiff = norm(recentWeights(i) - oldWeights(i)) / meanOfNorms

                      // confidence bound for the other reward
                      val otherCb = {
                        driftDetectConstant * math.sqrt(
                          (1 + math.log(1 + oldRewardStats(i).totalWeights)) /
                            (1 + oldRewardStats(i).totalWeights)
                        )
                      }

                      // confidence bound for this reward
                      val cb = {
                        driftDetectConstant * math.sqrt(
                          (1 + math.log(1 + recentRewardStats(i).totalWeights)) /
                            (1 + recentRewardStats(i).totalWeights)
                        )
                      }

                      // If drifting was detected, reset our knowledge.
                      // Otherwise move the recent observations to the old ones
                      if (meanDiff > cb + otherCb) {
                        oldFeatures(i) = recentFeatures(i)
                        oldRewards(i) = recentRewards(i)
                        oldRewardStats(i) = recentRewardStats(i)
                      } else {
                        oldFeatures(i) = recentFeatures(i) + oldFeatures(i)
                        oldRewards(i) = recentRewards(i) + oldRewards(i)
                        oldRewardStats(i).merge(recentRewardStats(i))
                      }

                      // Then reset the recent observations tracker
                      recentFeatures(i) = DenseMatrix.zeros(numFeatures, numFeatures)
                      recentRewards(i) = DenseVector.zeros(numFeatures)
                      recentRewardStats(i) = new WeightedStats()
                    }
                  }
                }
              }
            }
          }, 0, driftDetectRate, TimeUnit.MILLISECONDS)
        }

        val communicationRate = conf.getTimeAsMs("spark.bandits.communicationRate", "5s")
        if (communicationRate > 0) {
          banditDistributedUpdateTask = eventLoopThread.scheduleAtFixedRate(new Runnable {
            override def run(): Unit = Utils.tryLogNonFatalError {
              // Clone & clear the ids to update
              val ids = updatedBandits.synchronized {
                val idSetClone = updatedBandits.toSeq
                updatedBandits.clear()
                idSetClone
              }

              // Construct the updates, setting locks as necessary
              val updates: Seq[BanditUpdate] = ids.map { case (id, threadId) =>
                if (policies.containsKey((id, threadId))) {
                  val (
                    policy, localRewards, oldRewards
                    ) = policies.get((id, threadId))
                  val totalLocalRewards = localRewards.zip(oldRewards).map {
                    x => x._1.copy().merge(x._2)
                  }

                  policy.stateLock.synchronized {
                    MABBanditUpdate(id, threadId, totalLocalRewards)
                  }
                } else {
                  val (policy,
                  (localFeatures, localRewards, localRewardStats),
                  (oldLocalFeatures, oldLocalRewards, oldLocalRewardStats)) = {
                    contextualPolicies.get((id, threadId))
                  }
                  val (f, r, rs) = policy.stateLock.synchronized {
                    (
                      localFeatures.zip(oldLocalFeatures).map(x => x._1 + x._2),
                      localRewards.zip(oldLocalRewards).map(x => x._1 + x._2),
                      localRewardStats.zip(oldLocalRewardStats).map(x => x._1.copy().merge(x._2)))
                  }
                  val weights = Array.fill[DenseVector[Double]](f.length)(null)
                  var i = 0
                  val eye = DenseMatrix.eye[Double](f(i).rows)
                  while (i < f.length) {
                    weights(i) = (f(i) + eye) \ r(i)
                    i += 1
                  }
                  ContextualBanditUpdate(id,
                    threadId,
                    f,
                    r,
                    rs,
                  weights = weights)
                }
              }

              if (updates.nonEmpty) {
                master.ask[SendDistributedUpdates](
                  SendLocalUpdates(executorId, updates)
                ).onComplete {
                  _.foreach {
                    _.updates.foreach {
                      case ContextualBanditUpdate(
                      id,
                      threadId,
                      features,
                      rewards,
                      rewardStats,
                      _) =>
                        mergeDistributedContextualFeedback(
                          id, threadId, features, rewards, rewardStats)
                      case MABBanditUpdate(id, threadId, rewards) =>
                        mergeDistributedFeedback(id, threadId, rewards)
                    }
                  }
                }(forwardMessageExecutionContext)
              }
            }
          }, 0, communicationRate, TimeUnit.MILLISECONDS)
        }
        initialized = true
      }
    }
  }

  def stop() {
    if (banditDistributedUpdateTask != null) {
      banditDistributedUpdateTask.cancel(true)
    }
    if (banditDriftDetectTask != null) {
      banditDriftDetectTask.cancel(true)
    }
    if (eventLoopThread != null) {
      eventLoopThread.shutdownNow()
    }
  }

  def registerOrLoadPolicy(id: Long, threadId: Long, policy: BanditPolicy): BanditPolicy = {
    var policyAtId = policies.get((id, threadId))

    // Init the policy if necessary
    if (policyAtId == null) {
      policies.putIfAbsent((id, threadId), (policy,
        Array.fill(policy.numArms)(new WeightedStats),
        Array.fill(policy.numArms)(new WeightedStats)))
      policyAtId = policies.get((id, threadId))
    }
    policyAtId._1
  }

  def registerOrLoadPolicy(
                            id: Long,
                            threadId: Long,
                            policy: ContextualBanditPolicy
                          ): ContextualBanditPolicy = {
    var policyAtId = contextualPolicies.get((id, threadId))

    // Init the policy if necessary
    if (policyAtId == null) {
      val initFeatures: Array[DenseMatrix[Double]] = Array.fill(policy.numArms) {
        DenseMatrix.zeros(policy.numFeatures, policy.numFeatures)
      }

      val initRewards: Array[DenseVector[Double]] = Array.fill(policy.numArms) {
        DenseVector.zeros(policy.numFeatures)
      }

      val initRewardStats: Array[WeightedStats] = Array.fill(policy.numArms) {
        new WeightedStats()
      }

      val initOldFeatures: Array[DenseMatrix[Double]] = Array.fill(policy.numArms) {
        DenseMatrix.zeros(policy.numFeatures, policy.numFeatures)
      }

      val initOldRewards: Array[DenseVector[Double]] = Array.fill(policy.numArms) {
        DenseVector.zeros(policy.numFeatures)
      }

      val initOldRewardStats: Array[WeightedStats] = Array.fill(policy.numArms) {
        new WeightedStats()
      }

      contextualPolicies.putIfAbsent((id, threadId), (policy,
        (initFeatures, initRewards, initRewardStats),
        (initOldFeatures, initOldRewards, initOldRewardStats)))

      policyAtId = contextualPolicies.get((id, threadId))
    }

    policyAtId._1
  }

  def provideFeedback(id: Long,
                      threadId: Long,
                      arm: Int,
                      plays: Long,
                      reward: Double): Unit = {
    val (policy, localRewards, _) = policies.get((id, threadId))
    policy.stateLock.synchronized {
      policy.provideFeedback(arm, plays, reward)
      localRewards(arm).addMultiple(reward, 1.0, plays)
    }

    updatedBandits.synchronized {
      updatedBandits.add((id, threadId))
    }
    driftUpdatedBandits.synchronized {
      driftUpdatedBandits.add((id, threadId))
    }
  }

  def mergeDistributedFeedback(id: Long,
                               threadId: Long,
                               rewards: Array[WeightedStats]
                               ): Unit = {
    val (policy, localRewards, oldLocalRewards) = policies.get((id, threadId))

    policy.stateLock.synchronized {
      for (arm <- 0 until policy.numArms) {
        val newRewards = rewards(arm).merge(localRewards(arm)).merge(oldLocalRewards(arm))

        policy.setState(arm, newRewards)
      }
    }
  }

  def provideContextualFeedback(id: Long,
                                threadId: Long,
                                arm: Int,
                                features: DenseVector[Double],
                                rewardStats: WeightedStats): Unit = {
    val (policy, (localFeatures, localRewards, localRewardStats), _) = {
      contextualPolicies.get((id, threadId))
    }
    // Note that this assumes all weights are 1
    val xxT = features * features.t
    val rx = rewardStats.count * rewardStats.mean * features

    policy.stateLock.synchronized {
      policy.provideFeedback(arm, xxT, rx, rewardStats)
      localFeatures(arm) = localFeatures(arm) + xxT
      localRewards(arm) = localRewards(arm) + rx
      localRewardStats(arm).merge(rewardStats)
    }

    updatedBandits.synchronized {
      updatedBandits.add((id, threadId))
    }
    driftUpdatedBandits.synchronized {
      driftUpdatedBandits.add((id, threadId))
    }
  }

  def mergeDistributedContextualFeedback(id: Long,
                                         threadId: Long,
                                         features: Array[DenseMatrix[Double]],
                                         rewards: Array[DenseVector[Double]],
                                         rewardStats: Array[WeightedStats]): Unit = {
    logInfo(s"feedback: $id, ${features.toSeq} ${rewards.toSeq}")

    val (policy,
    (localFeatures, localRewards, localRewardStats),
    (oldLocalFeatures, oldLocalRewards, oldLocalRewardStats)) = {
      contextualPolicies.get((id, threadId))
    }

    val eye = DenseMatrix.eye[Double](policy.numFeatures)
    policy.stateLock.synchronized {
      for (arm <- 0 until policy.numArms) {
        val newFeatures = eye + localFeatures(arm) + oldLocalFeatures(arm) + features(arm)
        val newRewards = localRewards(arm) + oldLocalRewards(arm) + rewards(arm)
        val newRewardStats = {
          rewardStats(arm)
            .merge(localRewardStats(arm))
            .merge(oldLocalRewardStats(arm))
        }

        policy.setState(arm, newFeatures, newRewards, newRewardStats)
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
      case ConstantPolicyParams(arm) =>
        new ConstantPolicy(numArms = arms.length, arm)
      case EpsilonGreedyPolicyParams(epsilon) =>
        new EpsilonGreedyPolicy(numArms = arms.length, epsilon)
      case GaussianThompsonSamplingPolicyParams(multiplier) =>
        new GaussianThompsonSamplingPolicy(numArms = arms.length, varianceMultiplier = multiplier)
      case UCB1PolicyParams(range) =>
        new UCB1Policy(numArms = arms.length, boundsConst = range)
      case UCB1NormalPolicyParams(range) =>
        new UCB1NormalPolicy(numArms = arms.length, boundsConst = range)
      case GaussianBayesUCBPolicyParams(range) =>
        new GaussianBayesUCBPolicy(numArms = arms.length, boundsConst = range)
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
      case LinThompsonSamplingPolicyParams(numFeatures, v, useCholesky) =>
        new LinThompsonSamplingPolicy(numArms = arms.length, numFeatures, v, useCholesky)
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
    // TODO Fixme: Need to remove that id W/ ALL THREAD IDS!
  }
}
