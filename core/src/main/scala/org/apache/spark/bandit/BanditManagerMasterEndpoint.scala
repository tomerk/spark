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

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.rpc.{RpcCallContext, RpcEnv, ThreadSafeRpcEndpoint}
import org.apache.spark.util.StatCounter

import scala.collection.mutable

sealed trait BanditUpdate
case class ContextualBanditUpdate(banditId: Long,
                                  threadId: Long,
                                  features: Array[DenseMatrix[Double]],
                                  rewards: Array[DenseVector[Double]],
                                  rewardStats: Array[WeightedStats],
                                  weights: Array[DenseVector[Double]]
                                 ) extends BanditUpdate
case class MABBanditUpdate(banditId: Long, threadId: Long,
                           rewards: Array[WeightedStats])
  extends BanditUpdate

trait BanditManagerMessages
case class SendLocalUpdates(executorId: String, updates: Seq[BanditUpdate])
case class SendDistributedUpdates(updates: Seq[BanditUpdate])

private[spark] class BanditManagerMasterEndpoint(override val rpcEnv: RpcEnv, conf: SparkConf)
  extends ThreadSafeRpcEndpoint with Logging {

  private val banditClusterConstant: Double = {
    conf.getDouble("spark.bandits.clusterCoefficient", 0.25)
  }

  // banditId -> ((executorId, threadId) -> states)
  private val executorStates = mutable.Map[Long,
    mutable.Map[(String, Long), Array[WeightedStats]]]()
  private val executorContextualStates = mutable.Map[Long,
    mutable.Map[(String, Long), (Array[DenseMatrix[Double]], Array[DenseVector[Double]],
      Array[WeightedStats], Array[DenseVector[Double]])]]()

  override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
    case SendLocalUpdates(executorId, localUpdates) =>
      val responses = localUpdates.map {
        case MABBanditUpdate(id, threadId, rewards) =>
          // Store these observations
          val executorState = executorStates.getOrElseUpdate(id, mutable.Map())
          executorState.put((executorId, threadId), rewards)

          // Cluster observations from other partitions that could share the same reward
          // distribution, in order to merge w/ the local partition data
          val arms = rewards.length
          val responseRewards = Array.fill(arms)(new WeightedStats())
          val otherStates = executorState.filterNot(_._1 == (executorId, threadId))
          for (executor <- otherStates) {
            val otherRewards = executor._2
            for (i <- 0 until arms) {
              // Identify whether to cluster from the arm observations
              val meanDiff = math.abs(rewards(i).mean - otherRewards(i).mean)

              // confidence bound for the other reward
              val otherCb = {
                banditClusterConstant * math.sqrt(
                  otherRewards(i).variance *
                    (1 + math.log(1 + otherRewards(i).totalWeights)) /
                    (1 + otherRewards(i).totalWeights)
                )
              }

              // confidence bound for this reward
              val cb = {
                banditClusterConstant * math.sqrt(
                  rewards(i).variance *
                    (1 + math.log(1 + rewards(i).totalWeights)) /
                    (1 + rewards(i).totalWeights)
                )
              }

              // Cluster these observations if the difference is within
              // the confidence bounds
              if (meanDiff < cb + otherCb) {
                responseRewards(i).merge(otherRewards(i))
              }
            }
          }

          // Return the observations to use from the clustered partitions
          MABBanditUpdate(id, threadId, responseRewards)

        case ContextualBanditUpdate(id, threadId, features, rewards, rewardStats, weights) =>
          // Store these observations
          val executorState = executorContextualStates.getOrElseUpdate(id, mutable.Map())
          executorState.put((executorId, threadId), (features, rewards, rewardStats, weights))

          // Cluster observations from other partitions that could share the same reward
          // distribution, in order to merge w/ the local partition data
          val arms = rewards.length
          val numFeatures = features.head.rows
          val responseObservations = (
            Array.fill(arms)(DenseMatrix.zeros[Double](numFeatures, numFeatures)),
            Array.fill(arms)(DenseVector.zeros[Double](numFeatures)),
            Array.fill(arms)(new WeightedStats()))

          val otherStates = executorState.filterNot(_._1 == (executorId, threadId))
          for (executor <- otherStates) {
            val (otherFeatures, otherRewards, otherRewardStats, otherWeights) = executor._2
            for (i <- 0 until arms) {
              // Identify whether to cluster from the arm observations
              // TODO: It's probably possible to get a divide by zero error here?
              val meanOfNorms = (norm(weights(i)) + norm(otherWeights(i))) / 2.0
              val meanDiff = norm(weights(i) - otherWeights(i)) / meanOfNorms

              // confidence bound for the other reward
              val otherCb = {
                banditClusterConstant * math.sqrt(
                    (1 + math.log(1 + otherRewardStats(i).totalWeights)) /
                    (1 + otherRewardStats(i).totalWeights)
                )
              }

              // confidence bound for this reward
              val cb = {
                banditClusterConstant * math.sqrt(
                  (1 + math.log(1 + rewardStats(i).totalWeights)) /
                    (1 + rewardStats(i).totalWeights)
                )
              }

              // Cluster these observations if the difference is within
              // the confidence bounds
              if (meanDiff < cb + otherCb) {
                responseObservations._1(i) = responseObservations._1(i) + otherFeatures(i)
                responseObservations._2(i) = responseObservations._2(i) + otherRewards(i)
                responseObservations._3(i).merge(otherRewardStats(i))
              }
            }
          }

          // Return the observations to use from the clustered partitions
          ContextualBanditUpdate(id,
            threadId,
            responseObservations._1,
            responseObservations._2,
            responseObservations._3, weights)
      }

      context.reply(SendDistributedUpdates(responses))
  }

}

private[spark] object BanditManagerMaster {
  val DRIVER_ENDPOINT_NAME = "BanditManagerMaster"
}
