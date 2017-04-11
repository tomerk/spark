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
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.rpc.{RpcCallContext, RpcEnv, ThreadSafeRpcEndpoint}
import org.apache.spark.util.StatCounter

import scala.collection.mutable

sealed trait BanditUpdate
case class ContextualBanditUpdate(banditId: Long,
                                  features: Array[DenseMatrix[Double]],
                                  rewards: Array[DenseVector[Double]],
                                  rewardStats: Array[WeightedStats],
                                  weights: Array[DenseVector[Double]]
                                 ) extends BanditUpdate
case class MABBanditUpdate(banditId: Long,
                           rewards: Array[WeightedStats])
  extends BanditUpdate

trait BanditManagerMessages
case class SendLocalUpdates(executorId: String, updates: Seq[BanditUpdate])
case class SendDistributedUpdates(updates: Seq[BanditUpdate])

private[spark] class BanditManagerMasterEndpoint(override val rpcEnv: RpcEnv, conf: SparkConf)
  extends ThreadSafeRpcEndpoint with Logging {

  private val executorStates = mutable.Map[String,
    mutable.Map[Long, (Array[Long], Array[Double], Array[Double])]]()
  private val executorContextualStates = mutable.Map[String,
    mutable.Map[Long, (Array[DenseMatrix[Double]], Array[DenseVector[Double]],
      Array[StatCounter])]]()

  private val states = mutable.Map[Long, (Array[Long], Array[Double], Array[Double])]()
  private val contextualStates = mutable.Map[Long,
    (Array[DenseMatrix[Double]], Array[DenseVector[Double]], Array[StatCounter])]()


  override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
    case SendLocalUpdates(executorId, localUpdates) =>
      val responses = localUpdates.map {
        case MABBanditUpdate(id, plays, rewards, rewardsSquared) =>
          val arms = plays.length
          val executorState = executorStates.getOrElseUpdate(executorId, mutable.Map())
          val oldLocalState = executorState.getOrElse(id,
            (Array.fill(arms)(0L), Array.fill(arms)(0.0), Array.fill(arms)(0.0)))

          val globalState = states.getOrElseUpdate(id,
            (Array.fill(arms)(0L), Array.fill(arms)(0.0), Array.fill(arms)(0.0)))

          for (i <- 0 until arms) {
            globalState._1(i) -= oldLocalState._1(i)
            globalState._2(i) -= oldLocalState._2(i)
            globalState._3(i) -= oldLocalState._3(i)
          }

          val responseUpdate = MABBanditUpdate(id,
            globalState._1.clone(),
            globalState._2.clone(),
            globalState._3.clone())

          for (i <- 0 until arms) {
            globalState._1(i) += plays(i)
            globalState._2(i) += rewards(i)
            globalState._3(i) += rewardsSquared(i)
          }

          executorState.put(id, (plays, rewards, rewardsSquared))
          responseUpdate

        case ContextualBanditUpdate(id, features, rewards, rewardStats) =>
          val arms = rewards.length
          val numFeatures = features.head.rows

          val executorState = executorContextualStates.getOrElseUpdate(executorId, mutable.Map())
          val oldLocalState = executorState.getOrElse(id,
            (Array.fill(arms)(DenseMatrix.zeros[Double](numFeatures, numFeatures)),
              Array.fill(arms)(DenseVector.zeros[Double](numFeatures)),
            Array.fill(arms)(StatCounter())))

          val globalState = contextualStates.getOrElseUpdate(id,
            (Array.fill(arms)(DenseMatrix.zeros[Double](numFeatures, numFeatures)),
              Array.fill(arms)(DenseVector.zeros[Double](numFeatures)),
            Array.fill(arms)(StatCounter())))

          for (i <- 0 until arms) {
            globalState._1(i) = globalState._1(i) - oldLocalState._1(i)
            globalState._2(i) = globalState._2(i) - oldLocalState._2(i)
            globalState._3(i) = globalState._3(i).psuedoRemove(oldLocalState._3(i))
          }

          val responseUpdate = ContextualBanditUpdate(id,
            globalState._1.clone(),
            globalState._2.clone(),
            globalState._3.clone())

          for (i <- 0 until arms) {
            globalState._1(i) = globalState._1(i) + features(i)
            globalState._2(i) = globalState._2(i) + rewards(i)
            globalState._3(i).merge(rewardStats(i))
          }

          executorState.put(id, (features, rewards, rewardStats))
          responseUpdate

      }

      context.reply(SendDistributedUpdates(responses))
  }

}

private[spark] object BanditManagerMaster {
  val DRIVER_ENDPOINT_NAME = "BanditManagerMaster"
}
