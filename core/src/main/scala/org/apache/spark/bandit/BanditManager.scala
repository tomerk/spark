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

import breeze.linalg.DenseVector
import org.apache.spark.bandit.policies.{BanditPolicy, ContextualBanditPolicy}
import org.apache.spark.internal.Logging
import org.apache.spark.{SecurityManager, SparkConf}

import scala.reflect.ClassTag

private[spark] class BanditManager(
    val isDriver: Boolean,
    conf: SparkConf,
    securityManager: SecurityManager)
  extends Logging {

  private var initialized = false
  private var banditFactory: BanditFactory = null

  initialize()

  // Called by SparkContext or Executor before using Broadcast
  private def initialize() {
    synchronized {
      if (!initialized) {
        banditFactory = new CentralizedDistributedBanditFactory
        banditFactory.initialize(isDriver, conf, securityManager)
        initialized = true
      }
    }
  }

  def stop() {
    banditFactory.stop()
  }

  private val nextBanditId = new AtomicLong(0)

  def newBandit[A: ClassTag, B: ClassTag](arms: Seq[A => B],
                                          policy: BanditPolicy,
                                          isLocal: Boolean): Bandit[A, B] = {
    banditFactory.newBandit(
      arms,
      policy,
      isLocal,
      nextBanditId.getAndIncrement())
  }

  def newContextualBandit[A: ClassTag, B: ClassTag](arms: Seq[A => B],
                                                    features: A => DenseVector[Double],
                                                    policy: ContextualBanditPolicy,
                                                    isLocal: Boolean): ContextualBandit[A, B] = {
    banditFactory.newContextualBandit(arms, features, policy, isLocal,
      nextBanditId.getAndIncrement())
  }

  def removeBandit(id: Long, removeFromDriver: Boolean, blocking: Boolean) {
    banditFactory.removeBandit(id, removeFromDriver, blocking)
  }
}
