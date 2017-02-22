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

import breeze.linalg.DenseVector
import org.apache.spark.bandit.policies.{BanditPolicy, ContextualBanditPolicy}
import org.apache.spark.{SecurityManager, SparkConf}

import scala.reflect.ClassTag

/**
 * An interface for all the Bandit implementations in Spark (to allow
 * multiple bandit implementations).
 */
private[spark] class CentralizedDistributedBanditFactory extends BanditFactory {

  def initialize(isDriver: Boolean, conf: SparkConf, securityMgr: SecurityManager): Unit = ???

  /**
   * Creates a new bandit.
   *
   * @param arms The arms to choose between
   * @param policy The learning policy to use
   * @param isLocal whether we are in local mode (single JVM process)
   * @param id unique id representing this broadcast variable
   */
  def newBandit[A: ClassTag, B: ClassTag](arms: Seq[A => B],
                                          policy: BanditPolicy,
                                          isLocal: Boolean,
                                          id: Long): Bandit[A, B] = ???

  /**
   * Creates a new contextual bandit.
   *
   * @param arms The arms to choose between
   * @param policy The learning policy to use
   * @param isLocal whether we are in local mode (single JVM process)
   * @param id unique id representing this broadcast variable
   */
  def newContextualBandit[A: ClassTag, B: ClassTag](arms: Seq[A => B],
                                                    features: A => DenseVector[Double],
                                                    policy: ContextualBanditPolicy,
                                                    isLocal: Boolean,
                                                    id: Long): ContextualBandit[A, B] = ???

  def removeBandit(id: Long, removeFromDriver: Boolean, blocking: Boolean): Unit = ???

  def stop(): Unit = ???
}
