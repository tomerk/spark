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

import org.apache.spark.bandit.policies.{BanditPolicy, ContextualBanditPolicy}

import scala.reflect.ClassTag

/**
 * The bandit class is used for dynamically tuned methods appearing in spark tasks.
 */
abstract class Bandit[A: ClassTag, B: ClassTag] {
  val id: Long
  def apply(in: A): B

  /**
   * A vectorized bandit strategy. Given a sequence of input, choose a single arm
   * to apply to all of the input. The learning will treat this as the reward
   * averaged over all the input items, given this decision.
   *
   * @param in The vector of input
   */
  def vectorizedApply(in: Seq[A]): Seq[B]

  def saveTunedSettings(file: String): Unit
  def loadTunedSettings(file: String): Unit
}



// Also need state access & merging code that is complete, not just per-arm??!!
// (ALTHOUGH if we can send state updates for each arm separately it could lower
// the total required communication, but that's a premature optimization)


// The state & state updates are basically the same for thompson's sampling and LinUCB
// (will need to note that the thompson sampling code was modified to be on disjoint models)
// The only difference is the reward computation (and it may be worth pre-computing the inverse?)

// Future work: lower-overhead, sgd based updates!?
