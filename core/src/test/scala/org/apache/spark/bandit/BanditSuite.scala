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

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.stats.distributions.{MultivariateGaussian, Rand}
import org.apache.spark._

class BanditSuite extends SparkFunSuite with LocalSparkContext {

  test("Using TorrentBroadcast locally") {
    sc = new SparkContext("local", "test")
    val list = List[Int](1, 2, 3, 4)
    val broadcast = sc.broadcast(list)
    val results = sc.parallelize(1 to 2).map(x => (x, broadcast.value.sum))
    assert(results.collect().toSet === Set((1, 10), (2, 10)))
  }

  test("Accessing TorrentBroadcast variables from multiple threads") {
    sc = new SparkContext("local[10]", "test")
    val list = List[Int](1, 2, 3, 4)
    val broadcast = sc.broadcast(list)
    val results = sc.parallelize(1 to 10).map(x => (x, broadcast.value.sum))
    assert(results.collect().toSet === (1 to 10).map(x => (x, 10)).toSet)
  }

  test("Accessing TorrentBroadcast Bandit variables in a local cluster") {
    val numSlaves = 4
    val conf = new SparkConf
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.broadcast.compress", "true")
      .set("spark.driver.extraClassPath", sys.props("java.class.path"))
      .set("spark.executor.extraClassPath", sys.props("java.class.path"))
    logInfo("Waiting fer more :O")
    logError("Waiting fer more :O")
    sc = new SparkContext("local-cluster[%d, 1, 1024]".format(numSlaves), "test", conf)
    logInfo("Waiting fer more :O")
    logError("Waiting fer more :O")
    val list = List[Int](1, 2, 3, 4)
    val broadcast = sc.broadcast(list)
    val results = sc.parallelize(1 to numSlaves).map(x => (x, broadcast.value.sum))
    assert(results.collect().toSet === (1 to numSlaves).map(x => (x, 10)).toSet)
  }

  test("Test Bandit Timings") {
    val numFeatures = 3
    val arms = 10
    val policy = new LinUCBPolicy(numArms = arms, numFeatures = numFeatures)

    val bestArm = 3

    val rewardDist = Rand.gaussian

    var i = 0
    val start = System.currentTimeMillis()
    while (i < 10000) {
      val featureVec = DenseVector.rand[Double](numFeatures)
      val arm = policy.chooseArm(featureVec)

      logInfo(s"$i: $arm")
      val reward = rewardDist.draw() - (if (arm == bestArm) 10 else 13)
      policy.provideFeedback(arm, featureVec, reward)
      // mat * mat
      i += 1
    }
    val end = System.currentTimeMillis()
    logInfo(s"${(end - start)/10000.0} millis per round")
  }

}