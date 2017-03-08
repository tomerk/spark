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
import org.apache.spark.bandit.policies._

class BanditSuite extends SparkFunSuite with LocalSparkContext {

  test("Trying bandits?") {
    val numSlaves = 2
    val numCoresPerSlave = 2
    val memoryPerSlave = 2048
    val conf = new SparkConf
    //conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.driver.extraClassPath", sys.props("java.class.path"))
        .set("spark.executor.extraClassPath", sys.props("java.class.path"))

    sc = new SparkContext(s"local-cluster[$numSlaves, $numCoresPerSlave, $memoryPerSlave]",
      "test", conf)

    val invOne: Int => Int = numFeatures => {
      val mat = DenseMatrix.rand[Double](100, 100)
      mat * mat * mat * mat
      17
    }

    val invTwo: Int => Int = numFeatures => {
      val mat = DenseMatrix.rand[Double](100, 100)
      mat * mat * mat * mat
      17
    }

    val list = Seq.fill(10000)(50)
    val x = sc.parallelize(list, numSlices = 50)

    //x.map(y => invOne(y)).collect()
    //x.map(y => invTwo(y)).collect()

    val feat: Int => DenseVector[Double] = _ => DenseVector.fill[Double](1)(1.0)
    (0 until 1).foreach { _ =>
      val bandit = sc.contextualBandit(Seq(invOne, invTwo), feat,
        LinUCBPolicyParams(1))

      //x.map(y => invOne(y)).collect()
      x.map(y => bandit.apply(y)).collect()
    }
    //assert(results.collect().toSet === (1 to numSlaves).map(x => (x, 10)).toSet)
  }


  test("Test Contextual Bandit Timings") {
    val numFeatures = 3
    val arms = 10
    val policy = new LinUCBPolicy(numArms = arms, numFeatures = numFeatures, 2.36)

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
      i += 1
    }
    val end = System.currentTimeMillis()
    logInfo(s"${(end - start)/10000.0} millis per round")
  }

  test("Test Bandit Timings") {
    val arms = 10
    val policy = new GaussianThompsonSamplingPolicy(numArms = arms, varianceMultiplier = 1.0)

    val bestArm = 3

    val rewardDist = Rand.gaussian

    var i = 0
    val start = System.currentTimeMillis()
    while (i < 10000) {
      val arm = policy.chooseArm(1)

      logInfo(s"$i: $arm")
      val reward = (rewardDist.draw() - (if (arm == bestArm) 10 else 10.5))*0.001
      policy.provideFeedback(arm, 1, reward, reward*reward)
      i += 1
    }
    val end = System.currentTimeMillis()
    logInfo(s"${(end - start)/10000.0} millis per round")
  }
}