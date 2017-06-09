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
import org.apache.commons.math3.optim.linear._
import org.apache.spark._
import org.apache.spark.bandit.policies._
import org.apache.spark.util.StatCounter

class BanditSuite extends SparkFunSuite with LocalSparkContext {

  test("Trying bandits?") {
    val numSlaves = 2
    val numCoresPerSlave = 2
    val memoryPerSlave = 2048
    val conf = new SparkConf
    //conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.driver.extraClassPath", sys.props("java.class.path"))
      .set("spark.executor.extraClassPath", sys.props("java.class.path"))
      .set("spark.bandits.communicationRate", "1s")
      .set("spark.bandits.driftDetectionRate", "1s")

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
    Seq(2, 2, 4, 8, 16, 32, 64).foreach { arms =>

      Seq(2, 4, 8, 16, 32, 64).foreach { numFeatures =>
        val policy = new LinUCBPolicy(numArms = arms, numFeatures = numFeatures, 1)

        val bestArm = 7

        val rewardDist = Rand.gaussian

        var i = 0
        val start = System.currentTimeMillis()
        while (i < 10000) {
          val featureVec = DenseVector.rand[Double](numFeatures)
          val arm = policy.chooseArm(featureVec)

          //logInfo(s"$i: $arm")
          val reward = (rewardDist.draw() - (if (arm == bestArm) 10 else 10.5)) * 1e-10
          policy.provideFeedback(arm, featureVec, new WeightedStats().add(reward))
          i += 1
        }
        val end = System.currentTimeMillis()
        logInfo(s"${
          (end - start) / 10000.0
        } millis per round with $arms arms and $numFeatures features")
      }
    }
  }

  test("Test Simplex timings") {
    var i = 0
    val numArms = 20
    val start = System.currentTimeMillis()
    val solver = new SimplexSolver()
    val ones = DenseVector.ones[Double](numArms).toArray
    val threshold = 0.3
    val n = 1000000.0
    while (i < n) {
      val objectiveVec = DenseVector.rand[Double](numArms)
      val objective = new LinearObjectiveFunction(objectiveVec.toArray, 0)
      val nonNegative = new NonNegativeConstraint(true)
      val probabilityConstraint = new LinearConstraint(ones, Relationship.EQ, 1.0)
      val thresholdVec = DenseVector.rand[Double](numArms)
      val thresholdConstraint = new LinearConstraint(
        thresholdVec.toArray, Relationship.GEQ, threshold)

      val constraintSet = new LinearConstraintSet(
        probabilityConstraint, thresholdConstraint)

      try {
        val results = solver.optimize(objective, nonNegative, constraintSet)
        //logInfo(s"${results.getPoint.toList} ${results.getValue}")
      } catch {
        case e: NoFeasibleSolutionException =>
          logInfo("No feasible solution :(")
      }
      i += 1
    }
    val end = System.currentTimeMillis()
    logInfo(s"${(end - start)/n} millis per round")
  }

  test("Test Bandit Timings") {
    Seq(2, 2, 4, 8, 16, 32, 64).foreach { arms =>
      val policy = new UCB1NormalPolicy(numArms = arms, 0.25)

      val bestArm = 3

      val rewardDist = Rand.gaussian

      var i = 0
      val start = System.currentTimeMillis()
      while (i < 10000) {
        val arm = policy.chooseArm(1)

        //logInfo(s"$i: $arm")
        val reward = (rewardDist.draw() - (if (arm == bestArm) 10 else 10.2)) * 0.001
        policy.provideFeedback(arm, 1, reward)
        i += 1
      }
      val end = System.currentTimeMillis()
      logInfo(s"${(end - start) / 10000.0} millis per round with $arms arms")
    }
  }

  test("Test Bandit Timings bayesian") {
    val rewardDists = Seq(Rand.gaussian(10, 1000))
    val arms = rewardDists.length

    val policy = new GaussianBayesUCBPolicy(numArms = arms, 1.0)

    val bestArm = 3


    var rewards = 0.0
    var i = 0
    val start = System.currentTimeMillis()
    val n = 10000
    while (i < n) {
      val arm = policy.chooseArm(1)
      val rewardDist = rewardDists(arm)

      logInfo(s"$i: $arm, $rewards")
      val reward = (rewardDist.draw() + (if (arm == bestArm) 10.2 else -100.0))
      rewards += reward
      policy.provideFeedback(arm, 1, reward)
      i += 1
    }
    val end = System.currentTimeMillis()
    logInfo(s"${(end - start)/10000.0} millis per round")
    logInfo(s"Mean reward: ${rewards/n}")
  }
}