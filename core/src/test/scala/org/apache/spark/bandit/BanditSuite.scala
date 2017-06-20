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

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.stats.distributions.{Gamma, MultivariateGaussian, Rand}
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
    //val rewardDists: Seq[Rand[Double]] = Seq(new Gamma(0.1, 200.0), new Gamma(0.2, 200.0),
    // new Gamma(0.25, 200.0), new Gamma(0.29, 200.0), new Gamma(0.3, 200.0))

    val rewardDists: Seq[Rand[Double]] = (0 until 10).flatMap(x =>
      Seq(Rand.gaussian(-10, 6), Rand.gaussian(-20, 6)))/*, Rand.gaussian(-10, 0),
      Rand.gaussian(-20, 6), Rand.gaussian(-10, 0), Rand.gaussian(-20, 6), Rand.gaussian(-10, 0),
      Rand.gaussian(-20, 6), Rand.gaussian(-10, 0), Rand.gaussian(-20, 6))*/

    // val rewardDists = Seq(Rand.gaussian(10, 50), Rand.gaussian(20, 1000))
    //val rewardDists = Seq(Rand.gaussian(10, 1000), Rand.gaussian(20, 50))

    val numArms = rewardDists.length

    var rewards = 0.0
    val start = System.currentTimeMillis()
    val n = 1000//10000
    val numTrials = 100
    val banditResults = (0 until numTrials).flatMap { trial =>
      val policy = new UCB1Policy(numArms, 1.0)
      //val policy = new UCB1Policy(numArms, 1.0)
      //val policy = new UCB1NormalPolicy(numArms = numArms, 0.5)
      //val policy = new GaussianThompsonSamplingPolicy(numArms = numArms, 1.0)
      //val policy = new GaussianBayesUCBPolicy(numArms, 1.0)

      (0 until n).map { i =>
        val arm = policy.chooseArm(1)
        val rewardDist = rewardDists(arm)

        val reward = rewardDist.draw()
        rewards += reward
        policy.provideFeedback(arm, 1, reward)

        s"$trial,$i,$arm,$reward"
      }
    }
    val end = System.currentTimeMillis()

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
      s"/Users/tomerk11/Desktop/tests/${System.currentTimeMillis()}.csv")))
    writer.write(s"trial,step,arm,reward\n")

    for (x <- banditResults) {
      writer.write(x + "\n")
    }
    writer.close()

    logInfo(s"${(end - start)/(n.toDouble * numTrials)} millis per round")
    logInfo(s"Mean reward: ${rewards/(n.toDouble * numTrials)}")
  }

  test("Test Bandit Timings batches") {
    val rewardDists: Seq[Rand[Double]] = Seq(new Gamma(0.1, 200.0), new Gamma(0.2, 200.0),
      new Gamma(0.25, 200.0), new Gamma(0.29, 200.0), new Gamma(0.3, 200.0))

    //val rewardDists: Seq[Rand[Double]] = (0 until 10).flatMap(x =>
    // Seq(Rand.gaussian(-10, 6), Rand.gaussian(-20, 6)))

    // val rewardDists = Seq(Rand.gaussian(10, 50), Rand.gaussian(20, 1000))
    //val rewardDists = Seq(Rand.gaussian(10, 1000), Rand.gaussian(20, 50))

    val numArms = rewardDists.length

    var rewards = 0.0
    val start = System.currentTimeMillis()
    val n = 1000
    val batchSize = 1
    val numTrials = 100
    val banditResults = (1 to 10).flatMap {
      batchSize =>
        val numBatches = n / batchSize
      (0 until numTrials).flatMap { trial =>
        //val policy = new UCB1Policy(numArms, 1.0)
        //val policy = new UCB1Policy(numArms, 1.0)
        //val policy = new UCB1NormalPolicy(numArms = numArms, 0.5)
        val policy = new GaussianThompsonSamplingPolicy(numArms = numArms, 1.0)
        //val policy = new EpsilonDecreasingPolicy(numArms = numArms, 5.0 / batchSize)
        //val policy = new GaussianBayesUCBPolicy(numArms, 1.0)

        (0 until numBatches).flatMap { i =>
          val arm = policy.chooseArm(1)//batchSize)
        val rewardDist = rewardDists(arm)

          val rewardList = (0 until batchSize).map(_ => rewardDist.draw())
          rewards += rewardList.sum/rewardList.length
          policy.provideFeedback(arm, 1, rewardList.sum/rewardList.length)

          //logInfo(s"$trial,${i * batchSize},$arm")
          rewardList.zipWithIndex.map { case (reward, batchIndex) =>
            s"$trial,${i * batchSize + batchIndex},$batchSize,$arm,$reward"
          }
        }
      }
    }
    val end = System.currentTimeMillis()

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
      s"/Users/tomerk11/Desktop/tests/${System.currentTimeMillis()}.csv")))
    writer.write(s"trial,step,batchSize,arm,reward\n")

    for (x <- banditResults) {
      writer.write(x + "\n")
    }
    writer.close()

    //logInfo(s"${(end - start)/(numBatches.toDouble * numTrials * batchSize)} millis per round")
    //logInfo(s"Mean reward: ${rewards/(numBatches.toDouble * numTrials * batchSize)}")
  }

  test("Test Bandit Contextual batches") {
    val rewardDists: Seq[Rand[Double]] = Seq(Rand.gaussian(0, 1.0), Rand.gaussian(1, 0.5),
      Rand.gaussian(2, 0.5))
    /*Seq(new Gamma(0.1, 200.0), new Gamma(0.2, 200.0),
      new Gamma(0.25, 200.0), new Gamma(0.29, 200.0), new Gamma(0.3, 200.0))*/

    val numFeatures = 10
    val weights: Seq[DenseVector[Double]] = (0 until rewardDists.length).map(
      x => DenseVector((0.0 +: Rand.gaussian.sample(numFeatures - 1)): _*))

    def genFeatures(): DenseVector[Double] = {
      DenseVector((1.0 +: Rand.gaussian.sample(numFeatures - 1)): _*)
    }

    //val rewardDists: Seq[Rand[Double]] = (0 until 10).flatMap(x =>
    // Seq(Rand.gaussian(-10, 6), Rand.gaussian(-20, 6)))

    // val rewardDists = Seq(Rand.gaussian(10, 50), Rand.gaussian(20, 1000))
    //val rewardDists = Seq(Rand.gaussian(10, 1000), Rand.gaussian(20, 50))

    val numArms = rewardDists.length

    val n = 1000
    val batchSize = 1
    val numTrials = 100
    val banditResults = (1 to 30).flatMap {
      batchSize =>
        val numBatches = n / batchSize
        val start = System.currentTimeMillis()

        var rewards = 0.0

        val res = (0 until numTrials).flatMap { trial =>
          //val policy = new UCB1Policy(numArms, 1.0)
          //val policy = new UCB1Policy(numArms, 1.0)
          //val policy = new UCB1NormalPolicy(numArms = numArms, 0.5)
          val policy = new LinThompsonSamplingPolicy(numFeatures = numFeatures,
            numArms = numArms, v = 1.0, useCholesky = true, usingBias = false)
          //val policy = new GaussianThompsonSamplingPolicy(numArms = numArms, 1.0)

          //val policy = new EpsilonDecreasingPolicy(numArms = numArms, 5.0 / batchSize)
          //val policy = new GaussianBayesUCBPolicy(numArms, 1.0)

          (0 until numBatches).flatMap { i =>
            val features = (0 until batchSize).map(_ => genFeatures())
            val avgFeature = features.reduce(_ + _) / batchSize.toDouble

            val arm = policy.chooseArm(avgFeature)
            val rewardDist = rewardDists(arm)
            val armWeights = weights(arm)

            val rewardList = (0 until batchSize).map { batchIndex =>
              features(batchIndex).dot(armWeights) + rewardDist.draw()
            }
            rewards += rewardList.sum
            //policy.provideFeedback(arm, 1, rewardList.sum/rewardList.length)

            policy.provideFeedback(arm, avgFeature, new WeightedStats().add(
              rewardList.sum/rewardList.length))

            //logInfo(s"$trial,${i * batchSize},$arm")
            rewardList.zipWithIndex.map { case (reward, batchIndex) =>
              s"$trial,${i * batchSize + batchIndex},$batchSize,$arm,$reward"
            }
          }
        }
        val end = System.currentTimeMillis()

        logInfo(s"${(end - start)/(numBatches.toDouble * numTrials * batchSize)} ms per round")
        logInfo(s"Mean reward: ${rewards/(numBatches.toDouble * numTrials * batchSize)}")
        res
    }

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
      s"/Users/tomerk11/Desktop/tests/${System.currentTimeMillis()}.csv")))
    writer.write(s"trial,step,batchSize,arm,reward\n")

    for (x <- banditResults) {
      writer.write(x + "\n")
    }
    writer.close()

  }

}