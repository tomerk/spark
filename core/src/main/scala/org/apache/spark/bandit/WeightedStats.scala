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

/**
 * Modified from MultivariateOnlineSummarizer in MLlib to only process a single weighted variable
 *
 * :: DeveloperApi ::
 * MultivariateOnlineSummarizer implements to compute the mean,
 * variance, minimum, maximum, counts, and nonzero counts for instances in sparse or dense vector
 * format in an online fashion.
 *
 * Two MultivariateOnlineSummarizer can be merged together to have a statistical summary of
 * the corresponding joint dataset.
 *
 * A numerically stable algorithm is implemented to compute the mean and variance of instances:
 * Reference: <a href="http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance">
 * variance-wiki</a>
 * Zero elements (including explicit zero values) are skipped when calling add(),
 * to have time complexity O(nnz) instead of O(n) for each column.
 *
 * For weighted instances, the unbiased estimation of variance is defined by the reliability
 * weights:
 * see <a href="https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights">
 * Reliability weights (Wikipedia)</a>.
 */
class WeightedStats extends Serializable {

  private var currMean: Double = 0
  private var currM2n: Double = 0
  private var currM2: Double = 0
  private var currL1: Double = 0
  private var totalCnt: Long = 0
  private var totalWeightSum: Double = 0.0
  private var weightSquareSum: Double = 0.0
  private var weightSum: Double = 0
  private var nnz: Long = 0
  private var currMax: Double = Double.MinValue
  private var currMin: Double = Double.MaxValue

  /**
   * Add a new sample to this summarizer, and update the statistical summary.
   *
   * @param sample The sample in dense/sparse vector format to be added into this summarizer.
   * @return This MultivariateOnlineSummarizer object.
   */
  def add(sample: Double): this.type = add(sample, 1.0)

  private[spark] def addMultiple(value: Double, weight: Double, freq: Long): this.type = {
    require(weight >= 0.0, s"sample weight, ${weight} has to be >= 0.0")
    if (weight == 0.0) return this


    if (value != 0.0) {
      if (currMax < value) {
        currMax = value
      }
      if (currMin > value) {
        currMin = value
      }

      val prevMean = currMean
      val diff = value - prevMean
      currMean = prevMean + freq * weight * diff / (weightSum + freq * weight)
      currM2n += weight * (value - currMean) * diff * freq
      currM2 += weight * value * value * freq
      currL1 += weight * math.abs(value) * freq

      weightSum += weight * freq
      nnz += freq
    }

    totalWeightSum += weight * freq
    weightSquareSum += weight * weight * freq
    totalCnt += freq
    this
  }

  private[spark] def add(value: Double, weight: Double): this.type = {
    require(weight >= 0.0, s"sample weight, ${weight} has to be >= 0.0")
    if (weight == 0.0) return this


    if (value != 0.0) {
      if (currMax < value) {
        currMax = value
      }
      if (currMin > value) {
        currMin = value
      }

      val prevMean = currMean
      val diff = value - prevMean
      currMean = prevMean + weight * diff / (weightSum + weight)
      currM2n += weight * (value - currMean) * diff
      currM2 += weight * value * value
      currL1 += weight * math.abs(value)

      weightSum += weight
      nnz += 1
    }

    totalWeightSum += weight
    weightSquareSum += weight * weight
    totalCnt += 1
    this
  }

  /**
   * Merge another MultivariateOnlineSummarizer, and update the statistical summary.
   * (Note that it's in place merging; as a result, `this` object will be modified.)
   *
   * @param other The other MultivariateOnlineSummarizer to be merged.
   * @return This MultivariateOnlineSummarizer object.
   */
  def merge(other: WeightedStats): this.type = {
    if (this.totalWeightSum != 0.0 && other.totalWeightSum != 0.0) {

      totalCnt += other.totalCnt
      totalWeightSum += other.totalWeightSum
      weightSquareSum += other.weightSquareSum

      val thisNnz = weightSum
      val otherNnz = other.weightSum
      val totalNnz = thisNnz + otherNnz
      val totalCnnz = nnz + other.nnz
      if (totalNnz != 0.0) {
        val deltaMean = other.currMean - currMean
        // merge mean together
        currMean += deltaMean * otherNnz / totalNnz
        // merge m2n together
        currM2n += other.currM2n + deltaMean * deltaMean * thisNnz * otherNnz / totalNnz
        // merge m2 together
        currM2 += other.currM2
        // merge l1 together
        currL1 += other.currL1
        // merge max and min
        currMax = math.max(currMax, other.currMax)
        currMin = math.min(currMin, other.currMin)
      }
      weightSum = totalNnz
      nnz = totalCnnz
    } else if (totalWeightSum == 0.0 && other.totalWeightSum != 0.0) {
      this.currMean = other.currMean
      this.currM2n = other.currM2n
      this.currM2 = other.currM2
      this.currL1 = other.currL1
      this.totalCnt = other.totalCnt
      this.totalWeightSum = other.totalWeightSum
      this.weightSquareSum = other.weightSquareSum
      this.weightSum = other.weightSum
      this.nnz = other.nnz
      this.currMax = other.currMax
      this.currMin = other.currMin
    }
    this
  }

  /**
   * Sample mean of each dimension.
   *
   */
  def mean: Double = {
    require(totalWeightSum > 0, s"Nothing has been added to this summarizer.")

    currMean * (weightSum / totalWeightSum)
  }

  /**
   * Unbiased estimate of sample variance of each dimension.
   *
   */
  def variance: Double = {
    require(totalWeightSum > 0, s"Nothing has been added to this summarizer.")

    val denominator = totalWeightSum - (weightSquareSum / totalWeightSum)

    // Sample variance is computed, if the denominator is less than 0, the variance is just 0.
    if (denominator > 0.0) {
      val deltaMean = currMean
      (currM2n + deltaMean * deltaMean * weightSum *
          (totalWeightSum - weightSum) / totalWeightSum) / denominator
    } else {
      0
    }
  }

  /**
   * Sample size.
   *
   */
  def count: Long = totalCnt

  /**
   * Number of nonzero elements in each dimension.
   *
   */
  def numNonzeros: Double = {
    require(totalCnt > 0, s"Nothing has been added to this summarizer.")

    nnz
  }

  /**
   * Maximum value of each dimension.
   *
   */
  def max: Double = {
    require(totalWeightSum > 0, s"Nothing has been added to this summarizer.")

    if ((nnz < totalCnt) && (currMax < 0.0)) currMax = 0.0

    currMax
  }

  /**
   * Minimum value of each dimension.
   *
   */
  def min: Double = {
    require(totalWeightSum > 0, s"Nothing has been added to this summarizer.")

    if ((nnz < totalCnt) && (currMin > 0.0)) currMin = 0.0
    currMin
  }

  /**
   * L2 (Euclidian) norm of each dimension.
   *
   */
  def normL2: Double = {
    require(totalWeightSum > 0, s"Nothing has been added to this summarizer.")

    math.sqrt(currM2)
  }

  /**
   * L1 norm of each dimension.
   *
   */
  def normL1: Double = {
    require(totalWeightSum > 0, s"Nothing has been added to this summarizer.")

    currL1
  }

  def totalWeights: Double = {
    totalWeightSum
  }

  /**
   * Scale all weight by a constant factor
   * @param w
   */
  def scaleWeights(w: Double): Unit = {
    currM2n *= w
    currM2 *= w
    currL1 *= w

    weightSum *= w
    nnz += 1

    totalWeightSum *= w
    weightSquareSum *= w * w
  }

  def copy(): WeightedStats = {
    new WeightedStats().merge(this)
  }
}
