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

package org.apache.spark.sql.execution.joins

import breeze.linalg.DenseVector
import org.apache.spark.TaskContext
import org.apache.spark.bandit.{Bandit, BanditTrait, ContextualBandit}
import org.apache.spark.bandit.policies.{LinThompsonSamplingPolicyParams, UCB1NormalPolicyParams}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, GenericInternalRow, JoinedRow, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.{BinaryExecNode, RowIterator, SparkPlan}

import scala.collection.mutable.ArrayBuffer

/**
 * Performs a join of two child relations by first shuffling the data using the join keys,
 * Then selecting either a sort-merge join or a hash join for each partition
 */
case class PartitionContextShuffledHashOrSortMergeJoinExec(
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    joinType: JoinType,
    buildSide: BuildSide,
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan)
  extends BinaryExecNode with HashJoin {

  def features(in: (Array[InternalRow], Array[InternalRow], SQLMetric,
    Seq[Attribute], Seq[Attribute])): DenseVector[Double] = {
    DenseVector(1.0, in._1.length, in._2.length, in._1.length * in._2.length)
  }

  //val sc = left.sqlContext.sparkContext
  def joinByHash(in: (Array[InternalRow], Array[InternalRow], SQLMetric,
    Seq[Attribute], Seq[Attribute])): Iterator[InternalRow] = {
    val (buildPartition, streamPartition) = buildSide match {
      case BuildLeft => (in._1, in._2)
      case BuildRight => (in._2, in._1)
    }

    val hashed = buildHashedRelation(buildPartition.iterator)
    join(streamPartition.iterator, hashed, in._3)//.map(_.copy()).toBuffer.iterator
  }

  def joinBySort(in: (Array[InternalRow], Array[InternalRow], SQLMetric,
    Seq[Attribute], Seq[Attribute])): Iterator[InternalRow] = {

    val (leftOutput, rightOutput) = (in._4, in._5)
    val boundCondition: (InternalRow) => Boolean = {
      condition.map { cond =>
        newPredicate(cond, leftOutput ++ rightOutput).eval _
      }.getOrElse {
        (r: InternalRow) => true
      }
    }

    // An ordering that can be used to compare keys from both sides.
    val keyOrdering = newNaturalAscendingOrdering(leftKeys.map(_.dataType))
    val resultProj: InternalRow => InternalRow = UnsafeProjection.create(output, output)

    /*import scala.collection.JavaConverters._
    val lList = new util.ArrayList[InternalRow]()
    in._1.foreach {
      row => lList.add(row.copy())
    }

    val rList = new util.ArrayList[InternalRow]()
    in._2.foreach {
      row => rList.add(row.copy())
    }*/
    val leftList = in._1
    val rightList = in._2
    //val start = System.currentTimeMillis()

    //logError(s"read: ${TaskContext.get().taskMetrics().shuffleReadMetrics.recordsRead}")
    java.util.Arrays.sort(leftList, keyOrdering)
    java.util.Arrays.sort(rightList, keyOrdering)

    /*val leftSorter = new ExternalSorter[InternalRow, Null, InternalRow](
      TaskContext.get(), ordering = Some(keyOrdering))
    leftSorter.insertAll(leftList.iterator.map(r => (r, null)))*/
    val leftIter = leftList.iterator

    /*val rightSorter = new ExternalSorter[InternalRow, Null, InternalRow](
      TaskContext.get(), ordering = Some(keyOrdering))
    rightSorter.insertAll(rightList.iterator.map(r => (r, null)))*/
    val rightIter = rightList.iterator
    val numOutputRows = in._3

    joinType match {
      case _: InnerLike =>
        new RowIterator {
          private[this] var currentLeftRow: InternalRow = _
          private[this] var currentRightMatches: ArrayBuffer[InternalRow] = _
          private[this] var currentMatchIdx: Int = -1
          private[this] val smjScanner = new SortMergeJoinScanner(
            UnsafeProjection.create(leftKeys, leftOutput),
            UnsafeProjection.create(rightKeys, rightOutput),
            keyOrdering,
            RowIterator.fromScala(leftIter),
            RowIterator.fromScala(rightIter)
          )
          private[this] val joinRow = new JoinedRow

          if (smjScanner.findNextInnerJoinRows()) {
            currentRightMatches = smjScanner.getBufferedMatches
            currentLeftRow = smjScanner.getStreamedRow
            currentMatchIdx = 0
          }

          override def advanceNext(): Boolean = {
            while (currentMatchIdx >= 0) {
              if (currentMatchIdx == currentRightMatches.length) {
                if (smjScanner.findNextInnerJoinRows()) {
                  currentRightMatches = smjScanner.getBufferedMatches
                  currentLeftRow = smjScanner.getStreamedRow
                  currentMatchIdx = 0
                } else {
                  currentRightMatches = null
                  currentLeftRow = null
                  currentMatchIdx = -1
                  return false
                }
              }
              joinRow(currentLeftRow, currentRightMatches(currentMatchIdx))
              currentMatchIdx += 1
              if (boundCondition(joinRow)) {
                numOutputRows += 1
                return true
              }
            }
            false
          }

          override def getRow: InternalRow = resultProj(joinRow)
        }.toScala

      case LeftOuter =>
        val smjScanner = new SortMergeJoinScanner(
          streamedKeyGenerator = UnsafeProjection.create(leftKeys, leftOutput),
          bufferedKeyGenerator = UnsafeProjection.create(rightKeys, rightOutput),
          keyOrdering,
          streamedIter = RowIterator.fromScala(leftIter),
          bufferedIter = RowIterator.fromScala(rightIter)
        )
        val rightNullRow = new GenericInternalRow(right.output.length)
        new LeftOuterIterator(
          smjScanner, rightNullRow, boundCondition, resultProj, numOutputRows).toScala

      case RightOuter =>
        val smjScanner = new SortMergeJoinScanner(
          streamedKeyGenerator = UnsafeProjection.create(rightKeys, rightOutput),
          bufferedKeyGenerator = UnsafeProjection.create(leftKeys, leftOutput),
          keyOrdering,
          streamedIter = RowIterator.fromScala(rightIter),
          bufferedIter = RowIterator.fromScala(leftIter)
        )
        val leftNullRow = new GenericInternalRow(leftOutput.length)
        new RightOuterIterator(
          smjScanner, leftNullRow, boundCondition, resultProj, numOutputRows).toScala

      case FullOuter =>
        val leftNullRow = new GenericInternalRow(leftOutput.length)
        val rightNullRow = new GenericInternalRow(rightOutput.length)
        val smjScanner = new SortMergeFullOuterJoinScanner(
          leftKeyGenerator = UnsafeProjection.create(leftKeys, leftOutput),
          rightKeyGenerator = UnsafeProjection.create(rightKeys, rightOutput),
          keyOrdering,
          leftIter = RowIterator.fromScala(leftIter),
          rightIter = RowIterator.fromScala(rightIter),
          boundCondition,
          leftNullRow,
          rightNullRow)

        new FullOuterIterator(
          smjScanner,
          resultProj,
          numOutputRows).toScala

      case LeftSemi =>
        new RowIterator {
          private[this] var currentLeftRow: InternalRow = _
          private[this] val smjScanner = new SortMergeJoinScanner(
            UnsafeProjection.create(leftKeys, leftOutput),
            UnsafeProjection.create(rightKeys, rightOutput),
            keyOrdering,
            RowIterator.fromScala(leftIter),
            RowIterator.fromScala(rightIter)
          )
          private[this] val joinRow = new JoinedRow

          override def advanceNext(): Boolean = {
            while (smjScanner.findNextInnerJoinRows()) {
              val currentRightMatches = smjScanner.getBufferedMatches
              currentLeftRow = smjScanner.getStreamedRow
              var i = 0
              while (i < currentRightMatches.length) {
                joinRow(currentLeftRow, currentRightMatches(i))
                if (boundCondition(joinRow)) {
                  numOutputRows += 1
                  return true
                }
                i += 1
              }
            }
            false
          }

          override def getRow: InternalRow = currentLeftRow
        }.toScala

      case LeftAnti =>
        new RowIterator {
          private[this] var currentLeftRow: InternalRow = _
          private[this] val smjScanner = new SortMergeJoinScanner(
            UnsafeProjection.create(leftKeys, leftOutput),
            UnsafeProjection.create(rightKeys, rightOutput),
            keyOrdering,
            RowIterator.fromScala(leftIter),
            RowIterator.fromScala(rightIter)
          )
          private[this] val joinRow = new JoinedRow

          override def advanceNext(): Boolean = {
            while (smjScanner.findNextOuterJoinRows()) {
              currentLeftRow = smjScanner.getStreamedRow
              val currentRightMatches = smjScanner.getBufferedMatches
              if (currentRightMatches == null) {
                return true
              }
              var i = 0
              var found = false
              while (!found && i < currentRightMatches.length) {
                joinRow(currentLeftRow, currentRightMatches(i))
                if (boundCondition(joinRow)) {
                  found = true
                }
                i += 1
              }
              if (!found) {
                numOutputRows += 1
                return true
              }
            }
            false
          }

          override def getRow: InternalRow = currentLeftRow
        }.toScala

      case j: ExistenceJoin =>
        new RowIterator {
          private[this] var currentLeftRow: InternalRow = _
          private[this] val result: InternalRow = new GenericInternalRow(Array[Any](null))
          private[this] val smjScanner = new SortMergeJoinScanner(
            UnsafeProjection.create(leftKeys, leftOutput),
            UnsafeProjection.create(rightKeys, rightOutput),
            keyOrdering,
            RowIterator.fromScala(leftIter),
            RowIterator.fromScala(rightIter)
          )
          private[this] val joinRow = new JoinedRow

          override def advanceNext(): Boolean = {
            while (smjScanner.findNextOuterJoinRows()) {
              currentLeftRow = smjScanner.getStreamedRow
              val currentRightMatches = smjScanner.getBufferedMatches
              var found = false
              if (currentRightMatches != null) {
                var i = 0
                while (!found && i < currentRightMatches.length) {
                  joinRow(currentLeftRow, currentRightMatches(i))
                  if (boundCondition(joinRow)) {
                    found = true
                  }
                  i += 1
                }
              }
              result.setBoolean(0, found)
              numOutputRows += 1
              return true
            }
            false
          }

          override def getRow: InternalRow = resultProj(joinRow(currentLeftRow, result))
        }.toScala//.map(_.copy()).toBuffer.iterator

      case x =>
        throw new IllegalArgumentException(
          s"SortMergeJoin should not take $x as the JoinType")
    }
  }

  //val bandit = sc.bandit(Seq(x => joinBySort(x)), EpsilonGreedyPolicyParams())

  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "buildDataSize" -> SQLMetrics.createSizeMetric(sparkContext, "data size of build side"),
    "buildTime" -> SQLMetrics.createTimingMetric(sparkContext, "time to build hash map"))

  override def requiredChildDistribution: Seq[Distribution] =
    ClusteredDistribution(leftKeys) :: ClusteredDistribution(rightKeys) :: Nil

  private def buildHashedRelation(iter: Iterator[InternalRow]): HashedRelation = {
    val buildDataSize = longMetric("buildDataSize")
    val buildTime = longMetric("buildTime")
    val start = System.nanoTime()
    val context = TaskContext.get()
    val relation = HashedRelation(iter, buildKeys, taskMemoryManager = context.taskMemoryManager())
    buildTime += (System.nanoTime() - start) / 1000000
    buildDataSize += relation.estimatedSize
    // This relation is usually used until the end of task.
    context.addTaskCompletionListener(_ => relation.close())
    relation
  }

  protected override def doExecute(): RDD[InternalRow] = {
    val bandit = GlobalContextJoinBandit.getBandit(sqlContext)

    val numOutputRows = longMetric("numOutputRows")
    val (leftOutput, rightOutput) = (left.output, right.output)
    left.execute().zipPartitions(right.execute()) { (leftIter, rightIter) =>
      // These .copy()'s are needed because the rows are streamed unsaferows.
      //val streamIterSeq = streamIter.map(_.copy()).toStream
      //val buildIterSeq = buildIter.map(_.copy()).toStream
      val context = TaskContext.get()

      val leftList = leftIter.map(_.copy()).toArray
      val rightList = rightIter.map(_.copy()).toArray
      //val start = System.currentTimeMillis()

      //logError(s"Left: ${leftList.size}, Right: ${rightList.size}")

      val startTime = System.nanoTime()
      val (result, delayedFeedback) = {
        val (action, delayedFeedback) = bandit.applyAndDelayFeedback(
          (leftList, rightList, numOutputRows, leftOutput, rightOutput))

        val result = action match {
          case UseSort => joinBySort((leftList, rightList, numOutputRows,
            leftOutput, rightOutput))
          case UseHash => joinByHash((leftList, rightList, numOutputRows,
            leftOutput, rightOutput))
        }

        (result, delayedFeedback)
      }

      context.addTaskCompletionListener(_ => {
        val endTime = System.nanoTime()
        val reward = startTime - endTime // This is intentionally negative. Long time is bad.
        delayedFeedback.provide(reward)

        /*logError(s"Id: ${delayedFeedback.banditId}, Arm: ${delayedFeedback.getArm}, " +
          s"Reward: $reward, " +
          s"Left: ${leftList.size}, Right: ${rightList.size}")*/
      })
      result
    }
  }

}


sealed trait HashOrJoin
case object UseHash extends HashOrJoin
case object UseSort extends HashOrJoin

object GlobalContextJoinBandit extends Serializable with Logging {
  def returnHash(x: Any): HashOrJoin = {
    //logError("Returning hash!")
    UseHash
  }
  def returnSort(x: Any): HashOrJoin = {
    //logError("Returning sort!")
    UseSort
  }

  @transient private var bandit: BanditTrait[(Array[InternalRow],
    Array[InternalRow], SQLMetric,
    Seq[Attribute], Seq[Attribute]), HashOrJoin] = null

  def features(in: (Array[InternalRow], Array[InternalRow], SQLMetric,
    Seq[Attribute], Seq[Attribute])): DenseVector[Double] = {
    DenseVector(1.0, in._1.length, in._2.length, in._1.length * in._2.length)
  }

  def getBandit(sqlContext: SQLContext): BanditTrait[(Array[InternalRow],
    Array[InternalRow], SQLMetric,
    Seq[Attribute], Seq[Attribute]), HashOrJoin] = {
    if (bandit == null) {
      val sc = sqlContext.sparkContext
      val banditMode = sqlContext.sparkSession.conf.get(
        "spark.sql.join.bandit.shuffleSortHash", "both")
      logError(banditMode)
      bandit = {
        banditMode match {
          case "both" => sc.contextualBandit(Seq (returnSort(_), returnHash(_)
          ), features, LinThompsonSamplingPolicyParams(4, 1.0, useCholesky = true) )
          case "sort" => sc.bandit (Seq (returnSort (_)),
            UCB1NormalPolicyParams(0.4) )
          case "hash" => sc.bandit (Seq (returnHash(_)),
            UCB1NormalPolicyParams(0.4) )
        }
      }
    }

    bandit
  }

}
