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

package org.apache.spark.sql.execution.benchmark

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.TaskInfo
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedRelation
import org.apache.spark.sql.catalyst.expressions.SubqueryExpression
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.util._
import org.apache.spark.ui.scope.RDDOperationGraphContainsStringUtil
import org.apache.spark.util.Benchmark

/**
 * Benchmark to measure TPCDS query performance.
 * To run this:
 *  spark-submit --class <this class> --jars <spark sql test jar>
 */
object TPCDSQueryBenchmark extends Logging {
  val conf =
    new SparkConf()
      .setMaster("local[4]")
      .setAppName("test-sql-context")
      .set("spark.sql.parquet.compression.codec", "snappy")
      .set("spark.sql.shuffle.partitions", "512")
      .set("spark.driver.memory", "3g")
      .set("spark.executor.memory", "3g")
      .set("spark.bandits.driftDetectionRate", "99999999s")
      .set("spark.bandits.alwaysShare", "true")
      .set("spark.bandits.communicationRate", "500ms")
      .set("spark.sql.join.banditJoin", "true")
      .set("spark.sql.join.banditJoin.contextual", "false")
      .set("spark.sql.join.bandit.shuffleSortHash", "both")
    .set("spark.sql.autoBroadcastJoinThreshold", "10")

  val spark = SparkSession.builder.config(conf).getOrCreate()

  val tables = Seq("catalog_page", "catalog_returns", "customer", "customer_address",
    "customer_demographics", "date_dim", "household_demographics", "inventory", "item",
    "promotion", "store", "store_returns", "catalog_sales", "web_sales", "store_sales",
    "web_returns", "web_site", "reason", "call_center", "warehouse", "ship_mode", "income_band",
    "time_dim", "web_page")

  def setupTables(dataLocation: String): Map[String, Long] = {
    tables.map { tableName =>
      spark.read.parquet(s"$dataLocation/$tableName").createOrReplaceTempView(tableName)
      tableName -> spark.table(tableName).count()
    }.toMap
  }

  def tpcdsAll(dataLocation: String, queries: Seq[String]): Unit = {
    require(dataLocation.nonEmpty,
      "please modify the value of dataLocation to point to your local TPCDS data")
    val tableSizes = setupTables(dataLocation)
    var stagesWithJoins = Set[Int]()

    queries.foreach { name =>
      val queryString = fileToString(new File(Thread.currentThread().getContextClassLoader
        .getResource(s"tpcds/$name.sql").getFile))

      // This is an indirect hack to estimate the size of each query's input by traversing the
      // logical plan and adding up the sizes of all tables that appear in the plan. Note that this
      // currently doesn't take WITH subqueries into account which might lead to fairly inaccurate
      // per-row processing time for those cases.
      val queryRelations = scala.collection.mutable.HashSet[String]()
      spark.sql(queryString).queryExecution.logical.map {
        case ur @ UnresolvedRelation(t: TableIdentifier, _) =>
          queryRelations.add(t.table)
        case lp: LogicalPlan =>
          lp.expressions.foreach { _ foreach {
            case subquery: SubqueryExpression =>
              subquery.plan.foreach {
                case ur @ UnresolvedRelation(t: TableIdentifier, _) =>
                  queryRelations.add(t.table)
                case _ =>
              }
            case _ =>
          }
        }
        case _ =>
      }

      logError(name)
      spark.sql(queryString).explain()

      /*val numRows = queryRelations.map(tableSizes.getOrElse(_, 0L)).sum
      val benchmark = new Benchmark(s"TPCDS Snappy", numRows, 1)
      benchmark.addCase(name) { i =>
        spark.sql(queryString).collect()
      }
      benchmark.run()


      val nextStagesWithJoins = SparkNamespaceUtils.matchingStages(spark, "SortMergeJoin")
      val xwer = SparkNamespaceUtils.taskTimes(spark, nextStagesWithJoins.diff(stagesWithJoins))
      stagesWithJoins = nextStagesWithJoins


      logError(s"Found Callsites: ${xwer}") */

    }
  }

  object SparkNamespaceUtils {
    def matchingStages(spark: SparkSession, stringToMatch: String): Set[Int] = {
      spark.sparkContext.jobProgressListener.completedStages
        .filter(x => RDDOperationGraphContainsStringUtil(x, stringToMatch))
        .map(_.stageId).toSet

    }

    def stageExecutorRunTime(spark: SparkSession, stageIds: Set[Int]): Long = {
      spark.sparkContext.jobProgressListener.completedStages
        .filter(x => stageIds.contains(x.stageId)).map(_.taskMetrics.executorRunTime).sum
    }

    def taskTimes(spark: SparkSession, stageIds: Set[Int]): Seq[(Int, Seq[TaskInfo])] = {
      val taskInfos = stageIds.toSeq.map{ stageId =>

        (stageId,
          spark.sparkContext.jobProgressListener.stageIdToData((stageId, 0))
            .taskData.values.map(_.taskInfo).toSeq)

      }

      taskInfos
    }


  }

  def main(args: Array[String]): Unit = {
    //
    val tpcdsQueries = Seq("q72")//Seq("q14b", "q1", "q2", "q4", "q5", "q6", "q10", "q11", "q14a", "q14b", "q16", "q17", "q24a", "q24b", "q25", "q29", "q30", "q31", "q32", "q35", "q37", "q38", "q39a", "q39b", "q40", "q47", "q49", "q50", "q54", "q57", "q58", "q59", "q64", "q65", "q72", "q74", "q75", "q78", "q80", "q81", "q82", "q83", "q84", "q85", "q87", "q92", "q93", "q94", "q95")
    //val tpcdsQueries = Seq("q1", "q2", "q4", "q5", "q6", "q10", "q11", "q14a", "q14b", "q16", "q17", "q24a", "q24b", "q25", "q29", "q30", "q31", "q32", "q35", "q37", "q38", "q39a", "q39b", "q40", "q47", "q49", "q50", "q54", "q57", "q58", "q59", "q64", "q65", "q72", "q74", "q75", "q78", "q80", "q81", "q82", "q83", "q84", "q85", "q87", "q92", "q93", "q94", "q95")
    // List of all TPC-DS queries
     /*Seq(
      "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
      "q12", "q13", "q14a", "q14b", "q15", "q16", "q17", "q18", "q19", "q20",
      "q21", "q22", "q23a", "q23b", */
    // Seq("q24a", "q24b", "q25", "q26", "q27",
    // Q28 crashed
      /*Seq("q29", "q30",
      "q31", "q32", "q33", "q34", "q35", "q36", "q37", "q38", "q39a", "q39b", "q40",
      "q41", "q42", "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50",
      "q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58", "q59", "q60",*/
//"q62", "q63"
    /*
    "q64", "q65", "q66", "q67", "q68", "q69", "q70",
      "q71", "q72", "q73", "q74", "q75", "q76"
     */
     // Query 61 errors
    /*
    "q78", "q79", "q80",
      "q81", "q82", "q83", "q84", "q85", "q86", "q87"
     */
    // Query 88 errors
    // Query 89 is only broadcast join
    // Query 90 errors
     /*val tpcdsQueries = Seq(
      "q91", "q92", "q93", "q94", "q95", "q96", "q97", "q98", "q99")*/
      //Seq("q49", "q72", "q75", "q78", "q80", "q93")
    //Seq("q13", "q14a", "q14b", "q15", "q16", "q17", "q18", "q19", "q20",
    //  "q21", "q22", "q23a", "q23b", "q24a", "q24b")
    //Seq("q49", "q72", "q75", "q78", "q80", "q93") //q72 is slow on hash
    //"q19,q42,q52,q55,q63,q68,q73,q98,q27,q3,q43,q53,q7,q89,q34,q46,q59,q79".split(",")
    // "q77" gives error
    // "q5" and "q40" are only broadcast joins w/ the current scale factor of 5 & 16 partitions
    // "q51" and "q97" don't support hash joins because not a hashable relation
    // All the other queries don't support joins or just use hashjoin

    /*Seq(
      "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
      "q12", "q13", "q14a", "q14b", "q15", "q16", "q17", "q18", "q19", "q20",
      "q21", "q22", "q23a", "q23b", "q24a", "q24b", "q25", "q26", "q27", "q28", "q29", "q30",
      "q31", "q32", "q33", "q34", "q35", "q36", "q37", "q38", "q39a", "q39b", "q40",
      "q41", "q42", "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50",
      "q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58", "q59", "q60",
      "q61", "q62", "q63", "q64", "q65", "q66", "q67", "q68", "q69", "q70",
      "q71", "q72", "q73", "q74", "q75", "q76", "q77", "q78", "q79", "q80",
      "q81", "q82", "q83", "q84", "q85", "q86", "q87", "q88", "q89", "q90",
      "q91", "q92", "q93", "q94", "q95", "q96", "q97", "q98", "q99")*/

    // In order to run this benchmark, please follow the instructions at
    // https://github.com/databricks/spark-sql-perf/blob/master/README.md to generate the TPCDS data
    // locally (preferably with a scale factor of 5 for benchmarking). Thereafter, the value of
    // dataLocation below needs to be set to the location where the generated data is stored.
    val dataLocation = "/Users/tomerk11/Desktop/tpcds-data"

    tpcdsAll(dataLocation, queries = tpcdsQueries)
  }
}

object TPCDSDataGen {
  val conf =
    new SparkConf()
      .setMaster("local[4]")
      .setAppName("tpcds-data-gen")
      .set("spark.sql.parquet.compression.codec", "snappy")
      .set("spark.driver.memory", "3g")
      .set("spark.executor.memory", "3g")

  val spark = SparkSession.builder.config(conf).getOrCreate()

  def main(args: Array[String]): Unit = {
    val dsdgenDir = "/Users/tomerk11/Development/tpcds-kit/tools"
    val scaleFactor = 5
    import org.apache.spark.sql.execution.benchmark.tpcds.Tables
    // Tables in TPC-DS benchmark used by experiments.
    // dsdgenDir is the location of dsdgen tool installed in your machines.
    val tables = new Tables(spark.sqlContext, dsdgenDir, scaleFactor)
    // Generate data.
    tables.genData("/Users/tomerk11/Desktop/tpcds-data",
      "parquet",
      true,
      true,
      true,
      false,
      false,
      numPartitions = 16)
    // Create metastore tables in a specified database for your data.
    // Once tables are created, the current database will be switched to the specified database.
    //tables.createExternalTables(location, format, databaseName, overwrite)
    // Or, if you want to create temporary tables
    //tables.createTemporaryTables(location, format)
    // Setup TPC-DS experiment
  }
}
