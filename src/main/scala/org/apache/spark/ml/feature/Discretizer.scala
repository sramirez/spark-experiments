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

package org.apache.spark.ml.feature

import java.{util => ju}

import org.apache.spark.SparkException
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.Model
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
 * :: Experimental ::
 * `Discretizer` maps a column of continuous features to a column of feature buckets.
 */
@Experimental
final class Discretizer(override val uid: String)
  extends Model[Discretizer] with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("Discretizer"))

  /**
   * Parameter for mapping continuous features into buckets. With n+1 splits, there are n buckets.
   * A bucket defined by splits x,y holds values in the range [x,y) except the last bucket, which
   * also includes y. Splits should be strictly increasing.
   * Values at -inf, inf must be explicitly provided to cover all Double values;
   * otherwise, values outside the splits specified will be treated as errors.
   * @group param
   */
  val splits: DoubleArrayParam = new DoubleArrayParam(this, "splits",
    "Split points for mapping continuous features into buckets. With n+1 splits, there are n " +
      "buckets. A bucket defined by splits x,y holds values in the range [x,y) except the last " +
      "bucket, which also includes y. The splits should be strictly increasing. " +
      "Values at -inf, inf must be explicitly provided to cover all Double values; " +
      "otherwise, values outside the splits specified will be treated as errors.",
    Discretizer.checkSplits)

  /** @group getParam */
  def getSplits: Array[Double] = $(splits)

  /** @group setParam */
  def setSplits(value: Array[Double]): this.type = set(splits, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema)
    val disc = udf { feature: Double =>
      Discretizer.binarySearchForBuckets($(splits), feature)
    }
    val newCol = disc(dataset($(inputCol)))
    val newField = prepOutputField(dataset.schema)
    dataset.withColumn($(outputCol), newCol, newField.metadata)
  }

  private def prepOutputField(schema: StructType): StructField = {
    val buckets = $(splits).sliding(2).map(bucket => bucket.mkString(", ")).toArray
    val attr = new NominalAttribute(name = Some($(outputCol)), isOrdinal = Some(true),
      values = Some(buckets))
    attr.toStructField()
  }

  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    SchemaUtils.checkColumnType(schema, $(inputCol), DoubleType)
    SchemaUtils.appendColumn(schema, prepOutputField(schema))
  }

  override def copy(extra: ParamMap): Discretizer = {
    defaultCopy[Discretizer](extra).setParent(parent)
  }
}

object Discretizer extends DefaultParamsReadable[Discretizer] {

  /** We require splits to be of length >= 3 and to be in strictly increasing order. */
  private[feature] def checkSplits(splits: Array[Double]): Boolean = {
    if (splits.length > 2) {
      var i = 0
      val n = splits.length - 1
      while (i < n) {
        if (splits(i) >= splits(i + 1)) return false
        i += 1
      }
      true
    } else {
      true
    }
  }

  /**
   * Binary searching in several buckets to place each data point.
   * @throws SparkException if a feature is < splits.head or > splits.last
   */
  private[feature] def binarySearchForBuckets(splits: Array[Double], feature: Double): Double = {
    if(splits.length > 0) {
      binarySearch(splits, feature)
    } else {
      feature
    }
  }
  
  private def binarySearch[A <% Ordered[A]](a: Array[A], v: A) = {
    def recurse(low: Int, high: Int): Int = (low + high) / 2 match {
      case _ if high < low => high + 1
      case mid if a(mid) > v => recurse(low, mid - 1)
      case mid if a(mid) < v => recurse(mid + 1, high)
      case mid => mid
    }
    recurse(0, a.size - 1)
  }

  @Since("1.6.0")
  override def load(path: String): Discretizer = super.load(path)
}
