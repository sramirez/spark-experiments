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

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.attribute.NominalAttribute

/**
 * Params for [[MDLPDiscretizer]] and [[MDLPDiscretizerModel]].
 */
private[feature] trait MDLPDiscretizerParams extends Params with HasInputCol with HasOutputCol with HasLabelCol {

  /**
   * Maximum number of buckets into which data points are grouped. Must
   * be >= 2.
   * default: 2
   * @group param
   */
  val maxBins = new IntParam(this, "maxBins", "Maximum number of bins" +
    "into which data points are grouped. Must be >= 2.",
    ParamValidators.gtEq(2))
  setDefault(maxBins -> 2)

  /** @group getParam */
  def getMaxBins: Int = getOrDefault(maxBins)
  
    /**
   * Maximum number of buckets (quantiles, or categories) into which data points are grouped. Must
   * be >= 2.
   * default: 2
   * @group param
   */
  val maxByPart = new IntParam(this, "maxByPart", "Maximum number of elements per partition" +
    "to considere in each evaluation process. Must be >= 10,000.",
    ParamValidators.gtEq(10000))
  setDefault(maxByPart -> 10000)

  /** @group getParam */
  def getMaxByPart: Int = getOrDefault(maxByPart)

}

/**
 * :: Experimental ::
 * MDLPDiscretizer trains a model to project vectors to a low-dimensional space using MDLPDiscretizer.
 */
@Experimental
class MDLPDiscretizer (override val uid: String) extends Estimator[MDLPDiscretizerModel] with MDLPDiscretizerParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("MDLPDiscretizer"))

  /** @group setParam */
  def setMaxBins(value: Int): this.type = set(maxBins, value)
  
  /** @group setParam */
  def setMaxByPart(value: Int): this.type = set(maxByPart, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)
  
  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /**
   * Computes a [[MDLPDiscretizerModel]] that contains the principal components of the input vectors.
   */
  override def fit(dataset: DataFrame): MDLPDiscretizerModel = {
    transformSchema(dataset.schema, logging = true)
    val input = dataset.select($(labelCol), $(inputCol)).map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }
    val discretizer = feature.MDLPDiscretizer.train(input, None, $(maxBins), $(maxByPart))
    println(discretizer.thresholds.mkString("\n"))
    val bucketizers = discretizer.thresholds.map(splits => new Discretizer(uid).setSplits(splits.map(_.toDouble) :+ Double.PositiveInfinity))
    copyValues(new MDLPDiscretizerModel(uid, bucketizers).setParent(this))
  }
  
  override def transformSchema2(schema: StructType): StructType = {

    
    validateParams()
    SchemaUtils.checkColumnType(schema, $(inputCol), DoubleType)
    val inputFields = schema.fields
    require(inputFields.forall(_.name != $(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val featureAttributes = Array.fill[attribute.Attribute](schema.fields.length)(NominalAttribute.defaultAttr)
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes.toArray)
    val outputFields = schema.fields :+ newAttributeGroup.toStructField()
    StructType(outputFields)
  }
  
  

  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): MDLPDiscretizer = defaultCopy(extra)
}

@Since("1.6.0")
object MDLPDiscretizer extends DefaultParamsReadable[MDLPDiscretizer] {

  @Since("1.6.0")
  override def load(path: String): MDLPDiscretizer = super.load(path)
}

/**
 * :: Experimental ::
 * Model fitted by [[MDLPDiscretizer]].
 *
 * @param pc A principal components Matrix. Each column is one principal component.
 * @param explainedVariance A vector of proportions of variance explained by
 *                          each principal component.
 */
@Experimental
class MDLPDiscretizerModel private[ml] (
    override val uid: String,
    val splits: Array[Discretizer])
  extends Model[MDLPDiscretizerModel] with MDLPDiscretizerParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Transform a vector by computed Principal Components.
   * NOTE: Vectors to be transformed must be the same length
   * as the source vectors given to [[MDLPDiscretizer.fit()]].
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    def discretize(dataset: DataFrame) = {
      dataset.select($(inputCol)).columns
    }
    
    def discretize2(dataset: DataFrame) = {
      val cols = dataset.select($(inputCol)).columns
      val transfCols = for(i <- 0 until cols.length) yield splits(i).transform(dataset.select(cols(i)))
      transfCols.reduce(_ unionAll _)   
    }
    
    val mdlpOp = udf { discretize2 _ }
    dataset.withColumn($(outputCol), mdlpOp(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateParams()
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): MDLPDiscretizerModel = {
    val copied = new MDLPDiscretizerModel(uid, splits)
    copyValues(copied, extra).setParent(parent)
  }

  //@Since("1.6.0")
  //override def write: MLWriter = new MDLPDiscretizerModelWriter(this)
}
