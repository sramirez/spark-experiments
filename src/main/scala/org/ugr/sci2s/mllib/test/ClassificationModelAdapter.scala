package org.ugr.sci2s.mllib.test

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg._

trait ClassificationModelAdapter extends Serializable {
      
  def predict(data: RDD[Vector]): RDD[Double]
      
  def predict(data: Vector): Double

}