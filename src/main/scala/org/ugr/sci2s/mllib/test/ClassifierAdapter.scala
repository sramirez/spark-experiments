package org.ugr.sci2s.mllib.test

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg._

trait ClassifierAdapter extends Serializable {
  
	def classify (
	    train: RDD[LabeledPoint], 
	    parameters: Map[String, String]): ClassificationModelAdapter  
	    
	def algorithmInfo (parameters: Map[String, String]): String

}