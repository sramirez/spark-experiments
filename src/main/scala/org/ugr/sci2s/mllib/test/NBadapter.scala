package org.ugr.sci2s.mllib.test

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.NaiveBayes
import org.ugr.sci2s.mllib.test.{MLExperimentUtils => MLEU}
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg._

object NBadapter extends ClassifierAdapter {
  
	override def algorithmInfo (parameters: Map[String, String]): String = {
		  val lambda = parameters.getOrElse("cls-lambda", "1.0")
		  s"Algorithm: Naive Bayes (NB)\nlambda: $lambda\n\n"		
	}
  
	override def classify (train: RDD[LabeledPoint], parameters: Map[String, String]) = {
  		val lambda = MLEU.toDouble(parameters.getOrElse("cls-lambda", "1.0"), 1.0)
		  val model = NaiveBayes.train(train, lambda)
		  new NBadapter(model)
	}

}

class NBadapter(model: ClassificationModel) extends ClassificationModelAdapter {
  
  override def predict(data: RDD[Vector]): RDD[Double] = {
    model.predict(data)
  }
      
  override def predict(data: Vector): Double = {
    model.predict(data)
  }
}