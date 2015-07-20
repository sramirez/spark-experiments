package org.ugr.sci2s.mllib.test

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.ugr.sci2s.mllib.test.{MLExperimentUtils => MLEU}
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg._

object SVMadapter extends ClassifierAdapter {
  
	def algorithmInfo (parameters: Map[String, String]): String = {
  		val numIter = parameters.getOrElse("cls-numIter", "100")
  		val stepSize = parameters.getOrElse("cls-stepSize", "1.0")
  		val regParam = parameters.getOrElse("cls-regParam", "1.0")
  		val miniBatchFraction = parameters.getOrElse("cls-miniBatchFraction", "1.0")
		
  		s"Algorithm: Support Vector Machine (SVM)\n" + 
			s"numIter: $numIter\n" +
			s"stepSize: $stepSize\n" + 
			s"regParam: $regParam\n" +
			s"miniBatchFraction: $miniBatchFraction\n\n"		
	}
  
	private def calcThreshold(model: SVMModel, 
	    data: RDD[LabeledPoint]): Unit = {
	  
  		// Clear the default threshold.
		model.clearThreshold()
  		// Compute raw scores on the test set. 
		val scoreAndLabels = data.map { point =>
		  val score = model.predict(point.features)
		  (score, point.label)
		}
		
		// Get evaluation metrics.
		val metrics = new BinaryClassificationMetrics(scoreAndLabels)
		val measuresByThreshold = metrics.fMeasureByThreshold.collect()
		val maxThreshold = measuresByThreshold.maxBy{_._2}
		
		//println("Max (Threshold, Precision):" + maxThreshold)
		model.setThreshold(maxThreshold._1)			  
	}
  
	def classify (train: RDD[LabeledPoint], parameters: Map[String, String]): ClassificationModelAdapter = {
  	val numIter = MLEU.toInt(parameters.getOrElse("cls-numIter", "100"), 100)
		val stepSize = MLEU.toDouble(parameters.getOrElse("cls-stepSize", "1.0"), 1.0)
		val regParam = MLEU.toDouble(parameters.getOrElse("cls-regParam", "1.0"), 1.0)
		val miniBatchFraction = MLEU.toDouble(parameters.getOrElse("cls-miniBatchFraction", "1.0"), 1.0)
		val model = SVMWithSGD.train(train, numIter, stepSize, regParam, miniBatchFraction)
		calcThreshold(model, train)
		new SVMadapter(model)
	}

}

class SVMadapter(model: ClassificationModel) extends ClassificationModelAdapter {
  
  override def predict(data: RDD[Vector]): RDD[Double] = {
    model.predict(data)
  }
      
  override def predict(data: Vector): Double = {
    model.predict(data)
  }
}


