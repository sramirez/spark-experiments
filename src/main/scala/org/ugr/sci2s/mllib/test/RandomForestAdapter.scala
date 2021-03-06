package org.ugr.sci2s.mllib.test

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.ugr.sci2s.mllib.test.{MLExperimentUtils => MLEU}
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

object RandomForestAdapter extends ClassifierAdapter {
  
	override def algorithmInfo (parameters: Map[String, String]): String = {
      val numClasses = parameters.getOrElse("cls-numClasses", "2")
      val impurity = parameters.getOrElse("cls-impurity", "gini")
      val featSubSet = parameters.getOrElse("cls-featureSubsetStrategy", "auto")
      val numTrees = MLEU.toInt(parameters.getOrElse("cls-numTrees", "100"), 100)
      val maxDepth = parameters.getOrElse("cls-maxDepth", "4")
      val maxBins = parameters.getOrElse("cls-maxBins", "100")
		
  		s"Algorithm: Random Forest (RF)\n" + 
			s"numClasses: $numClasses\n" +
      s"numTrees: $numTrees\n" +
      s"featureSubsetStrategy: $featSubSet\n" +
			s"impurity: $impurity\n" + 
      s"maxBins: $maxBins\n" +
			s"maxDepth: $maxDepth\n\n"		
	}
  
  override def classify (train: RDD[LabeledPoint], parameters: Map[String, String]): ClassificationModelAdapter = {
    val numClasses = MLEU.toInt(parameters.getOrElse("cls-numClasses", "2"), 2)
    val impurity = parameters.getOrElse("cls-impurity", "gini")
    val featSubSet = parameters.getOrElse("cls-featureSubsetStrategy", "auto")
    val numTrees = MLEU.toInt(parameters.getOrElse("cls-numTrees", "100"), 100)
    val maxDepth = MLEU.toInt(parameters.getOrElse("cls-maxDepth", "4"), 4)
    val maxBins = MLEU.toInt(parameters.getOrElse("cls-maxBins", "100"), 100)
    /*val categoricalFeaturesInfo = parameters.get("disc") match {              
      case Some(s) if s matches "(?i)yes" => 
        val bins = MLEU.toInt(parameters.getOrElse("disc-nbins", "15"), 15) + 2 // discretization starts on 1
        val categInfo = for(i <- 0 until train.first().features.size) yield (i, bins) 
        categInfo.toMap
      case _ =>  Map.empty[Int, Int]
    }*/
    val model = RandomForest.trainClassifier(train, 
        numClasses, Map.empty[Int, Int], numTrees, featSubSet, 
        impurity, maxDepth, maxBins)
    new RandomForestAdapter(model)
  }
  
	def classify (train: RDD[LabeledPoint], parameters: Map[String, String], nominalInfo: Map[Int, Int]): ClassificationModelAdapter = {
    val numClasses = MLEU.toInt(parameters.getOrElse("cls-numClasses", "2"), 2)
    val impurity = parameters.getOrElse("cls-impurity", "gini")
    val featSubSet = parameters.getOrElse("cls-featureSubsetStrategy", "auto")
    val numTrees = MLEU.toInt(parameters.getOrElse("cls-numTrees", "100"), 100)
    val maxDepth = MLEU.toInt(parameters.getOrElse("cls-maxDepth", "4"), 4)
    val maxBins = MLEU.toInt(parameters.getOrElse("cls-maxBins", "100"), 100)
    /*val categoricalFeaturesInfo = if(nominalInfo.isEmpty) {
      parameters.get("disc") match {              
        case Some(s) if s matches "(?i)yes" => 
          val bins = MLEU.toInt(parameters.getOrElse("disc-nbins", "15"), 15) + 2
          val categInfo = for(i <- 0 until train.first().features.size) yield (i, bins) 
          categInfo.toMap
        case _ =>  Map.empty[Int, Int]
      }
    } else {
      nominalInfo
    }*/
    val model = RandomForest.trainClassifier(train, 
        numClasses, nominalInfo, numTrees, featSubSet, 
        impurity, maxDepth, maxBins)
    new RandomForestAdapter(model)
	}

}

class RandomForestAdapter(model: RandomForestModel) extends ClassificationModelAdapter {
  
  override def predict(data: RDD[Vector]): RDD[Double] = {
    model.predict(data)
  }
      
  override def predict(data: Vector): Double = {
    model.predict(data)
  }
}


