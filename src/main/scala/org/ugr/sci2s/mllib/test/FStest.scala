package org.ugr.sci2s.mllib.test

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import java.util.ArrayList
import org.apache.spark.mllib.feature._

object FStest {

  def main(args: Array[String]): Unit = {
	  val initStartTime = System.nanoTime()

		val conf = new SparkConf().setAppName("FS test")
		val sc = new SparkContext(conf)
		
		// Load data
    val path = "hdfs://localhost:8020/user/sramirez/datasets/epsilon_1K.data"
    val sparse = MLUtils.loadLibSVMFile(sc, path)
    val data = sparse.map{lp => LabeledPoint(lp.label, 
          Vectors.dense(lp.features.toArray.map(p => BigDecimal(p).setScale(6, BigDecimal.RoundingMode.HALF_UP).toDouble)))
        }
    // BigDecimal(d).setScale(6, BigDecimal.RoundingMode.HALF_UP).toFloat
      //.map{ str => 
      //  val arr = str.split(",").map(_.toDouble)
      //  new LabeledPoint(arr.last, Vectors.dense(arr.slice(0, arr.length - 1)))
      //}
    
    val discretizer = MDLPDiscretizer.train(data)
    val discData = data.map { lp => 
      LabeledPoint(lp.label, discretizer.transform(lp.features)) 
    } 
    
    val criterion = new InfoThCriterionFactory("mrmr")
    val selector = InfoThSelector.train(criterion, discData)
    
    val redData = discData.map { lp => 
      LabeledPoint(lp.label, selector.transform(lp.features)) 
    } 
    
    println(redData.first().toString())
  }

}