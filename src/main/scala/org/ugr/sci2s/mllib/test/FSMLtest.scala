package org.ugr.sci2s.mllib.test

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature._

object FSMLtest {

  def main(args: Array[String]): Unit = {

		val conf = new SparkConf().setMaster("local[3]").setAppName("FS test in ml API")
		val sc = new SparkContext(conf)
    // sc is an existing SparkContext.
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
		import sqlContext.implicits._
    
		val data = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 1.0), 0.0)
    )
    
    val df = sc.parallelize(data, 1).toDF("id", "features", "clicked")
    
    val discretizer = new MDLPDiscretizer()
      .setMaxBins(10)
      .setInputCol("features")
      .setLabelCol("clicked")
      .setOutputCol("buckedFeatures")
      
    val result = discretizer.fit(df).transform(df)
    
    /*val selector = new InfoThSelector()
      .setSelectCriterion("mrmr")
      .setNPartitions(1)
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")
    
    val result = selector.fit(df).transform(df)*/
    result.show()
    
  }

}