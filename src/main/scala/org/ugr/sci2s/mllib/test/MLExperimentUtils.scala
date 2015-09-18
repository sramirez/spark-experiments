package org.ugr.sci2s.mllib.test

import scala.util.Random
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.annotation.Experimental
import org.apache.spark.SparkContext
import scala.collection.immutable.List
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.hadoop.mapreduce.lib.input.InvalidInputException
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.SparseVector

object MLExperimentUtils {
  
	    def toInt(s: String, default: Int): Int = {
  			try {
  				s.toInt
  			} catch {
  				case e:Exception => default
  			}
	    }	
	
	    def toDouble(s: String, default: Double): Double = {
  			try {
  				s.toDouble
  			} catch {
  				case e: Exception => default
  			}
		  }
  
  		private def parseThresholds (str: String) = {
  			val tokens = str split ","
        val index = tokens(0).toInt // Int as key (ordering)
  			val thresh = if (tokens.length > 1) tokens.slice(1, tokens.length).map(_.toFloat) else Array.empty[Float]
        (index, thresh)
  			//val attIndex = tokens(0).toInt
  			//(attIndex, points.toSeq)
        //points
  		}
  		
  		private def parseSelectedAtts (str: String) = {
        val tokens = str split "\t"
        tokens(0).toInt
  		}
  		
  		private def parsePredictions(str: String) = {
  			val tokens = str split "\t"
  			(tokens(0).toDouble, tokens(1).toDouble)
  		}
  
  		def computePredictions(model: ClassificationModelAdapter, data: RDD[LabeledPoint], threshold: Double = .5) =
			  data.map(point => (point.label, if(model.predict(point.features) >= threshold) 1.0 else 0.0))

 		  def computePredictions (model: ClassificationModelAdapter, data: RDD[LabeledPoint]) =
			  data.map(point => (point.label, model.predict(point.features)))
        
      def computePredictions2 (model: ClassificationModelAdapter, data: RDD[LabeledPoint]) = {
        model.predict(data.map(_.features)).zip(data.map(_.label)).map(_.swap)
      }    
		
		  def computeAccuracy (valuesAndPreds: RDD[(Double, Double)]) = 
		    valuesAndPreds.filter(r => r._1 == r._2).count.toDouble / valuesAndPreds.count
		  
  		def computeAccuracyLabels (valuesAndPreds: RDD[(String, String)]) = 
		    valuesAndPreds.filter(r => r._1 == r._2).count.toDouble / valuesAndPreds.count
		
		private val positive = 1
		private val negative = 0	
		
		private def calcAggStatistics = (scores: Seq[Double]) => {
	  		val sum = scores.reduce(_ + _)
	  		val mean = sum / scores.length
	  		val devs = scores.map(score => (score - mean) * (score - mean))
	  		val stddev = Math.sqrt(devs.reduce(_ + _) / devs.length)
	  		(mean, stddev)
		}
    
		private def discretization(
				discretize: (RDD[LabeledPoint]) => DiscretizerModel, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int,
				save: Boolean = false) = {
		  
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val thresholds = sc.textFile(outputDir + "/discThresholds_" + iteration)
                  .filter(!_.isEmpty())
									.map(parseThresholds)
                  .sortByKey()
                  .values.collect()
        
				val discAlgorithm = new DiscretizerModel(thresholds)
				val discTime = sc.textFile(outputDir + "/disc_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)
						.first
        // More efficient than by-instance version
        println("Readed tresholds")
        val discData = discAlgorithm.transform(train.map(_.features))
          .zip(train.map(_.label))
          .map{case (v, l) => LabeledPoint(l, v)}
        val discTestData = discAlgorithm.transform(test.map(_.features))
          .zip(test.map(_.label))
          .map{case (v, l) => LabeledPoint(l, v)}
        
        // Save discretized data 
        if(save) {
          MLUtils.saveAsLibSVMFile(discData, outputDir + "/disc_train_" + iteration + ".csv")
          MLUtils.saveAsLibSVMFile(discTestData, outputDir + "/disc_test_" + iteration + ".csv")
          //discData.saveAsTextFile(outputDir + "/disc_train_" + iteration + ".csv")
          //discTestData.saveAsTextFile(outputDir + "/disc_test_" + iteration + ".csv")       
        }
        
        println("Discretized all data")
        
				(discData, discTestData, discTime)			
				
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException =>
					
          val dtrain = train.persist(StorageLevel.MEMORY_ONLY)
          val c = dtrain.count()
          
          val initStartTime = System.nanoTime()
					val discAlgorithm = discretize(dtrain)
					val discTime = (System.nanoTime() - initStartTime) / 1e9 
          
          val thrsRDD = sc.parallelize(discAlgorithm.thresholds.zipWithIndex)
            .map(_.swap)
            .sortByKey()
            .map({case (i, arr) => i + "," + arr.mkString(",")})
          
          thrsRDD.saveAsTextFile(outputDir + "/discThresholds_" + iteration)
          
          // More efficient than by-instance version
          val discData = discAlgorithm.transform(dtrain.map(_.features))
            .zip(dtrain.map(_.label))
            .map{case (v, l) => LabeledPoint(l, v)}
          val discTestData = discAlgorithm.transform(test.map(_.features))
            .zip(test.map(_.label))
            .map{case (v, l) => LabeledPoint(l, v)}
          
          // Save discretized data 
          if(save) {
            MLUtils.saveAsLibSVMFile(discData, outputDir + "/disc_train_" + iteration + ".csv")
            MLUtils.saveAsLibSVMFile(discTestData, outputDir + "/disc_test_" + iteration + ".csv")
          //discData.saveAsTextFile(outputDir + "/disc_train_" + iteration + ".csv")
          //discTestData.saveAsTextFile(outputDir + "/disc_test_" + iteration + ".csv")             
          } 
          
					val strTime = sc.parallelize(Array(discTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "/disc_time_" + iteration)
          
          dtrain.unpersist()
					
					(discData, discTestData, discTime)
			}		
		}
		
		private def featureSelection(
				fs: (RDD[LabeledPoint]) => SelectorModel, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int,
        save: Boolean) = {
		  
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val selectedAtts = sc.textFile(outputDir + "/fs_scheme_" + iteration).filter(!_.isEmpty())
										.map(parseSelectedAtts).collect				
				val featureSelector = new SelectorModel(selectedAtts.map(_ - 1).sorted) // Must be transformed to indices (-1)
				
				val FSTime = sc.textFile(outputDir + "/fs_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)
						.first
         
        val redTrain = train.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
        val redTest = test.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
        
          // Save reduced data 
          if(save) {
             redTrain.saveAsTextFile(outputDir + "/fs_train_" + iteration + ".csv")
             redTest.saveAsTextFile(outputDir + "/fs_test_" + iteration + ".csv")     
          }         
        
				(redTrain, redTest, FSTime)
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException =>
					
          val fstrain = train.persist(StorageLevel.MEMORY_ONLY)
          val c = fstrain.count()
          
          val initStartTime = System.nanoTime()
					val featureSelector = fs(fstrain)
					val FSTime = (System.nanoTime() - initStartTime) / 1e9
          
          
          val redTrain = fstrain.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
          val redTest = test.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
          
          // Save reduced data 
          if(save) {
             redTrain.saveAsTextFile(outputDir + "/fs_train_" + iteration + ".csv")
             redTest.saveAsTextFile(outputDir + "/fs_test_" + iteration + ".csv")       
          }    
					
					// Save the obtained FS scheme in a HDFS file (as a sequence)					
					val selectedAtts = featureSelector.selectedFeatures
          val output = selectedAtts.mkString("\n")
					val parFSscheme = sc.parallelize(Array(output), 1)
					parFSscheme.saveAsTextFile(outputDir + "/fs_scheme_" + iteration)
					val strTime = sc.parallelize(Array(FSTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "/fs_time_" + iteration)
          
          fstrain.unpersist()
					
					(redTrain, redTest, FSTime)
			}
		}
		
		private def classification(
				classify: (RDD[LabeledPoint]) => ClassificationModelAdapter, 
				train: RDD[LabeledPoint], 
				test: RDD[LabeledPoint], 
				outputDir: String,
				iteration: Int) = {
		  	
			val sc = train.context
		  	/** Check if the results for this fold are already written in disk
		  	 *  if not, we calculate them
		  	 **/
			try {
				val traValuesAndPreds = sc.textFile(outputDir + "/result_" + iteration + ".tra")
						.filter(!_.isEmpty())
						.map(parsePredictions)
						
				val tstValuesAndPreds = sc.textFile(outputDir + "/result_" + iteration + ".tst")
						.filter(!_.isEmpty())
						.map(parsePredictions)
						
				val classifficationTime = sc.textFile(outputDir + "/classification_time_" + iteration)
						.filter(!_.isEmpty())
						.map(_.toDouble)	
						.first
				
				(traValuesAndPreds, tstValuesAndPreds, classifficationTime)
			} catch {
				case iie: org.apache.hadoop.mapred.InvalidInputException => 
          val ctrain = train.persist(StorageLevel.MEMORY_ONLY)
          val nInstances = ctrain.count() // to persist train and not to affect time measurements
					
          val initStartTime = System.nanoTime()	
					val classificationModel = classify(ctrain)
					val classificationTime = (System.nanoTime() - initStartTime) / 1e9
					
					val traValuesAndPreds = computePredictions2(classificationModel, ctrain).cache()
					val tstValuesAndPreds = computePredictions2(classificationModel, test).cache()
					
          //val c = tstValuesAndPreds.count()
					// Save prediction results
					/*val outputTrain = traValuesAndPreds.map(t => t._1.toInt + "\t" + t._2.toInt)   
					outputTrain.saveAsTextFile(outputDir + "/result_" + iteration + ".tra")
					val outputTest = tstValuesAndPreds.map(t => t._1.toInt + "\t" + t._2.toInt)  
          outputTest.saveAsTextFile(outputDir + "/result_" + iteration + ".tst")*/
					
          val strTime = sc.parallelize(Array(classificationTime.toString), 1)
					strTime.saveAsTextFile(outputDir + "/classification_time_" + iteration)
          
          ctrain.unpersist()
					
					(traValuesAndPreds, tstValuesAndPreds, classificationTime)
			}
		}

		/**
		 * Execute a MLlib experiment with three optional phases (discretization, feature selection and classification)
		 * @param sc Spark context
		 * @param discretize Optional function to discretize a dataset
		 * @param featureSelect Optional function to reduce the set of features
		 * @param classify Optional function to classify a dataset
		 * @param inputData File or directory path where the data set files are placed
		 * @param outputDir HDFS output directory for the experiment
		 * @param algoInfo Some basis information about the algorithm to be executed
     * @param npart Number of partitions to re-format the dataset
		 */		
		def executeExperiment(
		    sc: SparkContext,
		    discretize: (Option[(RDD[LabeledPoint]) => DiscretizerModel], Boolean), 
		    featureSelect: (Option[(RDD[LabeledPoint]) => SelectorModel], Boolean), 
		    classify: Option[(RDD[LabeledPoint]) => ClassificationModelAdapter],
		    inputData: (Any, String, Boolean), 
		    outputDir: String, 
		    algoInfo: String,
        npart: Int) {

			def getDataFiles(dirPath: String, k: Int): Array[(String, String)] = {
				val subDir = dirPath.split("/").last
				def genName = (i: Int) => dirPath + "/" + subDir.replaceFirst("fold", i.toString)
				val cvdata = for (i <- 1 to k) yield (genName(i) + "tra.data", genName(i) + "tst.data")
				cvdata.toArray
			}
      
		      val (headerFile, dataFiles, format, dense) = inputData match {
		        case ((header: Option[String], dataPath: String, kfold: Int), format: String, dense: Boolean) => 
		          (header, getDataFiles(dataPath, kfold), format, dense)
		        case ((header: Option[String], train: String, test: String), format: String, dense: Boolean) => 
		          (header, Array((train, test)), format, dense)
		      } 
			
		      val samplingRate = 1.0      
    		  // Create the function to read the labeled points
    		  val readFile = format match {
		          case s if s matches "(?i)LibSVM" => 
		            (filePath: String) => {
		              val svmData = MLUtils.loadLibSVMFile(sc, filePath)
                  svmData.unpersist()
		              val data = if(samplingRate < 1.0) svmData.sample(false, samplingRate) else svmData
		              if(dense) {
		                data.map{ case LabeledPoint(label, features) =>  
                      val flabel = if(label == -1.0) 0.0 else label
		                  new LabeledPoint(flabel, Vectors.dense(features.toArray))
                    }
		              } else {
                    data.map{ case LabeledPoint(label, features) =>       
                      val flabel = if(label == -1.0) 0.0 else label
                      new LabeledPoint(flabel, features)
                    }
                    // data
		              }
		            }
		          case _ => 
		           (filePath: String) => {
		              val typeConversion = KeelParser.parseHeaderFile(sc, headerFile.get) 
		              val bcTypeConv = sc.broadcast(typeConversion)
		              val lines = sc.textFile(filePath: String)
		              val data = if(samplingRate < 1.0) lines.sample(false, samplingRate) else lines
		              data.map(line => KeelParser.parseLabeledPoint(bcTypeConv.value, line))         
		           }
		      }      
      
			val info = Map[String, String]("algoInfo" -> algoInfo)
			val times = scala.collection.mutable.Map[String, Seq[Double]] ("FullTime" -> Seq(),
			    "DiscTime" -> Seq(),
			    "FSTime" -> Seq(),
			    "ClsTime" -> Seq())
			
			val nFolds = dataFiles.length
			var predictions = Array.empty[(RDD[(Double, Double)], RDD[(Double, Double)])]
			    
			val accTraResults = Seq.empty[(Double, Double)]
			for (i <- 0 until nFolds) {
								var initAllTime = System.nanoTime()
                
				val (trainFile, testFile) = dataFiles(i)
				val rawtra = readFile(trainFile)
				val rawtst = readFile(testFile)
				
        val nparttr = rawtra.partitions.size; val npartts = rawtst.partitions.size
        val trainData = if(npart <= nparttr) rawtra.coalesce(npart, false).cache() else rawtra.repartition(npart).cache()
        //val tstData = if(npart > npartts) rawtst.coalesce(npart, false) else rawtst.repartition(npart)
        val testData = rawtst
        
				// Discretization
				var trData = trainData; var tstData = testData        
				var taskTime = 0.0
				discretize match { 
				  case (Some(disc), b) => 
				    val (discTrData, discTstData, discTime) = discretization(
								disc, trData, tstData, outputDir, i, save = b) 
					trData = discTrData
					tstData = discTstData
					taskTime = discTime
				  case _ => /* criteria not fulfilled, do not discretize */
				}				
				times("DiscTime") = times("DiscTime") :+ taskTime         

				// Feature Selection
				featureSelect match { 
				  case (Some(fs), b) => 
				    val (fsTrainData, fsTestData, fsTime) = 
				      featureSelection(fs, trData, tstData, outputDir, i, save = b)
					trData = fsTrainData
					tstData = fsTestData
					taskTime = fsTime
				  case _ => taskTime = 0.0 /* criteria not fulfilled, do not do select */
				}
				times("FSTime") = times("FSTime") :+ taskTime
				
				//Classification        
				classify match { 
				  case Some(cls) => 
				    val (traValuesAndPreds, tstValuesAndPreds, classificationTime) = 
				  		classification(cls, trData, tstData, outputDir, i)
					taskTime = classificationTime
					predictions = predictions :+ (traValuesAndPreds, tstValuesAndPreds)
				  case None => taskTime = 0.0 /* criteria not fulfilled, do not classify */
				}
				times("ClsTime") = times("ClsTime") :+ taskTime
				
				trainData.unpersist() 
        //testData.unpersist()
				
				var fullTime = (System.nanoTime() - initAllTime) / 1e9
				times("FullTime") = times("FullTime") :+ fullTime
			}
			
			// Print the aggregated results
			printResults(sc, outputDir, predictions, info, getTimeResults(times.toMap))
		}
    
	    private def getTimeResults(timeResults: Map[String, Seq[Double]]) = {
	        "Mean Discretization Time:\t" + 
	            timeResults("DiscTime").sum / timeResults("DiscTime").size + " seconds.\n" +
	        "Mean Feature Selection Time:\t" + 
	            timeResults("FSTime").sum / timeResults("FSTime").size + " seconds.\n" +
	       "Mean Classification Time:\t" + 
	            timeResults("ClsTime").sum / timeResults("ClsTime").size + " seconds.\n" +
	        "Mean Execution Time:\t" + 
	            timeResults("FullTime").sum / timeResults("FullTime").size + " seconds.\n" 
	    }

		private def printResults(
				sc: SparkContext,
				outputDir: String, 
				predictions: Array[(RDD[(Double, Double)], RDD[(Double, Double)])], 
				info: Map[String, String],
				timeResults: String) {
         
      var output = timeResults
      if(!predictions.isEmpty){
          // Statistics by fold
          output += info.get("algoInfo").get + "Accuracy Results\tTrain\tTest\n"
    			val traFoldAcc = predictions.map(_._1).map(computeAccuracy)
    			val tstFoldAcc = predictions.map(_._2).map(computeAccuracy)		
    			// Print fold results into the global result file
    			for (i <- 0 until predictions.size){
    				output += s"Fold $i:\t" +
    					traFoldAcc(i) + "\t" + tstFoldAcc(i) + "\n"
    			} 
    			
    			// Aggregated statistics
    			val (aggAvgAccTr, aggStdAccTr) = calcAggStatistics(traFoldAcc)
    			val (aggAvgAccTst, aggStdAccTst) = calcAggStatistics(tstFoldAcc)
    			output += s"Avg Acc:\t$aggAvgAccTr\t$aggAvgAccTst\n"
    			output += s"Svd acc:\t$aggStdAccTr\t$aggStdAccTst\n\n\n"
    					
    			// Confusion Matrix
          val trastat = new MulticlassMetrics(predictions.map(_._1).reduceLeft(_ ++ _))      
    			val tststat = new MulticlassMetrics(predictions.map(_._2).reduceLeft(_ ++ _))			    
  		    output += "Test Confusion Matrix\n" + tststat.confusionMatrix.toString + "\n"
          output += "Train Confusion Matrix\n" + trastat.confusionMatrix.toString + "\n"
          output += "F-Measure (tra/tst):" + trastat.fMeasure + " - " + tststat.fMeasure + "\n"
          output += "Precision (tra/tst):" + trastat.fMeasure + " - " + tststat.precision + "\n"
          output += "Recall(tra/tst):" + trastat.recall + " - " + tststat.recall + "\n"
          val traauc = (1 + trastat.truePositiveRate(1.0) - trastat.falsePositiveRate(1.0)) / 2
          val tstauc = (1 + tststat.truePositiveRate(1.0) - tststat.falsePositiveRate(1.0)) / 2
          output += "AUC (label: 1.0): " + traauc + " - " + tstauc + "\n"             
      }
			println(output)
			
			val hdfsOutput = sc.parallelize(Array(output), 1)
			hdfsOutput.saveAsTextFile(outputDir + "/globalResult.txt")
		}
}
