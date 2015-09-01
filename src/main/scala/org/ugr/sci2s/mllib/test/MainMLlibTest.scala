package org.ugr.sci2s.mllib.test

import org.apache.spark.mllib.classification._
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.ugr.sci2s.mllib.test.{MLExperimentUtils => MLEU}
import org.apache.spark.mllib.feature._
import breeze.linalg.SparseVector
import breeze.linalg.DenseVector
import breeze.linalg.Vector

class MLlibRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[LabeledPoint])    
    kryo.register(classOf[SparseVector[Byte]])    
    kryo.register(classOf[DenseVector[Byte]])  
    kryo.register(classOf[Vector[Byte]])  
  }
}

object MainMLlibTest {

	def main(args: Array[String]) {
	  
		val initStartTime = System.nanoTime()
		
		val conf = new SparkConf().setAppName("MLlibTest")
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
		conf.set("spark.kryo.registrator", "org.ugr.sci2s.mllib.test.MLlibRegistrator")
		val sc = new SparkContext(conf)

		println("Usage: MLlibTest --header-file=\"hdfs://\" (--train-file=\"hdfs://\" --test-file=\"hdfs://\" " 
		    + "| --data-dir=\"hdfs://\") --output-dir=\"hdfs:\\ --disc=yes [ --disc-nbins=10 --save-disc=yes ] --fs=yes [ --fs-criterion=mrmr "
		    + "--fs-nfeat=100 --fs-npart=864 --save-fs=yes ] --file-format=LibSVM|KEEL --data-format=sparse|dense "
        + "--classifier=no|SVM|NB|LR|DT [ --cls-lambda=1.0 --cls-numIter=1 --cls-stepSize = 1.0"
		    + "--cls-regParam=1.0 --cls-miniBatchFraction=1.0 ]")
        
		// Create a table of parameters (parsing)
		val params = args.map({arg =>
		  	val param = arg.split("--|=").filter(_.size > 0)
		  	param.size match {
		  		case 2 =>  (param(0) -> param(1))
		  		case _ =>  ("" -> "")
		  	}
		}).toMap		
		
		val outputDir = params.get("output-dir")
		
		// Header file and output dir must be present
		outputDir match {
			case None => 
			  System.err.println("Bad usage. Output dir is missing.")
			  System.exit(-1)
			case _ =>
		}
		
		// Discretization
		val disc = (train: RDD[LabeledPoint]) => {
			//val discretizedFeat = Some(((0 to 2) ++ (21 to 38) ++ (93 to 130) ++ (151 to 630)).toSeq)
      val discretizedFeat: Option[Seq[Int]] = None
			val nBins = MLEU.toInt(params.getOrElse("disc-nbins", "15"), 15)

			println("*** Discretization method: Fayyad discretizer (MDLP)")
			//println("*** Features to discretize: " + discretizedFeat.get.mkString(","))
			println("*** Number of bins: " + nBins)			

			val discretizer = MDLPDiscretizer.train(train,
					discretizedFeat, // continuous features 
					nBins) // max number of values per feature
		    discretizer
		}
		
		val discretization = params.get("disc") match {
			case Some(s) if s matches "(?i)yes" => 
		        params.get("save-disc") match {
		          case Some(s) if s matches "(?i)yes" => 
		            (Some(disc), true)
		          case _ => (Some(disc), false)
		        }
			case _ => (None, false)
		}		
		
		// Feature Selection
		val fs = (data: RDD[LabeledPoint]) => {
			val criterion = new InfoThCriterionFactory(params.getOrElse("fs-criterion", "mrmr"))
			val nToSelect = MLEU.toInt(params.getOrElse("fs-nfeat", "100"), 100)
			val npart = MLEU.toInt(params.getOrElse("fs-npart", "864"), 864) // 0 -> w/o pool

			println("*** FS criterion: " + criterion.getCriterion.toString)
			println("*** Number of features to select: " + nToSelect)
			println("*** Number of partitions to use in FS: " + npart)
      
			val model = InfoThSelector.train(criterion, 
		      data,
		      nToSelect, // number of features to select
		      npart) // number of features in pool
		    model
		}
		
		val featureSelection = params.get("fs") match {
			case Some(s) if s matches "(?i)yes" => 
	        params.get("save-fs") match {
	          case Some(s) if s matches "(?i)yes" => 
	            (Some(fs), true)
	          case _ => (Some(fs), false)
	        }
			case _ => (None, false)
		}
	
		
	    
	    val format = params.get("file-format") match {
	        case Some(s) if s matches "(?i)LibSVM" => s
	        case _ => "KEEL"             
	    }
	    
	    val dense = params.get("data-format") match {
	        case Some(s) if s matches "(?i)sparse" => false
	        case _ => true             
	    }
					
	// Extract data files    
    val header = params.get("header-file")
		val dataFiles = params.get("data-dir") match {
			case Some(dataDir) => (header, dataDir)
			case _ =>
			  val trainFile = params.get("train-file")
			  val testFile = params.get("test-file")		
			  (trainFile, testFile) match {
					case (Some(tr), Some(tst)) => (header, tr, tst)
					case _ => 
					  System.err.println("Bad usage. Either train or test file is missing.")
					  System.exit(-1)
			  }
		}
      
    // Classification
    val arity = header match {
          case Some(file) => params.get("disc") match {              
              case Some(s) if s matches "(?i)no" => 
                val c = KeelParser.parseHeaderFile(sc, file)
                val categInfo = for(i <- 0 until (c.size - 1) if !c(i).isDefinedAt("min")) yield (i, c(i).size) 
                categInfo.toMap
              case _ => Map.empty[Int, Int]
          }
          case None => Map.empty[Int, Int]
    }
    
    val (algoInfo, classification) = params.get("classifier") match {
      case Some(s) if s matches "(?i)no" => ("", None)
      case Some(s) if s matches "(?i)DT" => 
        (TreeAdapter.algorithmInfo(params), 
              Some(TreeAdapter.classify(_: RDD[LabeledPoint], params, arity)))
      case Some(s) if s matches "(?i)RF" => 
        (RandomForestAdapter.algorithmInfo(params), 
            Some(RandomForestAdapter.classify(_: RDD[LabeledPoint], params, arity))) 
      case Some(s) if s matches "(?i)NB" => (NBadapter.algorithmInfo(params), 
          Some(NBadapter.classify(_: RDD[LabeledPoint], params)))
      case Some(s) if s matches "(?i)LR" => (LRadapter.algorithmInfo(params), 
          Some(LRadapter.classify(_: RDD[LabeledPoint], params)))        
      case _ => (SVMadapter.algorithmInfo(params), // Default: SVM
          Some(SVMadapter.classify(_: RDD[LabeledPoint], params)))              
    }
      
    println("*** Classification info:" + algoInfo)
    
		
		// Perform the experiment
	  MLExperimentUtils.executeExperiment(sc, discretization, featureSelection, classification,
					  (dataFiles, format, dense) , outputDir.get, algoInfo)
		
		sc.stop()
	}
    
}
