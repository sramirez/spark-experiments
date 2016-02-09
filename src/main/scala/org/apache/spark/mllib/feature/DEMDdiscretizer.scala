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

package org.apache.spark.mllib.feature

import scala.collection.mutable
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.Logging
import org.apache.spark.rdd._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import scala.util.Random
import keel.Algorithms.Discretizers.ecpsd._
import scala.collection.mutable.ArrayBuffer

/**
 * Distributed Evolutionary Multivarite Discretizer, a distributed implementation of the algorithm
 * proposed by Ramirez et al. [1].
 * 
 * [1] S. Ramírez-Gallego, S. García, J.M. Benítez, F. Herrera. 
 * Multivariate Discretization Based on Evolutionary Cut Points Selection for Classification. 
 * IEEE Transaction on Cybernetics, in press. doi: 10.1109/TCYB.2015.2410143
 * 
 * @param data RDD of LabeledPoint
 * 
 */
class DEMDdiscretizer private (val data: RDD[LabeledPoint]) extends Serializable with Logging {

  private case class Feature(id: Int, init: Int, end: Int) {
    def size = end - init + 1
  }  
  
  private val isBoundary = (f1: Array[Long], f2: Array[Long]) => {
    (f1, f2).zipped.map(_ + _).filter(_ != 0).size > 1
  }
  
  
  private def updateBigChromosome(newch: Array[Boolean]) = data.context.broadcast(newch)
  
  /**
   * Get information about the attributes before performing discretization.
   * 
   * @param contIndices Indexes to discretize (if not specified, they are calculated).
   * @param nFeatures Total number of input features.
   * @return Indexes of continuous features.
   * 
   */  
  private def processContinuousAttributes(
      contIndices: Option[Seq[Int]], 
      nFeatures: Int) = {
    contIndices match {
      case Some(s) => 
        // Attributes are in range 0..nfeat
        val intersect = (0 until nFeatures).seq.intersect(s)
        require(intersect.size == s.size)
        s.toArray
      case None =>
        (0 until nFeatures).toArray
    }
  }
  
  /**
   * Computes the initial candidate points by feature.
   * 
   * @param points RDD with distinct points by feature ((feature, point), class values).
   * @param firstElements First elements in partitions 
   * @return RDD of candidate points.
   * 
   */
  private def initialThresholds(
      points: RDD[((Int, Float), Array[Long])], 
      firstElements: Array[Option[(Int, Float)]],
      nLabels: Int) = {
    
    val numPartitions = points.partitions.length
    val bcFirsts = points.context.broadcast(firstElements)      

    points.mapPartitionsWithIndex({ (index, it) =>      
      if(it.hasNext) {  
        var ((lastK, lastX), lastFreqs) = it.next()
        var result = Seq.empty[((Int, Float), Array[Long])]
        var accumFreqs = lastFreqs      
        
        for (((k, x), freqs) <- it) {           
          if(k != lastK) {
            // new attribute: add last point from the previous one
            result = ((lastK, lastX), accumFreqs.clone) +: result
            accumFreqs = Array.fill(nLabels)(0L)
          } else if(isBoundary(freqs, lastFreqs)) {
            // new boundary point: midpoint between this point and the previous one
            result = ((lastK, (x + lastX) / 2), accumFreqs.clone) +: result
            accumFreqs = Array.fill(nLabels)(0L)
          }
          
          lastK = k
          lastX = x
          lastFreqs = freqs
          accumFreqs = (accumFreqs, freqs).zipped.map(_ + _)
        }
       
        // Evaluate the last point in this partition with the first one in the next partition
        val lastPoint = if(index < (numPartitions - 1)) {
          bcFirsts.value(index + 1) match {
            case Some((k, x)) => if(k != lastK) lastX else (x + lastX) / 2 
            case None => lastX // last point in the attribute
          }
        }else{
            lastX // last point in the dataset
        }                    
        (((lastK, lastPoint), accumFreqs.clone) +: result).reverse.toIterator
      } else {
        Iterator.empty
      }             
    })
  }
  
  /**
   * Compute the boundary points to evaluate from the input data.
   * 
   * @param nFeatures Number of features.
   * @param contVars Indices from the continuous features to discretize.
   * @param labels2Int Map correspondence between the string labels and their indices.
   * @param classDist Class distribution information
   * @param nLabels Number of distinct clasess.
   * @param isDense Is the dataset dense?
   * @return RDD of tuples (feat. index, point value)
   * 
   */
  private def computeBoundaryPoints(
      nFeatures: Int, 
      contVars: Array[Int], 
      labels2Int: Map[Double, Int],
      classDist: Map[Int, Long], 
      nLabels: Int,
      isDense: Boolean) = {
    
    val bClassDist = data.context.broadcast(classDist)
    val bLabels2Int = data.context.broadcast(labels2Int)
    
    // Generate pairs ((feature, point), class)
    val featureValues = isDense match {
      case true => 
        data.flatMap({ case LabeledPoint(label, dv: DenseVector) =>
            for(i <- 0 until dv.values.length) yield ((i, dv(i).toFloat), label.toFloat)                  
        })
      case false =>      
        data.flatMap{ case LabeledPoint(label, sv: SparseVector) =>
          for(i <- 0 until sv.indices.length) yield ((sv.indices(i), sv.values(i).toFloat), label.toFloat)
        }
    }
    
    // Group elements by feature and point (get distinct points)
    val bContinuousVars = data.context.broadcast(contVars)
    val createCombiner = (v: Float) => {
      val c = Array.fill[Long](nLabels)(0L)
      c(bLabels2Int.value(v)) += 1
      c
    }
    
    val mergeValue = (acc: Array[Long], v: Float) => {
      acc(bLabels2Int.value(v)) += 1
      acc
    }
    
    val mergeCombiners = (acc1: Array[Long], acc2: Array[Long]) => (acc1, acc2).zipped.map(_ + _)    
    val nonzeros = featureValues.combineByKey(createCombiner, mergeValue, mergeCombiners)
      
    // Add zero elements for sparse data
    val zeros = nonzeros
      .map{case ((k, p), v) => (k, v)}
      .reduceByKey{ case (v1, v2) =>  (v1, v2).zipped.map(_ + _)}
      .map{ case (k, v) => 
        val v2 = for(i <- 0 until v.length) yield bClassDist.value(i) - v(i)
        ((k, 0.0F), v2.toArray)
      }.filter{case (_, v) => v.sum > 0}
    val distinctValues = nonzeros.union(zeros)
    
    // Sort these values to perform the boundary points evaluation
    val sortedValues = distinctValues.sortByKey()
          
    // Get the first elements by partition for the boundary points evaluation
    val firstElements = data.context.runJob(sortedValues, { case it =>
      if (it.hasNext) Some(it.next()._1) else None
    }: (Iterator[((Int, Float), Array[Long])]) => Option[(Int, Float)])
      
    // Filter those features selected by the user
    val arr = Array.fill(nFeatures) { false }
    contVars.map(arr(_) = true)
    val barr = data.context.broadcast(arr)
    
    // Get only boundary points from the whole set of distinct values
    val boundaryPairs = initialThresholds(sortedValues, firstElements, nLabels)
      .keys
      .filter({case (a, _) => barr.value(a)})
    boundaryPairs
  }
  
  private def divideChromosome(featBySize: Array[Feature], nBins: Int) = {
    val chunks = Array.fill[List[Feature]](nBins)(List.empty[Feature])
    val itwindow = featBySize.grouped(nBins)
    while(!itwindow.isEmpty){
      val window = itwindow.next().toSeq
      val perm = Random.shuffle(window)
      (0 until perm.length).map(b => chunks(b) = perm(b) +: chunks(b)) 
    }    
    chunks
  }
 
  /**
   * Run DEMD process on input data.
   * 
   * @param contFeaturesIndexes Indices to discretize (if not specified, the algorithm will find them).
   * @param nChr Number of chromosomes per GA.
   * @param userFactor Replication factor for point chunks (if equal to 2, two chunks by partition)
   * @param nGeneticEval Number of chromosome evaluations (GA).
   * @param alpha Weight factor for the fitness function (GA).
   * @param samplingRate Percentage of input data used to evaluate points (GA eval).
   * @param votingThreshold Number of points selected by chunk in the reduce phase.
   * @return A discretization model with the thresholds obtained (by feature).
   * 
   */
  def runAll(
      contFeaturesIndexes: Option[Seq[Int]],
      nChr: Int,
      userFactor: Int,
      nGeneticEval: Int,
      alpha: Float,
      samplingRate: Float,
      votingThreshold: Int) = {
    
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Obtain basic information about the dataset
    val sc = data.context    
    val seed = 518614915
    val labels2Int = data.map(_.label).distinct.collect.zipWithIndex.toMap
    val nLabels = labels2Int.size
    val classDistrib = data.map(d => labels2Int(d.label)).countByValue()
    val nInstances = classDistrib.map(_._2).sum
    val (isDense, nFeatures) = data.first.features match {
      case v: DenseVector => 
        (true, v.size)
      case v: SparseVector =>
        (false, v.size)
    }
            
    val contVars = processContinuousAttributes(contFeaturesIndexes, nFeatures)
    logInfo("Number of continuous attributes: " + contVars.distinct.size)
    logInfo("Total number of attributes: " + nFeatures)      
    if(contVars.isEmpty) logWarning("Discretization aborted. " +
      "No continous attribute in the dataset")
    
    val boundaryPairs = computeBoundaryPoints(nFeatures, contVars, labels2Int, 
        classDistrib.toMap, nLabels, isDense).cache()
      
    // Order the boundary points by size and by id to yield the vector of features
    val nBoundPoints = boundaryPairs.count().toInt
    val featById = boundaryPairs.mapValues(_ => 1).reduceByKey(_ + _).sortByKey().collect()
    
    val featInfo = new Array[Feature](featById.length)
    var iindex = 0
    for(i <- 0 until featById.length) {
      val (id, l) = featById(i)
      featInfo(i) = new Feature(id, iindex, iindex + l - 1)
      iindex += l
    }    
    val featInfoBySize = featInfo.sortBy(-_.size) // sorted in descending order
    
    /** Get in the driver the vector of boundary points and the big chromosome **/
    val boundaryPoints = boundaryPairs.values.collect()
    val bBoundaryPoints = sc.broadcast(boundaryPoints)
    var bigChromosome = Array.fill(nBoundPoints)(true)
    var bBigChromosome = sc.broadcast(bigChromosome)    

    /** Compute numerical variables to control the number of chunks and chromosome partitions **/
    val nPart = data.partitions.size
    val defChPartSize = nBoundPoints / nPart    
    // The largest feature should not be splitted in several parts
    val maxChPartSize = math.max(featInfoBySize(0).size, defChPartSize)
    // Compute the factor of multivariety according to the final size
    val multiVariateFactor = math.max(userFactor, maxChPartSize.toFloat / defChPartSize)
    val chPartSize = multiVariateFactor * defChPartSize
    val nChPart = (nBoundPoints / chPartSize).toInt
    
    /** Print some information about the final configuration **/
    logInfo(s"Total number of boundary points: $nBoundPoints")
    logInfo(s"Total number of data partitions: $nPart")
    logInfo("Number of points for the biggest feature: " + featInfoBySize(0).size)
    logInfo(s"Default number of boundary points by partition: $defChPartSize")    
    logInfo(s"Final Multivariate Factor: $multiVariateFactor")
    logInfo(s"Final number of chromosome partitions: $nChPart")
    logInfo(s"Final size by chromosome partition: $chPartSize")
    
    /** Divide the chromosome into chunks of features, using different combinations (multivar. eval.) **/
    val bChromChunks = sc.broadcast(divideChromosome(featInfoBySize, nChPart))
    val firstChunk = bChromChunks.value(0)
    
    // Defined the correspondence between the partitions and the chromosome partitions
    val partitionOrder = Random.shuffle((0 until nChPart).toVector)
    
    // Start the map partition phase
    val fractions = classDistrib.map({ case (k, v) => k.toDouble -> samplingRate.toDouble})
    val evolvChrom = data.map(lp => lp.label -> lp).sampleByKey(withReplacement = false, fractions, seed)
      .map(_._2).mapPartitionsWithIndex({ (index, it) =>  
      
        val chID = partitionOrder(index % nChPart)
        val chunk = bChromChunks.value(chID)
        val sumsize = chunk.map(_.size).sum
        
        // Create a data matrix for each partition
        val datapart = it.toArray.map({ lp =>
          var sample = new Array[Float](chunk.length + 1)
          (0 until chunk.length).map({i => sample(i) = lp.features(chunk(i).id).toFloat})
          sample(chunk.length) = lp.label.toFloat // class
          sample
        })
        
        if(datapart.length > 0) {
          // Compute the single chromosome and the seq of cut points for each chunk
          val initialChr = new Array[Boolean](sumsize)
          val cutPoints = new Array[Array[Float]](chunk.length)
          var indc = 0
          (0 until chunk.length).map({ indf =>
            val feat = chunk(indf)
            bBigChromosome.value.slice(feat.init, feat.end + 1).copyToArray(initialChr, indc)
            indc = indc + feat.size
            cutPoints(indf) = bBoundaryPoints.value.slice(feat.init, feat.end + 1)
          })
          
          // Applying the GA to each data partition
          val disc = new EMD(datapart, cutPoints, initialChr, alpha, nGeneticEval, classDistrib.size)
          disc.runAlgorithm()

          val fitness = disc.getBestFitness
          val ind = disc.getBestIndividual
          val err = disc.getBestError()
          
          Array((chID, (ind, err))).toIterator      
        } else {
           Iterator.empty 
        }
     }).mapValues({ case (a, e) =>
          val ca = new Array[Int](a.length)
          (0 until a.length).map(i => ca(i) = if(a(i)) 1 else 0)
          (ca, 1, e)
        }).reduceByKey({case ((a1, c1, e1), (a2, c2, e2)) =>
          ((a1, a2).zipped.map(_ + _), c1 + c2, e1)
        }).mapValues({ case (a, c, e) =>
          val sorteda = a.zipWithIndex.filter({case (v, _) => v > 0}).sortBy(-_._1).take((votingThreshold / nChPart).toInt)
          println("sorted first points: " + sorteda.take(5).mkString(" - "))
          val ba =  new Array[Boolean](a.length)
          sorteda.map({case (_, k) => ba(k) = true})
          (ba, a, e)
        })

      // Copy the partial results to the big chromosome
      val result = evolvChrom.collect()
      //println(s"Result for local: $nleval, multiVar: $comb - " + result.sortBy(_._1).mkString("\n"))
      for ((chID, (arr, vot, e)) <- result) {
        val nselect = arr.filter(_ == true).length
        val hist = vot.groupBy(identity).mapValues(_.size).toString()
        val size = arr.length
        var indc = 0; val chunk = bChromChunks.value(chID)
        (0 until chunk.length).map({ indf =>
           val feat = chunk(indf)
           arr.slice(indc, indc + feat.size).copyToArray(bigChromosome, feat.init)
           indc = indc + feat.size
        })              
      }

    // Send a new broadcasted copy of the big chromosome
    bBigChromosome = updateBigChromosome(bigChromosome) // Important to avoid task serialization problem
    
    // Update the full list features with the thresholds calculated
    // all features are supposed to be continuous (without thresholds)
    val thresholds = Array.fill[Array[Float]](nFeatures)(Array.empty[Float]) 
    contVars.map(i => thresholds(i) = Array(Float.PositiveInfinity)) // continuous feature
    bChromChunks.value.map{ lf =>
      lf.map({feat =>  
        val arr = ArrayBuffer.empty[Float]
        (feat.init to feat.end).map(ind => if(bigChromosome(ind)) arr += boundaryPoints(ind))
        if(arr.length > 0) {
          thresholds(feat.id) = arr.toArray
        }
      })
    }
    
    new DiscretizerModel(thresholds)
  }
  
}

object DEMDdiscretizer {

  /**
   * Train a entropy minimization discretizer given an RDD of LabeledPoints.
   * 
   * @param input RDD of LabeledPoints.
   * @param contFeaturesIndexes Indices to discretize (if not specified, the algorithm will find them).
   * @param nChr Number of chromosomes per GA.
   * @param nGeneticEval Number of chromosome evaluations (GA).
   * @param alpha Weight factor for the fitness function (GA).
   * @param multiVariateFactor Replication factor for point chunks (if equal to 2, two chunks by partition).
   * @param samplingRate Percentage of input data used to evaluate points (GA eval).
   * @param votingThreshold Number of points selected by chunk in the reduce phase.
   * 
   * @return A DiscretizerModel with the subsequent thresholds.
   * 
   */
  def train(
      input: RDD[LabeledPoint],
      contFeaturesIndexes: Option[Seq[Int]] = None,
      nChr: Int = 50,
      nGeneticEval: Int = 5000,
      alpha: Float = .7f,
      multiVariateFactor: Int = 1,
      samplingRate: Float = .1f,
      votingThreshold: Int = 100) = {
    new DEMDdiscretizer(input).runAll(contFeaturesIndexes, nChr, 
        multiVariateFactor, nGeneticEval, alpha, samplingRate, votingThreshold)
  }
}
