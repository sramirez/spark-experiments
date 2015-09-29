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
import org.apache.spark.broadcast.Broadcast

/**
 * Entropy minimization discretizer based on Minimum Description Length Principle (MDLP)
 * proposed by Fayyad and Irani in 1993 [1].
 * 
 * [1] Fayyad, U., & Irani, K. (1993). 
 * "Multi-interval discretization of continuous-valued attributes for classification learning."
 *
 * @param data RDD of LabeledPoint
 * 
 */
class DEMDdiscretizer private (val data: RDD[LabeledPoint]) extends Serializable with Logging {

  private case class Feature(id: Int, size: Int)
  private val isBoundary = (f1: Array[Long], f2: Array[Long]) => {
    (f1, f2).zipped.map(_ + _).filter(_ != 0).size > 1
  }
  
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
  
  private def divideChromosome(featBySize: Array[Feature], nComb: Int, nBins: Int) = {
    val chunks = Array.fill[List[Feature]](nComb, nBins)(List.empty[Feature])
    val itwindow = featBySize.grouped(nBins)
    while(!itwindow.isEmpty){
      val window = itwindow.next().toSeq
      (0 until nComb).map({ c =>
        val perm = Random.shuffle(window)
        (0 until perm.length).map(b => chunks(c)(b) = perm(b) +: chunks(c)(b))        
      })
    }    
    chunks
  }
  
  private def evaluateChromosomes(
      chunks: Array[List[Feature]],
      samplingRate: Float,
      EMDalgorithms: Array[EMD],
      partitionOrder: Array[Int]) = {
    
    data.sample(false, samplingRate, 347217811).mapPartitionsWithIndex({ (index, it) =>  
        val chID = partitionOrder(index % partitionOrder.length)
        val emd = EMDalgorithms(chID)
        if(emd.needsToEval()) {
          val chunk = chunks(chID)
          val sumsize = chunk.map(_.size).sum
        
          // Create a data matrix for each partition
          val datapart = it.toArray.map({ lp =>
            var sample = new Array[Float](chunk.length + 1)
            (0 until chunk.length).map({i => sample(i) = lp.features(chunk(i).id).toFloat})
            sample(chunk.length) = lp.label.toFloat // class
            sample
          })          
          
          val listEval = emd.evalPopulation(datapart)          
          Array((chID, listEval)).toIterator          
        } else {
          Iterator.empty
        }        
      }).reduceByKey({ case (ep1, ep2) =>
        (ep1, ep2).zipped.map({case (e1, e2) => e1.add(e2)})
        ep1
    })
  }
  
  private def updateBroadcast(newch: Array[Array[Boolean]]) = data.context.broadcast(newch)
 
  /**
   * Run the distributed ECSPD discretizer on input data.
   * 
   * @param contFeat Indices to discretize (if not specified, the algorithm try to figure it out).
   * @param elementsByPart Maximum number of elements to keep in each partition.
   * @param maxBins Maximum number of thresholds per feature.
   * @return A discretization model with the thresholds by feature.
   * 
   */
  def runAll(
      contFeat: Option[Seq[Int]],
      nChr: Int,
      userFactor: Int,
      nGeneticEval: Int,
      alpha: Float,
      nMultiVariateEval: Int,
      samplingRate: Float,
      votingThreshold: Float) = {
    
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Obtain basic information about the dataset
    val sc = data.context    
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
            
    val contVars = processContinuousAttributes(contFeat, nFeatures)
    logInfo("Number of continuous attributes: " + contVars.distinct.size)
    logInfo("Total number of attributes: " + nFeatures)      
    if(contVars.isEmpty) logWarning("Discretization aborted. " +
      "No continous attribute in the dataset")
    
    val boundaryPairs = computeBoundaryPoints(nFeatures, contVars, labels2Int, 
        classDistrib.toMap, nLabels, isDense).groupByKey().cache()
        
    val pointsMatrix = new Array[Array[Float]](nFeatures)
    val chromoMatrix = new Array[Array[Boolean]](nFeatures)
    val boundaryMatrix = boundaryPairs.mapValues(_.toArray.sorted).collect()
    for((i, a) <- boundaryMatrix) { pointsMatrix(i) = a; chromoMatrix(i) = Array.fill[Boolean](a.size)(true)}
    var bChromoMatrix = sc.broadcast(chromoMatrix)
      
    // Order the boundary points by size and by id to yield the vector of features
    /*val nBoundPoints = boundaryPairs.count().toInt
    val featById = boundaryPairs.mapValues(_ => 1).reduceByKey(_ + _).sortByKey().collect()
    
    val featInfo = new Array[Feature](featById.length)
    
   
    var iindex = 0
    for(i <- 0 until featById.length) {
      val (id, l) = featById(i)
      featInfo(i) = new Feature(id, iindex, iindex + l - 1)
      iindex += l
    } */   
    val featInfoBySize = boundaryPairs.map({case (id, it) => new Feature(id, it.size)})
      .collect().sortBy(-_.size) // sorted in descending order
    val nBoundPoints = featInfoBySize.map(_.size).sum
    
    /** Get in the driver the vector of boundary points and the big chromosome **/
    /*val boundaryPoints = boundaryPairs.values.collect()
    val bBoundaryPoints = sc.broadcast(boundaryPoints)
    val bigChromosome = Array.fill(nBoundPoints)(true)
    var bBigChromosome = sc.broadcast(bigChromosome)*/

    /** Compute numerical variables to control the number of chunks and chromosome partitions **/
    val nPart = data.partitions.size
    val defChPartSize = nBoundPoints / nPart    
    // The largest feature should not be splitted in several parts
    val maxChPartSize = math.max(featInfoBySize(0).size, defChPartSize)
    // Compute the factor of multivariety according to the final size
    val multiVariateFactor = math.max(userFactor, math.ceil(maxChPartSize.toFloat / defChPartSize).toInt)
    val chPartSize = multiVariateFactor * defChPartSize
    val nChPart = nBoundPoints / chPartSize
    
    /** Print some information about the final configuration **/
    println(s"Total number of boundary points: $nBoundPoints")
    println(s"Total number of data partitions: $nPart")
    println(s"Final Multivariate Factor: $multiVariateFactor")
    println("Maximum feature size: " + featInfoBySize(0).size)
    println(s"Default number of boundary points by partition: $defChPartSize")
    println(s"Final number of chromosome partitions: $nChPart")
    println(s"Final size by chromosome partition: $chPartSize")
    
    /** Divide the chromosome into chunks of features, using different combinations (multivar. eval.) **/
    val chromChunks = divideChromosome(featInfoBySize, nMultiVariateEval, nChPart)
    
    val firstChunk = chromChunks(0)
    println("first chromChunks: " + firstChunk.map(_.map(_.id)).mkString(","))
    println("Size by chunk: " + firstChunk.map(_.size).mkString(","))
    println("Sum by chunk: " + firstChunk.map(_.map(_.size).sum).mkString(","))
    
    for(comb <- 0 until nMultiVariateEval) {
      
      val partitionOrder = Random.shuffle((0 until nChPart).toVector).toArray
      val EMDalgorithms = new Array[EMD](nChPart)
      (0 until nChPart).map { chID =>
        val chunk = chromChunks(comb)(chID)
        val sumsize = chunk.map(_.size).sum
        var indc = 0        
        val initialChr = Array.fill[Boolean](sumsize)(true)
        chunk.map({ f =>
          chromoMatrix(f.id).copyToArray(initialChr, indc)
          indc = indc + f.size
        })
        val cutPoints = chunk.map(_.id).map(pointsMatrix).toArray
        EMDalgorithms(chID) = new EMD(cutPoints, initialChr, alpha, nGeneticEval, classDistrib.size)
        EMDalgorithms(chID).initPopulation()
      }    
    
      /** Evaluate initial solutions **/
      var evalPointsByChunk = evaluateChromosomes(chromChunks(comb), samplingRate, EMDalgorithms, partitionOrder).collect()      
      for((i, eps) <- evalPointsByChunk) {
         EMDalgorithms(i).setFitness(eps)
      }
      
      // Start the subsequent evaluations. All the EMD algorithms are synchronized in the evaluation phase.
      do{            
        val newPopulations = (0 until EMDalgorithms.length).map{ chID =>
          val emd = EMDalgorithms(chID)
          if(!emd.isFinished()) Some(emd.crossover()) else None             
        }
        
        evalPointsByChunk = evaluateChromosomes(chromChunks(comb), samplingRate, EMDalgorithms, partitionOrder).collect()      
        for((i, eps) <- evalPointsByChunk) {
           EMDalgorithms(i).setFitness(eps)
        }
        
        (0 until EMDalgorithms.length).map{ chID =>
          newPopulations(chID) match {
            case Some(pop) => EMDalgorithms(chID).newPopulation(pop)
            case _ => /** Do nothing **/
          }                     
        }      
        
        evalPointsByChunk = evaluateChromosomes(chromChunks(comb), samplingRate, EMDalgorithms, partitionOrder).collect()      
        for((i, eps) <- evalPointsByChunk) {
           EMDalgorithms(i).setFitness(eps)
        }
      } while(EMDalgorithms.filter(!_.isFinished()).length > 0)
        
        
      (0 until nChPart).map({ chID =>
        val ch = EMDalgorithms(chID).getBest(); var acc = 0; var indi = 0; val fitness = ch.getFitness;
        println(s"Best chromosome: $fitness")
        for(f <- chromChunks(comb)(chID)) {
          chromoMatrix(f.id) = ch.getIndividual.slice(acc, acc + f.size)
          acc = acc + f.size
        }
      })
        
      // Send a new broadcasted copy of the big chromosome
      bChromoMatrix = updateBroadcast(chromoMatrix) // Important to avoid task serialization problem
    }
    
    // Update the full list features with the thresholds calculated
    val thresholds = Array.fill[Array[Float]](nFeatures)(Array.empty[Float])
    (0 until chromoMatrix.length).map{ i =>
      val ths = chromoMatrix(i)
      if (ths != null) {
        thresholds(i) = if(ths.length > 0) {
          val arr = ArrayBuffer.empty[Float]
          (0 until ths.length).map { j => if(ths(j)) arr += pointsMatrix(i)(j)}
          arr.toArray
        } else {
          Array(Float.PositiveInfinity) 
        }
      }
    }
    
    new DiscretizerModel(thresholds)
  }
}

object DEMDdiscretizer {

  /**
   * Train a entropy minimization discretizer given an RDD of LabeledPoints.
   * 
   * @param input RDD of LabeledPoint's.
   * @param continuousFeaturesIndexes Indexes of features to be discretized. 
   * If it is not provided, the algorithm selects those features with more than 
   * 256 (byte range) distinct values.
   * @param maxBins Maximum number of thresholds to select per feature.
   * @param maxByPart Maximum number of elements by partition.
   * @return A DiscretizerModel with the subsequent thresholds.
   * 
   */
  def train(
      input: RDD[LabeledPoint],
      continuousFeaturesIndexes: Option[Seq[Int]] = None,
      nChr: Int = 50,
      nGeneticEval: Int = 5000,
      alpha: Float = .7f,
      multiVariateFactor: Int = 1,
      nMultiVariateEval: Int = 2,
      samplingRate: Float = .1f,
      votingThreshold: Float = .25f) = {
    new DEMDdiscretizer(input).runAll(continuousFeaturesIndexes, nChr, 
        multiVariateFactor, nGeneticEval, alpha, nMultiVariateEval, samplingRate, votingThreshold)
  }
}
