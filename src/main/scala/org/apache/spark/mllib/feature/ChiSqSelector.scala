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

import scala.collection.mutable.ArrayBuilder

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

/**
 * :: Experimental ::
 * Creates a ChiSquared feature selector.
 * @param numTopFeatures number of features that selector will select
 *                       (ordered by statistic value descending)
 */
@Experimental
class ChiSqSelector (val numTopFeatures: Int) {

  /**
   * Returns a ChiSquared feature selector.
   *
   * @param data an `RDD[LabeledPoint]` containing the labeled dataset with categorical features.
   *             Real-valued features will be treated as categorical for each distinct value.
   *             Apply feature discretizer before using this function.
   */
  def fit(data: RDD[LabeledPoint]): SelectorModel = {
    val indices = Statistics.chiSqTest(data)
      .zipWithIndex.sortBy { case (res, _) => -res.statistic }
      .take(numTopFeatures)
      .map { case (_, indices) => indices }
      .sorted
    new SelectorModel(indices)
  }
}
