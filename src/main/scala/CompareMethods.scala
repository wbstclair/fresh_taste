
/**
  * This object compares the runtime of IPCA and PCA models in Spark as vector history increases.
  * Created by wbstclair on 6/26/17.
  */

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.Row
import scala.util.Random.nextDouble


object CompareMethods {
  def main(args: Array[String]) {

    val numDataPoints = 20
    //val numReplications = 3
    var numVectorsSoFar = 0

    val alpha = 0.01
    val numComponents = 3
    var hasBeenInitiated = false
    val vectorLength = 10
    val numPartitions = 5
    val numVectors = 800

    val ipca = new IPCA()
      .setInputCol("features")
      .setOutputCol("ipca_features")
      .setK(numComponents)
      .setAlpha(alpha)

    var timeDataIPCA:Array[Long] = new Array[Long](numDataPoints+1)
    var timeDataPCA:Array[Long] = new Array[Long](numDataPoints+1)


    for (pointNumber <- 0 to numDataPoints) {

      numVectorsSoFar += numVectors

      //var comparisonData:Array[]

      var data:Array[Vector] = new Array[Vector](numVectors)
      var dataComplete:Array[Vector] = new Array[Vector](numVectorsSoFar)

      val randomGenerator = scala.util.Random

      // Fill in random vectors.
      for (vectorNumber <- 0 to (numVectors - 1)) {
        val randomArray = Array.fill(vectorLength)(randomGenerator.nextDouble)
        data(vectorNumber) = Vectors.dense(randomArray)
      }
      for (vectorNumber <- 0 to (numVectorsSoFar - 1)) {
        val randomArray = Array.fill(vectorLength)(randomGenerator.nextDouble)
        dataComplete(vectorNumber) = Vectors.dense(randomArray)
      }


      //val conf = new SparkConf().setAppName("CompareMethods").setMaster("local")
      //val sc = new SparkContext(conf)

      val dataRDD = sc.parallelize(data, numPartitions)
      val dataCompleteRDD = sc.parallelize(dataComplete, numPartitions)

      val mat = new RowMatrix(dataRDD.map(OldVectors.fromML))
      val pc = mat.computePrincipalComponents(3)
      val expected = mat.multiply(pc).rows.map(_.asML)

      val df = dataRDD.zip(expected).toDF("features","expected")

      val mat2 = new RowMatrix(dataCompleteRDD.map(OldVectors.fromML))
      val pc2 = mat2.computePrincipalComponents(3)
      val expected2 = mat2.multiply(pc).rows.map(_.asML)
      val dfComplete = dataCompleteRDD.zip(expected2).toDF("features","expected")


      // Compute PCA model and IPCA model and time each.

      // Begin IPCA timer
      val t0 = System.nanoTime()

      val ipcaModel = hasBeenInitiated match {
        case false => ipca.fit(dfComplete)
        case _ => ipca.incrementFit(df)(sc)
      }

      hasBeenInitiated = true

      val ipcaFeatures = ipcaModel.transform(df).select("ipca_features")

      // End IPCA timer
      val t1 = System.nanoTime()

      // Since ipca is created only once, exclude PCA creation from timer.
      val pca = new PCA()
        .setInputCol("features")
        .setOutputCol("pca_features")
        .setK(numComponents)

      // Begin PCA timer
      val t2 = System.nanoTime()

      val pcaModel = pca.fit(dfComplete)

      val pcaFeatures = pcaModel.transform(df).select("pca_features")

      val t3 = System.nanoTime()

      val timeComparison = ((t1-t0)/1000000, (t3-t2)/1000000)

      timeDataIPCA(pointNumber) = (t1-t0)/1000000
      timeDataPCA(pointNumber) = (t3-t2)/1000000

      println(numVectorsSoFar.toString)

      println(timeComparison._1.toString + " " + timeComparison._2.toString)

      println((timeComparison._1 - timeComparison._2).toString)
    }

  }



}