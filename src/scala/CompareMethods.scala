/**
  * Created by wbstclair on 6/26/17.
  */

import org.apache.spark.mllib.linalg.{Vectors => OldVectors}

object CompareMethods {
  def main(args: Array[String]) {

    val numVectors = 1000
    val vectorLength = 10
    val numComponents = 3
    val numPartitions = 3
    val alpha = 0.01

    //var comparisonData:Array[]

    var data:Array[Vector] = new Array[Vector](numVectors)

    //val randomGenerator = scala.util.Random

    // Fill in random vectors.
    for (vectorNumber <- 0 to (numVectors - 1)) {
      val randomArray = Seq.fill(vectorLength)(Random.nextDouble)
      data(vectorNumber) = Vectors.dense(randomArray)
    }

    val conf = new SparkConf().setAppName("CompareMethods").setMaster("local")
    val sc = new SparkContext(conf)

    val dataRDD = sc.parallelize(data, numPartitions)


    // Compute PCA model and IPCA model and time each.

    val df = dataRDD.toDF("features")

    // Begin IPCA timer
    val t0 = System.nanoTime()

    val ipca = new IPCA()
      .setInputCol("features")
      .setOutputCol("ipca_features")
      .setK(numComponents)
      .setAlpha(alpha)

    val ipcaModel = ipca.fit(df)

    val ipcaFeatures = ipcaModel.transform(df).select("ipca_features")

    // End IPCA timer
    val t1 = System.nanoTime()

    // Begin PCA timer
    val t2 = System.nanoTime()

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pca_features")
      .setK(numComponents)

    val pcaModel = pca.fit(df)

    val pcaFeatures = pcaModel.transform(df).select("pca_features")

    val t3 = System.nanoTime()

    val timeComparison = (t1-t0, t3-t2)

    println(timeComparison)

  }
}