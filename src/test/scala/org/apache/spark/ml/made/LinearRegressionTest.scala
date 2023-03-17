package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.made.LinearRegressionTest._weightsBig
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.DataFrame


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.0001
  val weightsBigDelta = 0.01
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val weights:Vector  = LinearRegressionTest._weights

  lazy val dataBig: DataFrame = LinearRegressionTest._dataBig
  lazy val weightsBig: Vector = LinearRegressionTest._weightsBig

  "Model" should "makes right predictions with pretrained weights" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val predictions: Array[Double] = model.transform(data).collect().map(_.getAs[Double]("prediction"))

    implicit val encoder : Encoder[Double] = ExpressionEncoder()
    val labels = data.select(data("label").as[Double]).collect()

    predictions.length should be(labels.length)

    predictions(0) should be (labels(0) +- delta)
    predictions(1) should be (labels(1) +- delta)
  }

  "Model" should "be work fine after fit" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
    ))

    val model = pipeline.fit(data)

    val predictions: Array[Double] = model.transform(data).collect().map(_.getAs[Double]("prediction"))
    implicit val encoder: Encoder[Double] = ExpressionEncoder()
    val labels = data.select(data("label").as[Double]).collect()

    predictions.length should be(labels.length)

    predictions(0) should be(labels(0) +- delta)
    predictions(1) should be(labels(1) +- delta)
  }

  "Model" should "works after save|load" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val loadedModel: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val predictions: Array[Double] = loadedModel.transform(data).collect().map(_.getAs[Double]("prediction"))
    implicit val encoder: Encoder[Double] = ExpressionEncoder()
    val labels = data.select(data("label").as[Double]).collect()

    predictions.length should be(labels.length)

    predictions(0) should be(labels(0) +- delta)
    predictions(1) should be(labels(1) +- delta)

  }

  "Model" should "be work fine after fit on real data" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
    ))

    val model = pipeline.fit(dataBig)

    val predictions: Array[Double] = model.transform(dataBig).collect().map(_.getAs[Double]("prediction"))
    implicit val encoder: Encoder[Double] = ExpressionEncoder()
    val labels = dataBig.select(dataBig("label").as[Double]).collect()

    predictions.length should be(labels.length)

    val fittedWeights = model.stages(0).asInstanceOf[LinearRegressionModel].weights

    fittedWeights.size should be(_weightsBig.size)

    // в других тестах мы уже проверили, что модель может давать правильные предсказания.
    // сейчас, так как данные специально "зашумлены" для большей правдоподобности,
    // сравним только обученные веса, с теми коэффициентами, с помощью которых генерировались данные для обучения.
    // они должны быть равны с определенной точностью (так как все таки был наложен шум).
    for ((real, fitted) <- _weightsBig.toArray zip fittedWeights.toArray) {
      fitted should be(real +- weightsBigDelta)
    }
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _weights = Vectors.dense(1.5, 0.75)
  lazy val points = Seq((Vectors.dense(0.0), 1.5), (Vectors.dense(2.0), 3.0))
  lazy val _data = {
    import sqlc.implicits._
    points.toDF("features", "label")
  }

  val n_samples = 100000
  lazy val X: DenseMatrix[Double] = DenseMatrix.rand[Double](n_samples, 3)
  lazy val ones = DenseVector.ones[Double](size=n_samples)
  lazy val _weightsBig = Vectors.dense(2.5,1.5, 0.3, -0.7)
  lazy val real_y: DenseVector[Double] =  DenseMatrix.horzcat(ones.asDenseMatrix.t, X) * _weightsBig.asBreeze
  lazy val noisy_y = real_y + 0.01 * DenseVector.rand[Double](n_samples)
  lazy val dataBig: DenseMatrix[Double] = DenseMatrix.horzcat(X, noisy_y.asDenseMatrix.t)

  import breeze.linalg._
  import spark.sqlContext.implicits._

  val _dataBig = dataBig(*, ::).iterator
    .map(x => (Vectors.dense(x(0), x(1), x(2)), x(3)))
    .toSeq
    .toDF("features", "label")

  spark.sparkContext.setLogLevel("ERROR")
}