package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    val df = spark.read.parquet("src/main/resources/dfTest4")

    println("STAGE 1 : récupérer les mots des textes")
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val wordsData : DataFrame = tokenizer.transform(df)

    println("STAGE 2 : retirer les stop words")
    val stopWordRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")
      .setStopWords(Array("the","a","http","i","me","to","what","in","rt"))

    //val wordRemover : DataFrame = stopWordRemover.transform(wordsData)

    println("STAGE 3 : computer la partie TF")
    val countVectorizer = new CountVectorizer()
      .setInputCol(stopWordRemover.getOutputCol)
      .setOutputCol("tf")

    //val dfTest = countVectorizer.fit(wordRemover)

    println("STAGE 4 : computer la partie IDF")
    val idf = new IDF().setInputCol(countVectorizer.getOutputCol).setOutputCol("tfidf")

    println("STAGE 5 : convertir country2 en quantité numérique")
    val indexercountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    println("STAGE 6 : convertir currency2 en quantité numérique")
    val indexercurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    println("STAGE 7 et 8 : One-Hot encoder ces 2 catégories")
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array(indexercountry.getOutputCol, indexercurrency.getOutputCol))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    println("STAGE 9 : assembler toutes les features en un unique vecteur")
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal",
        "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    println("STAGE 10 : créer / instancier le modèle de classification")
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    println(s"Création du pipeline :")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordRemover, countVectorizer, idf, indexercountry,
        indexercurrency, encoder, assembler, lr))

    val sets = df.randomSplit(Array[Double](0.9,0.1))
    val training = sets(0)
    val test = sets(1)

    val model1 = pipeline.fit(training)

    println(s"Entraînement et sauvegarde du modèle 1 sans réglage des hyper-paramètres : ")
    println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

    val dfWithSimplePredictions = model1.transform(test)

    println(s"Affichage des résultats du modèle 1 : ")
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()


    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions")
    val f1SimpleScore = evaluator.evaluate(dfWithSimplePredictions)
    println(f1SimpleScore)

    println(s"A présent, réglage des hyperparamètres et création d'un Cross-Validated modèle (cvModel) :")
    val elNetCvValues = getLogScale(1e-8, 10, 2)
    val minDFCvValues = Array(55.0, 75.0, 95.0)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, elNetCvValues)
      .addGrid(countVectorizer.minDF, minDFCvValues)
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setSeed(18)

    val cvModel = trainValidationSplit.fit(training)
    val dfWithPredictions = cvModel.transform(test)
    println(s"Affichage des résultats obtenus avec réglage des hyperparamètres :")
    dfWithPredictions.groupBy("final_status", "predictions").count.show()
    val f1SimpleScoreCV = cvModel.getEvaluator.evaluate(dfWithPredictions)
    println(f1SimpleScoreCV)

    println(s"Sauvegarde du modèle cross-validated")
    cvModel.write.overwrite.save("src/main/model/LogisticRegression")


  }
  private def getLogScale(from : Double, to : Double, logStep : Double):Array[Double] = {
    val initArray = new ArrayBuffer[Double]()
    initArray += from
    incrementLogScale(initArray, to, logStep).toArray
  }

  @scala.annotation.tailrec
  private def incrementLogScale(current : ArrayBuffer[Double], to : Double, logStep : Double):ArrayBuffer[Double] = {
    if (current.last < to){
      current += current.last*math.pow(10, logStep)
      incrementLogScale(current, to, logStep)
    }
    else{
      current
    }
  }
}
