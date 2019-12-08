package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /***************
      * Load data in a DataFrame
      ******/

    val df: DataFrame = spark
      .read
      .option("header", value = true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", value = true) // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    println(s"Affichage du DataFrame sous forme de tableau :")
    df.show()
    println(s"Affichage du shéma du DataFrame : ")
    df.printSchema()

    println(s"Assignement de type Int aux colonnes concernées :")
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    println(s"Affichage du shéma du DataFrame Casted : ")
    dfCasted.printSchema()

    println(s"Affichage de la description statistique des colonnes de type Int : ")
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    println(s"Etude des différentes colonnes en vue d'en proposer un nettoyage pertinent :")
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)

    dfCasted.select("deadline").dropDuplicates.show()

    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)

    dfCasted.select("goal", "final_status").show(30)

    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    println(s"On retire la colonne disable_communication en créant un nouveau DataFrame :")
    val df2: DataFrame = dfCasted.drop("disable_communication")

    df2.printSchema()

    println(s"On retire les fuites du futur :")
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    dfNoFutur.printSchema()

    //df.filter($"country" === "False")
      //.groupBy("currency")
      //.count
      //.orderBy($"count".desc)
      //.show(50)

    //Après avoir implémenté des udf sur notre Jupyter Notebook pour s'y exercer, on privilégie ici les fonctions de Spark
    println(s"En utilisant les fonctions de Spark, on nettoie les colonnes currency et country : ")
    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")

    //Changement des dates avec to_timestamp
    val dfTest2 : DataFrame = dfCountry
      .withColumn("deadline", to_timestamp($"deadline"))
      .withColumn("launched_at", to_timestamp($"launched_at"))
      .withColumn("created_at", to_timestamp($"created_at"))

    //Création des colonnes days_campaign et hours_prepa pour connaître les durées précises
    val dfTest3 : DataFrame = dfTest2
      .withColumn("days_campaign", datediff($"deadline",$"launched_at"))
      .withColumn("hours_prepa", unix_timestamp($"launched_at")-unix_timestamp($"created_at"))

    //Ultimes nettoyages des colonnes avec remplacement des valeurs Null, changement de la casse et conservation uniquement
    //des lignes où le final_status est correctement rempli afin de faciliter la classification à venir.
    val dfTest4 : DataFrame = dfTest3
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords",lower($"keywords"))
      .withColumn("text",concat_ws(" ",$"name",$"desc",$"keywords"))
      .withColumn("hours_prepa", round($"hours_prepa"/3600,3))
      .withColumn("hours_prepa",when($"hours_prepa".isNull, -1).otherwise($"hours_prepa"))
      .withColumn("days_campaign",when($"days_campaign".isNull, -1).otherwise($"days_campaign"))
      .withColumn("goal",when($"goal".isNull, -1).otherwise($"goal"))
      .withColumn("country2",when($"country2".isNull, "unknown").otherwise($"country2"))
      .withColumn("currency2",when($"currency2".isNull, "unknown").otherwise($"currency2"))
      .filter($"final_status"===0 || $"final_status" ===1)
      .drop("name","desc","keywords","deadline","created_at","launched_at")

    println(s"DataFrame nettoyé:")
    dfTest4.show()

    //Export du Dataframe ainsi créé.
    dfTest4.write.mode("overwrite").parquet("src/main/resources/dfTest4")

  }
}
