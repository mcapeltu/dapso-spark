ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "scala-spark-distributed-learning",

    scalacOptions += "-Xasync",

    libraryDependencies ++= Seq(
      "com.typesafe.akka" %% "akka-actor" % "2.8.0",
      "com.typesafe.akka" %% "akka-actor-typed" % "2.8.0",

      "org.scala-lang.modules" %% "scala-async" % "1.0.1",
      "org.scala-lang" % "scala-reflect" % scalaVersion.value % Provided,

      "org.apache.spark" %% "spark-core" % "3.4.0",
      "org.apache.spark" %% "spark-sql" % "3.4.0",

      "org.antlr" % "antlr4-runtime" % "4.13.1",
      "org.antlr" % "stringtemplate" % "3.2"
    )
  )
Compile / run / fork := true