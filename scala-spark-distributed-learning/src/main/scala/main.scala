import ann.trainer.PSO
import ann.trainer.dapso.DAPSO
import ann.{Classifier, Regressor}

import java.io.{FileWriter, PrintWriter}

object main {
  def main(args: Array[String]): Unit = {
    // testIterationsClas()
   testReg()
    //testData()
  }

  private def testIterationsClas(): Unit = {
    // Constants
    val nRows = 456
    val cI = 30
    val cH = 70
    val nParts = 100
    val maxPos = 1.0

    val net1 = new Classifier(cI, cH)
    net1.setTrainingData("cancer_train.csv", nRows)
    net1.setTestData("cancer_test.csv", nRows)
    val dapso1 = new DAPSO(nParts,maxPos,batchSize = 10,rddNum = 4)
    net1.setTrainer(dapso1)

    val net2 = new Classifier(cI, cH)
    net2.setTrainingData("cancer_train.csv", nRows)
    net2.setTestData("cancer_test.csv", nRows)
    val dapso2 = new DAPSO(nParts,maxPos,batchSize = 10,rddNum = 10)
    net2.setTrainer(dapso2)

    val netPSO = new Classifier(cI, cH)
    netPSO.setTrainingData("cancer_train.csv", nRows)
    netPSO.setTestData("cancer_test.csv", nRows)
    val pso1 = new PSO(nParts,maxPos)
    netPSO.setTrainer(pso1)

    val resultsPSOTime = new FileWriter(s"resultados_iters_pso_t.txt", true)
    val resultsPSOWriterT = new PrintWriter(resultsPSOTime)
    val resultsPSOConf = new FileWriter(s"resultados_iters_pso_conf.txt", true)
    val resultsPSOWriterC = new PrintWriter(resultsPSOConf)
    val resultsDAPSO1Time = new FileWriter(s"resultados_iters_dapso1_t.txt", true)
    val resultsDAPSO1WriterT = new PrintWriter(resultsDAPSO1Time)
    val resultsDAPSO1Conf = new FileWriter(s"resultados_iters_dapso1_conf.txt", true)
    val resultsDAPSO1WriterC = new PrintWriter(resultsDAPSO1Conf)
    val resultsDAPSO2Time = new FileWriter(s"resultados_iters_dapso2_t.txt", true)
    val resultsDAPSO2WriterT = new PrintWriter(resultsDAPSO2Time)
    val resultsDAPSO2Conf = new FileWriter(s"resultados_iters_dapso2_conf.txt", true)
    val resultsDAPSO2WriterC = new PrintWriter(resultsDAPSO2Conf)

    println("Iterations Test")

    for (i <- 1 until 11 ) {
      println("Iterations:"+10*i)
      val rPsoTime = netPSO.fit(10*i)
      val psoConf = netPSO.confusionMatrix()
      resultsPSOWriterT.println("("+10*i,rPsoTime+")")
      resultsPSOWriterC.println(psoConf)
      val rDapso1Time = net1.fit(10*i)
      val dapso1Conf = net1.confusionMatrix()
      resultsDAPSO1WriterT.println("("+10*i,rDapso1Time+")")
      resultsDAPSO1WriterC.println(dapso1Conf)
      val rDapso2Time = net2.fit(10*i)
      val dapso2Conf = net2.confusionMatrix()
      resultsDAPSO2WriterT.println("("+10*i,rDapso2Time+")")
      resultsDAPSO2WriterC.println(dapso2Conf)
    }

    resultsPSOWriterT.close()
    resultsPSOWriterC.close()
    resultsDAPSO1WriterT.close()
    resultsDAPSO1WriterC.close()
    resultsDAPSO2WriterT.close()
    resultsDAPSO2WriterC.close()
  }

  private def testReg(): Unit = {
    // Constants
    val nRows = 3000
    val rI = 15
    val rH = 30
    val nParts = 100
    val maxPos = 1.0
    val W = 1.0
    val c_1 = 0.8
    val c_2 = 0.2

    val net1 = new Regressor(rI, rH)
    net1.setTrainingData("energia_ugr_train.csv", nRows)
    net1.setTestData("energia_ugr_test.csv", nRows)
    val dapso1 = new DAPSO(nParts,maxPos,batchSize = 10,rddNum = 4,w = W,c1 = c_1,c2 = c_2)
    net1.setTrainer(dapso1)

    val net2 = new Regressor(rI, rH)
    net2.setTrainingData("energia_ugr_train.csv", nRows)
    net2.setTestData("energia_ugr_test.csv", nRows)
    val pso1 = new PSO(nParts,maxPos,w = W,c1 = c_1,c2 = c_2)
    net2.setTrainer(pso1)

    val resultsPSOTime = new FileWriter(s"resultados_reg_pso_t.txt", true)
    val resultsPSOWriterT = new PrintWriter(resultsPSOTime)
    val resultsPSOConf = new FileWriter(s"resultados_reg_pso_mse.txt", true)
    val resultsPSOWriterC = new PrintWriter(resultsPSOConf)
    val resultsDAPSO1Time = new FileWriter(s"resultados_reg_dapso_t.txt", true)
    val resultsDAPSO1WriterT = new PrintWriter(resultsDAPSO1Time)
    val resultsDAPSO1Conf = new FileWriter(s"resultados_reg_dapso_mse.txt", true)
    val resultsDAPSO1WriterC = new PrintWriter(resultsDAPSO1Conf)

    println("Regression Test")

    for (i <- 1 until 11 ) {
      println("Iterations:"+10*i)
      val rPsoTime = net2.fit(10*i)
      val psoMSE = net2.MSEFit()
      resultsPSOWriterT.println("("+10*i,rPsoTime+")")
      resultsPSOWriterC.println("("+10*i,psoMSE+")")
      val rDapsoTime = net1.fit(10*i)
      val dapsoMSE = net1.MSEFit()
      resultsDAPSO1WriterT.println("("+10*i,rDapsoTime+")")
      resultsDAPSO1WriterC.println("("+10*i,dapsoMSE+")")
    }

    resultsPSOWriterT.close()
    resultsPSOWriterC.close()
    resultsDAPSO1WriterT.close()
    resultsDAPSO1WriterC.close()
  }

  private def testData(): Unit = {
    // Constants
    val nRowsBase = 500
    val cI = 23
    val cH = 46
    val nParts = 50
    val maxPos = 1.0

    val net1 = new Classifier(cI, cH)
    val dapso1 = new DAPSO(nParts,maxPos,batchSize = 10,rddNum = 4)
    net1.setTrainer(dapso1)
    val netPSO = new Classifier(cI, cH)
    val pso1 = new PSO(nParts,maxPos)
    netPSO.setTrainer(pso1)

    val resultsPSO = new FileWriter(s"resultados_pso.txt", true)
    val resultsPSOWriter = new PrintWriter(resultsPSO)
    val resultsDAPSO = new FileWriter(s"resultados_dapso.txt", true)
    val resultsDAPSOWriter = new PrintWriter(resultsDAPSO)

    println("Size Test")

    for (i <- 1 until 11 ) {
      net1.setTrainingData("covid_training.csv", nRowsBase*i)
      net1.setTestData("covid_test.csv", nRowsBase*i)
      netPSO.setTrainingData("covid_training.csv", nRowsBase*i)
      netPSO.setTestData("covid_test.csv", nRowsBase*i)
      val rPsoTime = netPSO.fit(50)
      resultsPSOWriter.println("(",nRowsBase*i,rPsoTime,")")
      val rDapsoTime = net1.fit(50)
      resultsDAPSOWriter.println("(",nRowsBase*i,rDapsoTime,")")
    }

    resultsPSOWriter.close()
    resultsDAPSOWriter.close()
  }
}
