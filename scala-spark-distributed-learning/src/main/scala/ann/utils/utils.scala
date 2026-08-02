package ann

import java.io.{FileWriter, PrintWriter}
import scala.collection.mutable.ListBuffer
import scala.math.signum
import scala.util.Random
import scala.util.Try
package object utils {
  /**
   * Reads a CSV file with data
   *
   * @param fileName File path
   * @param nRows    Number of rows to keep from the file
   * @return A list with the rows of data
   */
 /* def readCSV(fileName: String, nRows: Int): List[List[Double]] = {
    import scala.io.Source

    // Get data from file
    val data = Source.fromFile(fileName)

    // Transform the data into a list of rows
    val rows = data.getLines().drop(1).filter(line => {
      val cols = line.split(",").map(_.trim)
      cols.forall(_.nonEmpty)
    }).take(nRows).map(line => {
      val cols = line.split(",").map(_.trim).toList.map(_.toDouble)
      cols
    }).toList

    data.close()
    rows
  }
*/

  def readCSV(path: String, nRows: Int): List[List[Double]] = {
    import scala.io.Source
    val source = Source.fromFile(path)

    try {
      val lines = source.getLines().filter(_.trim.nonEmpty)

      val parsedRows = lines.map { line =>
        val fields = line
          .split("[;,]")
          .map(_.trim)
          .filter(_.nonEmpty)

        fields.map(value => Try(value.toDouble))
      }

      parsedRows
        // Omite cabeceras u otras filas no completamente numéricas.
        .filter(row => row.nonEmpty && row.forall(_.isSuccess))
        .take(nRows)
        .map(row => row.map(_.get).toList)
        .toList

    } finally {
      source.close()
    }
  }
  /**
   * Write the weights into a CSV file
   * @param filename Name for the CSV file
   * @param weights Array of net weights
   */
  def writeCSV(filename: String, weights: Array[Double]): Unit = {
    val file = new FileWriter(filename+".csv", true)
    val writer = new PrintWriter(file)
    val wString = weights.mkString(";")
    writer.println(wString)
    writer.close()
    file.close()
  }

  /**
   * Separates a list of data into the characteristics and the outcome
   *
   * @param data      List of data
   * @param setPosNeg Whether to interpolate {0, 1} to {-1, 1} for the outcome values. (For classification)
   * @return The characteristics data and its outcome
   */
  def separateXY(data: List[List[Double]], setPosNeg: Boolean = false): (List[List[Double]], List[Double]) = {
    val rowMax = data.head.size - 1
    val xData = data.map(_.take(rowMax))
    var yData = data.map(_(rowMax))

    // Interpolate outcome values
    if (setPosNeg) {
      yData = yData.map {
        case 0.0 => -1.0
        case 1.0 => 1.0
        case _ => throw new IllegalArgumentException("Invalid outcome value")
      }
    }

    (xData, yData)
  }

  /**
   * Generates a number in the interval (a,b)
   *
   * @param a The minimum value of the interval
   * @param b The maximum value of the interval
   * @param rand Random object
   * @return A number between a and b
   */
  def uniform(a: Double, b: Double, rand: Random): Double = a + (b - a) * rand.nextDouble()

  /**
   * Calculates the fitness of the Net
   */
  def calculateFitness(x: Array[Array[Double]], y: Array[Double], weights: Array[Double], nInput: Int, nHidden: Int, isClas: Boolean): Array[Double] = {
    val nWeights: Int = nHidden * (nInput + 1)

    // Return if there are no weights
    if (weights == null) {
      return Array.empty[Double]
    }

    // Calculate fitness
    val bestLocalFit = weights(3 * nWeights)
    val weightSlice = weights.slice(0, nWeights)
    var fit: Double = 0
    if (isClas) {
      fit = calculateAccuracy(x, y, weightSlice, nInput, nHidden)
    } else {
      fit = MSEReg.compute(x, y, weightSlice, nInput, nHidden)
    }

    if (fit < bestLocalFit) {
      weights(3 * nWeights) = fit
      for (k <- 0 until nWeights) {
        weights(2 * nWeights + k) = weightSlice(k)
      }
    }
    weights
  }

  private def accuracy(yTrue: Array[Double], yPred: Array[Double]): Double = {
    require(yTrue.length == yPred.length, "Both lists must have the same length")

    val totalSamples = yTrue.length
    val correctPredictions = yTrue.zip(yPred).count { case (trueLabel, predictedLabel) => trueLabel == predictedLabel }

    correctPredictions.toDouble / totalSamples
  }

  /**
   * Calculates the accuracy of the net
   */
  def calculateAccuracy(x: Array[Array[Double]], y: Array[Double], weights: Array[Double], nInput: Int, nHidden: Int): Double = {
    var yPred: Array[Double] = Array.emptyDoubleArray
    for (i <- x.indices) {
      val pr = signum(ForwardPropClass.compute(x(i), weights, nInput, nHidden))
      yPred = yPred :+ pr
    }
    1.0 - accuracy(y, yPred)
  }

  /**
   * Calculates the position of a particle in PSO
   */
  def calculatePosition(particle: Array[Double], mpg: Array[Double], N: Int, rand: Random, W: Double, c1: Double, c2: Double, maxV: Double, maxPos: Double): Array[Double] = {
    val velocities = particle.slice(N, 2 * N)
    val mpl = particle.slice(2 * N, 3 * N)

    for (k <- 0 until N) {
      // Calculate velocity
      velocities(k) = W * velocities(k) + uniform(0, c1, rand) * (mpl(k) - particle(k)) + uniform(0, c2, rand) * (mpg(k) - particle(k))
      if (velocities(k) > maxV) {
        velocities(k) = maxV
      } else if (velocities(k) < -maxV) {
        velocities(k) = -maxV
      }
      // Calculate position
      particle(k) = particle(k) + velocities(k)
      if (particle(k) > maxPos){
        particle(k) = maxPos
      } else if (particle(k) < -maxPos){
        particle(k) = -maxPos
      }
      // Update velocity values
      particle(N + k) = velocities(k)
    }
    particle
  }

  /**
   * Transforms a 2D array into a ListBuffer of an array
   */
  def toListBuffer(arrayArray: Array[Array[Double]]): ListBuffer[Array[Double]] = {
    val listBuffer: ListBuffer[Array[Double]] = ListBuffer.empty[Array[Double]]

    for (subArray <- arrayArray) {
      val list: List[Double] = subArray.toList
      listBuffer += list.toArray
    }
    listBuffer
  }
}
