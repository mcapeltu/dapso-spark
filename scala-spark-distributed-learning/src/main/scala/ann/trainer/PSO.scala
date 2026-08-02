package ann.trainer

import ann.utils.{MSEClass, calculateFitness, calculatePosition, uniform}

import scala.util.Random

/**
 * Secuential particle swarm optimization
 *
 * @param nParticles Number of particles for the swarm optimization
 * @param maxPos Maximum value for the position of a particle
 * @param maxV Maximum value for the velocity of a particle (0.6 maxPos default)
 * @param w Parameter for velocity calculation (1/(2ln(2)) default)
 * @param c1 Parameter for velocity calculation (1/2+ln(2) default)
 * @param c2 Parameter for velocity calculation (1/2+ln(2) default)
 */
class PSO (
 val nParticles: Int,
 val maxPos: Double,
 val maxV: Double = 0,
 val w: Double = 0.721,
 val c1: Double = 1.193,
 val c2: Double = 1.193
) extends Trainer {
  //Random object
  val rand = new Random
  // Maximum velocity
  val vMax: Double = if (maxV==0) 0.6*maxPos else maxV
  // Particles' positions
  var particlesPos = Array.empty[Array[Double]]
  // Swarm's best position
  var bestPos = Array.empty[Double]
  // Swarm's best fitness
  var bestFitness: Double = Double.MaxValue

  // Data to use for the algorithm
  var x: Array[Array[Double]] = _
  var y: Array[Double] = _

  /**
   * Sets the data for the particle swarm optimization and initializes weights
   */
  override def initWeights(x: List[List[Double]], y: List[Double]): Unit = {
    // Reset optimizer state
    particlesPos = Array.empty[Array[Double]]
    bestFitness = Double.PositiveInfinity
    bestPos = Array.empty[Double]
    // Set data
    this.x = x.map(v => v.toArray).toArray
    this.y = y.toArray
    // Initialize weights
    for(i<-0 until nParticles){
      val position = Array.fill(nWeights)(uniform(-maxPos, maxPos, rand))
      val velocity = Array.fill(nWeights)(uniform(-maxPos, maxPos, rand))
      val fit = MSEFunction.compute(this.x, this.y, position, nInputs, nHidden)
      val part_ = position ++ velocity ++ position ++ Array(fit)
      if (fit < bestFitness) {
        bestFitness = fit
        bestPos = position
      }
      particlesPos = particlesPos :+ part_
    }
  }

  /**
   * Fit the data
   */
  override def fit(nIters: Int): Unit = {
    // Check if the net is a classifier
    val isClas = if (MSEFunction == MSEClass) true else false

    // PSO
    for (i <- 0 until nIters) {
      println("Starting iteration: ",i+1)
      for (j <- 0 until nParticles) {
        var part = particlesPos(j)
        part = calculateFitness(x, y, part, nInputs, nHidden, isClas)
        val fit = part(3*nWeights)
        if (fit < bestFitness) {
          bestFitness = fit
          bestPos = part.slice(0, nWeights)
        }
        part = calculatePosition(part, bestPos, nWeights, rand, w, c1, c2, maxV, maxPos)
        particlesPos(j) = part
      }
    }
  }

  /**
   * @return The weights (swarm's best position)
   */
  override def getWeights: Array[Double] = {
    bestPos
  }
}