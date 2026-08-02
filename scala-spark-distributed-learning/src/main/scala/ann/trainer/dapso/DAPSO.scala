package ann.trainer.dapso

import ann.utils.{ForwardProp, MSE, MSEClass, uniform}
import akka.actor.typed.ActorSystem
import ann.trainer.Trainer
import ann.trainer.dapso.GlobalActor.{StartSendingMessages, StopSendingMessages}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Channel}
import scala.util.Random

/**
 * Asynchronous distributed particle swarm optimization
 *
 * @param nParticles Number of particles for the swarm optimization
 * @param maxPos Maximum value for the position of a particle
 * @param maxV Maximum value for the velocity of a particle (0.6 maxPos default)
 * @param w Parameter for velocity calculation (1/(2ln(2)) default)
 * @param c1 Parameter for velocity calculation (1/2+ln(2) default)
 * @param c2 Parameter for velocity calculation (1/2+ln(2) default)
 * @param rddNum Number of RDDs (4 default)
 * @param batchSize Size for the batches (5 default)
 */
class DAPSO(
  val nParticles: Int,
  val maxPos: Double,
  val maxV: Double = 0,
  val w: Double = 0.721,
  val c1: Double = 1.193,
  val c2: Double = 1.193,
  val rddNum: Int = 4,
  val batchSize: Int = 5
) extends Trainer {
  // Spark configuration
  val sConf: SparkConf = new SparkConf()
    .setAppName("DAPSO")
    .setMaster("local[*]") // Modify this to set the port of the standalone GPU cluster
  val sContext: SparkContext = SparkContext.getOrCreate(sConf)

  // Channel definitions
  val srch = new Channel[BatchPSO]()
  val fuch = new Channel[ListBuffer[Array[Double]]]()

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

    FitnessActor.initialize(srch,fuch,x,y,nInputs,nHidden,sContext, isClas, rddNum)
    GlobalActor.initialize(srch,fuch,nWeights,batchSize,nIters,nParticles,rand,w,c1,c2,vMax, particlesPos)
    val dapsoController = new DAPSOController(this)
    val globalEvalActor = ActorSystem(GlobalActor(dapsoController),"GlobalEvalActor")

    globalEvalActor ! StartSendingMessages
    globalEvalActor ! StopSendingMessages
    Await.result(globalEvalActor.whenTerminated, Duration.Inf )

  }

  /**
   * @return The weights (swarm's best position)
   */
  override def getWeights: Array[Double] = {
    bestPos
  }

  /**
   * Updates the swarm's best position and fitness
   * @param pos Swarm position
   * @param fitness Swarm fitness
   */
  def updateResult(pos: Array[Double], fitness: Double): Unit = {
    bestPos = pos
    bestFitness = fitness
  }
}
