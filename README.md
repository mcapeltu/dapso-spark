# Automatic distribution of the training process for neural networks across multiple machines
# (Distribución automática en múltiples máquinas del proceso de aprendizaje para redes neuronales)

Repository of Manuel Gachs’s final-year project at the University of Granada.

## Contents

 - **[Scala Spark Distributed Learning](scala-spark-distributed-learning)**: It contains the project code. The `main.scala` file contains the code required to run the tests in Chapter 8 of the thesis book.

 - **[Memoria](memoria)**:Contains the LaTeX file with the text of the undergraduate dissertation.

## Resumen

In recent decades, advances in the field of machine learning (ML)
have led to the widespread adoption of its methods and techniques in the
private sector. These techniques often require repetitive calculations
to be carried out on large volumes of data in order to obtain useful results. The use of
these techniques involves performing operations that need to be completed within
a ‘reasonable’ time. As models become more complex, the time
required to perform the calculations increases considerably. This is why
the parallel and distributed execution of these calculations has become
a key factor in ensuring that the training process is feasible and can be
carried out efficiently.

The aim of this project is to provide a tool that enables the automation
of the distribution and parallelisation, across graphics processing units, of the
neural network learning process, thereby smoothing the learning curve
for parallelisation techniques for those without the technical knowledge
required to achieve high performance in parallel programming.

The initial objectives of this work are as follows: Firstly,
the automatic transformation of sections of sequential code for training
neural networks through calls to specific libraries—also implemented
in this work—that enable parallelisation using GPUs. Secondly, the
construction of an interpreter for selected functions that include
parallelised code for execution on GPUs. Thirdly, to apply knowledge
of the fundamental mathematical theory of compilers and automata to automate
the aforementioned parallelisation process. Finally, to help flatten the learning curve
for parallelisation techniques for people without low-level
knowledge of GPU hardware.
