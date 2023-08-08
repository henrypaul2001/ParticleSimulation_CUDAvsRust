# Particle Simulation CUDA vs Rust

Two implementations of a particle system were created. One in Rust and the other in CUDA hosted in a C++ application.

Each simulation creates a batch of particles at 3 locations in a 3D environment. The batch size is the same across both implementations. Every frame, a new batch is created and this is repeated until the target number of batches has been created.

The time taken for each implementation to reach the target batch count is comapared.

Each particle is affected by gravity and air resistance, each having their own colour. The particles move with an initial velocity that fires them above a peice of white paper that is lying flat on a surface. As each particle lands on the paper, the colour of the particle is applied and blended with the colour of the paper. The final result of the paper colouring is then exported as a bitmap image.

Both implementations make use of threading, utilising the threading techniques of each language.

For example, in the CUDA solution, the movement of each particle was simulated on its own thread. The collisions between the particle and the paper were implemented on one thread per collision.

In the Rust solution, scoped threadpools were used to run each step of the simulation. Starting with spray, followed by movement, ending with collision checking.

# Preview

![Preview](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGp2ZG5jNndsM3Z4bWlrMjZ5M3ZjMm91czN0b2Y1anA2emtuYmpzYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6VtG8QLi5vPw9yyVjL/giphy.gif)
