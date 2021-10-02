# Implementation of steepest descent algorithm, with 
# -- constant step size (SteepestDescent), and
# -- step size chosen via Armijo's rule (SteepestDescentArmijo)
#
# The built-in minimization problem is
#           min   0.5*x'*Q*x - b'*x + 1
#
# Run this as (for example):
# >> include("steep-hw2.jl");
# >> x0 = [10;10];
# >> alpha = 0.1;
# >> x = SteepestDescent(x0,alpha);
# >> c1 = 1e-3;
# >> y = SteepestDescentArmijo(x0, c1);
#


using Printf
using LinearAlgebra

# matrix and vector used in quadratic form. 
# defined here, because they are used in both F(x), and DF(x)
Q = [10 -3; -3 1];
b = [1;1];

# set parameters here, for all gradient descent algorithms
tol = 1e-6;     # tolerance on norm of gradient
MaxIter = 10000;  # maximum number of iterations of gradient descent

function square(n)
   return n*n
end

# define function
function F(x)
   val = 0.5*x'*Q*x - b'*x + 1
   return val
end

# define gradient
function DF(x)
   grad = Q*x-b;
   return grad
end


#
# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    alpha = step size. Constant, in this algorithm.
# output: 
#    x = final point
#
function SteepestDescent(x0,alpha)

   # setup for steepest descent
   x = x0;
   successflag = false;

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       Fgrad = DF(x);
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, Fval)
          successflag = true;
          break;
       end
       # perform steepest descent step
       x = x - alpha*Fgrad;

   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return x;
end



#
# steepest descent algorithm, with Armijo's rule for backtracking
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    c1 = slope, in Armijo's rule.
# output: 
#    x = final point
#
function SteepestDescentArmijo(x0, c1)

   # parameters for Armijo's rule
   alpha0 = 10.0;    # initial value of alpha, to try in backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps

   # setup for steepest descent
   x = x0;
   successflag = false;   

   # perform steepest descent iterations
   for iter = 1:MaxIter

      alpha = alpha0;
      Fval = F(x);
      Fgrad = DF(x);

      # check if norm of gradient is small enough
      if sqrt(Fgrad'*Fgrad) < tol
         @printf("Converged after %d iterations, function value %f\n", iter, Fval)
         successflag = true;
         break;
      end

      # perform line search
      for k = 1:MaxBacktrack
         x_try = x - alpha*Fgrad;
         Fval_try = F(x_try);
         if (Fval_try > Fval - c1*alpha *Fgrad'Fgrad)
            alpha = alpha * eta;
         else
            Fval = Fval_try;
            x = x_try;
            break;
         end
      end

      # print how we're doing, every 10 iterations
      if (iter%10==0)
         @printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Fval)
      end
   end

   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end

   return x;
end

function MomentumDescent(x0,alpha)

   # setup for steepest descent
   x = x0;
   successflag = false;
   eval = eigvals(Q);
   val1 = eval[1];
   val2 = eval[2];
   #al = square(2/(sqrt(val1)+sqrt(val2)));
   #be = square((sqrt(val1)-sqrt(val2))/(sqrt(val1)+sqrt(val2)));
   al = 0.2
   be = 1
   D = DF(x);

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       Fgrad = DF(x);
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, Fval)
          successflag = true;
          break;
       end
       x = x-al*D;
       Fgrad = DF(x);
       D = Fgrad+be*D;
       # perform steepest descent step

   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return x;
end
