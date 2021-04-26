The general structure of my test interface is

Initialization of the following:
    - memory location of input operands and parameters
    - memory location for result
    - performance and correctness memory locations
    - set of problems to test on
        * ex: sizes of matrices

Loop over elements of the problem space
    - create test buffers and update parameters according to the sizes of the problem
    - run N tests of kernel and store the best time
    - compare output of kernel to reference and store diff
    - print results (timing and diff)
    - free the matrices


    
