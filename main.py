import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

def matrixMultiplication(A, B, blockSize):
    #We will perform matrix multiplication of matrix A and B and store the result in matrix C
    start_time = time.time()
    # get the size of the matrices
    n = len(A)
    m = len(B[0])
    C = np.zeros((n, m)) #initialize squared matrix with zeros at first
    for i in range(0, n, blockSize):
        for j in range(0, len(B), blockSize):
            for k in range(0, len(B)):
                for ii in range(i, min(i + blockSize, n)):
                    for jj in range(j, min(j + blockSize, m)):
                        C[ii][jj] += A[ii][k] * B[k][jj]
                #This line multiplies a block from matrix A, starting at row i and column k, with a block from matrix B, starting at row k and column j, and adds the result to the corresponding block in matrix C, starting at row i and column j.
    end_time = time.time()            
    return end_time - start_time

matrixSizes = [32,64,128]
blockSizes = [1,2,4,8,16,32,64]

results = [] 

for matrixSize in matrixSizes:
    for blockSize in blockSizes:
        A = np.random.rand(matrixSize,matrixSize)
        B = np.random.rand(matrixSize,matrixSize)
        time_taken =matrixMultiplication(A,B,blockSize)
        print("Matrix size is : "+str(matrixSize)+"  ,Block size is : "+str(blockSize)+"  time taken is : "+str(time_taken)+" seconds")
        results.append({"matrixSize": matrixSize, "blockSize": blockSize, "time_taken": time_taken})

results_df = pd.DataFrame(results)

for matrixSize in matrixSizes:
    df = results_df[results_df["matrixSize"] == matrixSize]
    plt.plot(df["blockSize"], df["time_taken"], label=f"Matrix size: {matrixSize}")

plt.xlabel("Block size")
plt.ylabel("Time taken (s)")
plt.legend()
plt.show()