import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":
    file = sys.argv[1]
    hw5_data = pd.read_csv(file)

    fig, ax = plt.subplots() 
    ax.plot(hw5_data['year'], hw5_data['days'])  
    ax.set_ylabel('Number of Frozen Days')
    ax.set_xlabel('Year')
    plt.savefig("plot.jpg")

    print("Q3a:")
    X = np.column_stack((np.ones(len(hw5_data)), hw5_data['year']))
    print(X.astype('int64'))

    print("Q3b:")
    Y = np.array(hw5_data['days'])
    print(Y.astype('int64'))

    print("Q3c:")
    xT = X.T
    Z = np.dot(xT, X)
    print(Z.astype('int64'))

    print("Q3d:")
    I = np.linalg.inv(Z)
    print(I)


    print("Q3e:")
    PI = np.dot(I, xT)
    print(PI)


    print("Q3f:")
    B = np.dot(PI, Y)
    print(B)

    xtest = 2022
    ytest = B[0] + B[1] * xtest
    print("Q4: " + str(ytest))

    sign = ""
    if B[1] > 0:
        sign = ">"
        print("Q5a: " + sign)
        print("Q5b: A positive coefficient signifies the mean increase in frozen days for every additional year.")
    elif B[1] == 0:
        sign = "="
        print("Q5a: " + sign)
        print("Q5b: A 0 coefficient signifies that there is no correlation between frozen days and an increase in years")
    else:
        sign = "<"
        print("Q5a: " + sign)
        print("Q5b: A negative coefficient signifies the mean decrease in frozen days for every additional year")

    frozen_free_year = -B[0] / B[1]
    print("Q6a: " + str(frozen_free_year))
    print(f"Q6b: The prediction suggest that the lake will cease to freeze in the year ~" + str(int(frozen_free_year)) + 
        ". While it does make sense given the data and seeing the downtrend in the graph, it is hard to believe that the whole lake will stop freezing in this cold due to linear regression")