from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed, randrange
from scipy import stats



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(y, yh):
     return np.square(y - yh).mean()
def r2score(y,yh):    
    return (1 - np.sum(np.square(y-yh))/np.sum(np.square(y- np.mean(yh))))
def cofidentint(XXinv,y,yh,beta):    
    N=np.size(y,0)

    p=np.size(XXinv,0)

    confInt=np.zeros((p,2))
    sigma2=np.sum(np.square(y-yh))/(N-p-1)
    #95% CI
    for i in range(p):
        confInt[i,0]=beta[i]-1.645*np.sqrt(abs(XXinv[i,i]))*sigma2
        confInt[i,1]=beta[i]+1.645*np.sqrt(abs(XXinv[i,i]))*sigma2
    return confInt
def constructX(x,y,polyOrder):
    
    vectorSize=np.size(y,0)
    if polyOrder==3:       
        xMatrix = np.c_[np.ones((vectorSize,1)), x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3]
    elif polyOrder==2:       
        xMatrix = np.c_[np.ones((vectorSize,1)), x, y,x**2,y**2,x*y]
    elif polyOrder==4:
        xMatrix= np.c_[np.ones((vectorSize,1)), x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4]
    elif polyOrder==5:
        xMatrix=np.c_[np.ones((vectorSize,1)), x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]

    return xMatrix    
def OSLregression(xVector,yVector,zVector,polyOrder):
    
    vectorSize=np.size(yVector,0)
    #transform back to a matrix
    xMatrix=constructX(xVector,yVector,polyOrder)
    #pseudo inversion using SVD   
    XXinv=np.linalg.inv(xMatrix.T.dot(xMatrix))
    beta = XXinv.dot(xMatrix.T).dot(zVector)
    #print(beta)
    #zPredict=xMatrix.dot(beta)
    #zPredictReshape=np.reshape(zPredict,(10,10))
    return beta,XXinv

def Ridgeregression(xVector,yVector,zVector,polyOrder,lambda1):
    
    vectorSize=np.size(yVector,0)

    #transform back to a matrix
    #
    xMatrix=constructX(xVector,yVector,polyOrder)
    size1=np.size(xMatrix,1)
    I=np.identity(size1)
    #pseudo inversion using SVD
    XXinv=np.linalg.pinv(xMatrix.T.dot(xMatrix)+ lambda1*I)
    beta = XXinv.dot(xMatrix.T).dot(zVector)

    return beta,XXinv

def computeZpredict(xVector,yVector,beta,polyOrder):
    #
    xMatrix=constructX(xVector,yVector,polyOrder)
    zPredict=xMatrix.dot(beta)
    return zPredict

def plotTheSurface(x,y,z):
# Plot the surface.
    fig = plt.figure(figsize=(20,10))
    ax = fig.gca(projection='3d')


    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Z')
    #plt.show();
    return fig

def train_test_split(dataset, split):
    # Split a dataset into a train and test set
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        #print(index)
        train.append(dataset_copy.pop(index))
    #tranform into np.array
    train=np.array(train)
    dataset_copy=np.array(dataset_copy)
    return train, dataset_copy



    
def k_folds_CV(dataset, nfolds):
    # Split a dataset into k folds
    splitedDataset = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / nfolds)
    for i in range(nfolds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        splitedDataset.append(fold)
    return splitedDataset

def trainSetindex(indeces,testSetindex):
    #given indeces of the test set, find the indeces of the train set
    size=np.size(indeces)
    mask = np.ones(size, dtype=bool)
    mask[testSetindex] = False
    return indeces[mask]

def computeBiasandVar(zPredictmatrix,zVector):
    n=np.size(zPredictmatrix,0)
    m=np.size(zPredictmatrix,1)
    meanzPredictmatrix=np.mean(zPredictmatrix,1)
    bias=(np.sum(np.square(zVector-meanzPredictmatrix)))/n
    newMatrix=np.zeros((n,m))
    for i in range(m-1):
        newMatrix[:,i]=zPredictmatrix[:,i]-meanzPredictmatrix
    newMatrix=np.square(newMatrix)
    newMatrix=np.mean(newMatrix)
    var=np.sum(newMatrix,0)
    return bias,var


def olsModel(polynom_oders,xVector,yVector,zVector,numberOfFolds,folds,indeces):
#OLS Model
    sizeVector=np.size(zVector)
    numOfoders=len(polynom_oders)
    statsMatrix=np.zeros((2,numberOfFolds,numOfoders))
    zPredictmatrix=np.zeros((sizeVector,numberOfFolds,numOfoders))
    #21 = number of term in polynomial order 5
    betaMatrix=np.zeros((21,numberOfFolds,numOfoders))
    for j,order in enumerate(polynom_oders):
        for i in range(numberOfFolds):
            #print(i)
    
            test1=folds[i]
            train1= trainSetindex(indeces,test1)
            beta,XXinv=OSLregression(xVector[train1],yVector[train1],zVector[train1],order)
            betaMatrix[0:len(beta),i,j]=beta
        #zPredict=computeZpredict(xVector[test1],yVector[test1],beta,3)
            zPredictmatrix[:,i,j]=computeZpredict(xVector,yVector,beta,order)
            statsMatrix[0,i,j]=MSE(zVector[test1],zPredictmatrix[test1,i,j])
            statsMatrix[1,i,j]=r2score(zVector[test1],zPredictmatrix[test1,i,j])
        #print(len(beta))
        print('STATS of MSE for polynom order {} is:'.format(str(order)))
        print(stats.describe(statsMatrix[0,:,j]))
        print('STATS of R2score for polynom order {} is:'.format(str(order)))
        print(stats.describe(statsMatrix[1,:,j]))
        print('\n')
    return zPredictmatrix,statsMatrix,betaMatrix

def ridge_regress(lambda_values,polynom_oders,xVector,yVector,zVector,numberOfFolds,folds,indeces):
#Ridge Model
    sizeVector=np.size(zVector)
    numOfLambdas=len(lambda_values)
    
    numOfoders=len(polynom_oders)
    statsMatrix=np.zeros((2,numberOfFolds,numOfLambdas,numOfoders))
    zPredictmatrix=np.zeros((sizeVector,numberOfFolds,numOfLambdas,numOfoders))
    betaMatrix=np.zeros((21,numberOfFolds,numOfLambdas,numOfoders))
    for j,order in enumerate(polynom_oders):
        for i in range(numberOfFolds):
            #print(i)
            test1=folds[i]
            train1= trainSetindex(indeces,test1)
            for  h,lbd in enumerate(lambda_values):           
                beta,XXinv=Ridgeregression(xVector[train1],yVector[train1],zVector[train1],order,lbd)
        #zPredict=computeZpredict(xVector[test1],yVector[test1],beta,3)
                betaMatrix[0:len(beta),i,h,j]=beta
                zPredictmatrix[:,i,h,j]=computeZpredict(xVector,yVector,beta,order)
                statsMatrix[0,i,h,j]=MSE(zVector[test1],zPredictmatrix[test1,i,h,j])
                statsMatrix[1,i,h,j]=r2score(zVector[test1],zPredictmatrix[test1,i,h,j])
    return zPredictmatrix,statsMatrix,betaMatrix

#Lasso Model
def lassoRegress(lambda_values,polynom_oders,xVector,yVector,zVector,numberOfFolds,folds,indeces):
    from sklearn.linear_model import Lasso 
    from sklearn.metrics import mean_squared_error, r2_score


    numOfLambdas=len(lambda_values)
    sizeVector=np.size(zVector)
    numOfoders=len(polynom_oders)
    statsMatrix=np.zeros((2,numberOfFolds,numOfLambdas,numOfoders))
    zPredictmatrix=np.zeros((sizeVector,numberOfFolds,numOfLambdas,numOfoders))
    betaMatrix=np.zeros((21,numberOfFolds,numOfLambdas,numOfoders))
    for j,order in enumerate(polynom_oders):
        XMatrix=constructX(xVector,yVector,order)
        for i in range(numberOfFolds):
            #print(i)
            test1=folds[i]
            train1= trainSetindex(indeces,test1)
            for  h,lbd in enumerate(lambda_values):           
                #beta,XXinv=Ridgeregression(xVector[train1],yVector[train1],zVector[train1],order,lbd)
        #zPredict=computeZpredict(xVector[test1],yVector[test1],beta,3)
            
                lasso=Lasso(lbd,max_iter=1000,fit_intercept=True)
                lasso.fit(XMatrix[train1,:],zVector[train1])
                zPredictmatrix[:,i,h,j]=lasso.predict(XMatrix)
                beta=lasso.coef_
                betaMatrix[0:len(beta),i,h,j]=beta
                betaMatrix[0,i,h,j]=lasso.intercept_
                #zPredictmatrix[:,i,h,j]=computeZpredict(xVector,yVector,beta,order)
                statsMatrix[0,i,h,j]=mean_squared_error(zVector[test1],zPredictmatrix[test1,i,h,j])
                statsMatrix[1,i,h,j]=r2_score(zVector[test1],zPredictmatrix[test1,i,h,j])
    return zPredictmatrix,statsMatrix,betaMatrix

