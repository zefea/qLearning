/*
 * mouseAndcheese.c
 * 21.10.2020
 * 
 * 
 * Training several times (test1, test2...)
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


//0: up, 1:down, 2:left, 3:right
int row,column;
int noProblem(int action,int i, int j){
	
	if(action == 0){
		if(i == 0)return 0;
	}else if(action == 1){
		if(i == row-1) return 0;
	}else if(action ==2){
		if(j == 0) return 0;
	}else if(action == 3){
		if(j == column-1) return 0;
	}
	return 1;
}

double maximumValue(double q[][4],int state){
	
	int maxIndex = 0;
	for (int i = 1; i < 4; i++){
		if (q[state][i] > q[state][maxIndex]){
			maxIndex = i;
		}
	}
	
	return q[state][maxIndex];
}

double maximumValueIndex(double q[][4],int state){
	
	int maxIndex = 0;
	for (int i = 1; i < 4; i++){
		if (q[state][i] > q[state][maxIndex]){
			maxIndex = i;
		}
	}
	
	return maxIndex;
}

int chooseTheBest(double q[][4],int state){
		//printf("lets cheee\n");
	int i;
	int maxIndex = 0;
	
	//found the maximum value index
	for (i = 1; i < 4; i++){
			//printf("lets what now\n %d--%d\n ",state,maxIndex);
		if (q[state][i] > q[state][maxIndex]){
			maxIndex = i;
			//	printf("letddds \n");
		}
	}
	//check if there is another number same as the max value
	int n=0;
	for (i = 0; i < 4; i++){
		if (q[state][i] == q[state][maxIndex]){
			n++;
			//	printf("lets gg\n");
		}
	}
	//if there is,choose the action randomly between those same max values.
	if(n>1){
		int actionIndex[n];
		int m=0;
		for (i = 0; i < 4; i++){
			if (q[state][i] == q[state][maxIndex]){
				actionIndex[m++]=i;
				//	printf("letseeee \n");
			}
		}	
		//printf("lets \n");
		// srand((unsigned int)time(NULL));
		int act = rand()% n;	
		return actionIndex[act];
	}
	
	//if there is ony one maximum, index is also the action
	return maxIndex;
}

double calculateSD(int data[],int n) {
    double sum = 0.0, mean, SD = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        sum += data[i];
    }
    mean = sum / n;
    for (i = 0; i < n; ++i)
        SD += pow(data[i] - mean, 2);
    return sqrt(SD / n);
}

double variance(int a[], int n) 
{ 
    // Compute mean (average of elements) 
    int sum = 0; 
    for (int i = 0; i < n; i++) 
        sum += a[i]; 
    double mean = (double)sum /  
                  (double)n; 
  
    // Compute sum squared  
    // differences with mean. 
    double sqDiff = 0; 
    for (int i = 0; i < n; i++)  
        sqDiff += (a[i] - mean) *  
                  (a[i] - mean); 
    return sqDiff / n; 
} 

void sortArraywithProbality(double dis[][2], int size){
	
	int tempAction;
	double tempProb;
	
	
	int min=0;
	for(int i=0 ; i<size; ++i){
		min = i;
		for(int j = i+1 ; j<size; ++j){
			if(dis[j][1] < dis[min][1]){
				min = j;
			}
		}
		if(min != i){
			tempAction = dis[min][0];
			tempProb = dis[min][1];
			
			dis[min][0] =  dis[i][0] ;
			dis[min][1] = 	dis[i][1];		
			dis[i][0] =  tempAction;
			dis[i][1] = tempProb;			
		}
	}
	
}




int randomWithSoftmaxSelection(double q[][4],int state){
	
	int indexOut = maximumValueIndex(q,state);
	int n=0;
	for (int i = 0; i < 4; i++){
		if (q[state][i] == q[state][indexOut]){
			n++;
		}
	}
	int size;
	if(n==4){
		size=4;
	}else{
		size = 4-n;
	}
	
	double softmax[size][2]; // {0.qvalue} (action,qvalue)
	int idx = 0;
	double t = 2.5;
	
	if(size == 4){
		for (int i = 0; i < 4; i++){			
			softmax[idx][0] = i;
			softmax[idx++][1] = q[state][i];	
		}
	}else{
		for (int i = 0; i < 4; i++){
			if (q[state][i] != q[state][indexOut]){
				softmax[idx][0] = i;
				softmax[idx++][1] = q[state][i];
			}
		}
	}
	double qALL;
	double distribution[size][2];
	for(int i=0; i<size; i++){
		qALL += exp(softmax[i][1]/t);
	}
	int action = 0;
	for(int i=0; i<size; i++){
		distribution[i][action] = softmax[i][0]; 
		distribution[i][1] = (exp(softmax[i][1]/t))/qALL; 
	}
	int selectAction = -1;
	//sort distrubition
	sortArraywithProbality(distribution,size);
	
	double selectRandom = rand()/(double)RAND_MAX;
	double probMax = distribution[0][1];
	double probMin = 0;
	
	for(int i=0; i<size; ++i){
		if( selectRandom > probMin && selectRandom < probMax){
			selectAction = distribution[i][0];
			break; 
		}else{
			probMin = probMax;
			probMax = distribution[i+1][1] + probMin;		
		}
	}
	return selectAction;
}


int main(int argc, char **argv)
{
	//initiliaze a grid m*n
	srand(time(0)); 
	
	printf("Please enter the number of rows and colums.Example Input: 4 5\n");
	scanf("%d %d",&row,&column);
	
	int num;
	printf("enter \n: ");
	scanf("%d",&num);
	
	for(int i=0; i<num ; ++i){
		
		printf("\n***************************Test %d***************************\n",i+1);
		int grid[row][column];
		for(int i=0; i<row; ++i){
			for(int j=0; j<column; ++j){
				grid[i][j] = 0; 
			}
		}
			
		//mouse and cheese
		//grid[0][0] = 1;
		//grid[row-1][column-1] = 100;
			
		int startX= 0;
		int startY= 0;
		int goalX = row-1;
		int goalY = column-1;
		
		grid[startX][startY] = 1;
		grid[goalX][goalY] = 100;	
			
		/*for(int i=0; i<row; ++i){
			for(int j=0; j<column; ++j){
				printf("%d ",grid[i][j]);
			}
			printf("\n");
		}*/
		

		int states = row*column;
		int actionNumber = 4;	//up-down-left-right
		
		//initliaze Q table
		double qtable[states][actionNumber];
		for(int i=0; i<states; ++i){
			for(int j=0; j<actionNumber; ++j){
				qtable[i][j] = 0; 			//maybe later random numbers. not only 0's.
			}
		}
		/*printf("\n****\n");
		for(int i=0; i<states; ++i){
			for(int j=0; j<actionNumber; ++j){
				printf("%.1f ",qtable[i][j]);
			}
			printf("\n");
		}*/
		
		//define alfa, gama,epsilon
		double alpha = 0.1; //the learning rate
		double epsilon =  0.45;	//discount factor
		double gamma =  0.9 ; //discount rate
		
		int state;
		int i,j;
		int steps=0;
		int lastNumbers = 20;
		int stepNumber[lastNumbers];
		int x=0;
		
		int episode = 0;
		double var = 0;
		int loop = 1;
		
		while(loop){ 
			episode++;
			//printf("--------------------episode %d-------------",episode);
			//initiliaze s;	and grid[i][j] start point
			i = startX;
			j = startY;
			state = i*column+j; 
			grid[startX][startY] = 1;
			grid[goalX][goalY] = 100;
			steps=0;
			
			//find a path to the cheese
			while(grid[row-1][column-1] != 1 ){	
				int act;	
				double r = rand()/(double)RAND_MAX;
				
				if(r < epsilon){ //epsilon with merak
					//random selection
					//act = rand()%4;	
					act = randomWithSoftmaxSelection(qtable,state);	
				}else {
					act = chooseTheBest(qtable,state);	

				} 
				//take the action 
				if(noProblem(act,i,j)){
					if(act == 0){
						grid[i][j] = 0;
						grid[i-1][j] = 1;
						i--;
					}else if(act == 1){
						grid[i][j] = 0;
						grid[i+1][j] = 1;
						i++;
					}else if(act ==2){
						grid[i][j] = 0;
						grid[i][j-1] = 1;
						j--;
					}else if(act == 3){
						grid[i][j] = 0;
						grid[i][j+1] = 1;
						j++;
					}
				}
				

				//update Q table
				int newState = i*column+j;
				int reward = 0;
				if(newState == row*column-1){
					reward = 100;
				}
				
				qtable[state][act] = qtable[state][act] + alpha*(reward + gamma*(maximumValue(qtable,newState))- qtable[state][act]);
				
			
				state = newState;
				epsilon -= epsilon*(0.0000001);			//decreased 0.001 percent
				if(epsilon < 0.01){
						epsilon = 0.01;
					}
				steps++;
			}
			//printf("aa");
				stepNumber[x]=steps;
				x++;
				if(x-1 == lastNumbers){
					var = variance(stepNumber,lastNumbers);
					if(var >=0 && (var<0.1)){
						loop = 0;
					}
					x = 0;
				}
			
		}
		/*for(int i=0; i<row; ++i){
			for(int j=0; j<column; ++j){
				printf("%d ",grid[i][j]);
			}
			printf("\n");
		}*/
		printf("\n****\n");
		/*for(int i=0; i<states; ++i){
			for(int j=0; j<actionNumber; ++j){
				printf("%.1f ",qtable[i][j]);
			}
			printf("\n");
		}*/
		for(int i=0; i<lastNumbers; ++i){
			printf("%d	",stepNumber[i]);
			
		}
		
		printf("\n\nepisode: %d\n",episode);
	}
	
	return 0;
}

