/* mouse and cheese 
 * 
 * genius_mouse.c
 * 
 * copied main.c
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
double maximumValueIndex(double q[][4],int state){
	
	int maxIndex = 0;
	for (int i = 1; i < 4; i++){
		if (q[state][i] > q[state][maxIndex]){
			maxIndex = i;
		}
	}
	return maxIndex;
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

int howManyMax(double q[][4],int state,int maxIndex){
	int n=0;
	for (int i = 0; i < 4; i++){
		if (q[state][i] == q[state][maxIndex]){
			n++;
		}
	}
	return n;
}

int chooseTheBest(double q[][4],int state){
	
	int maxIndex = 0;
	//found the maximum value index
	maxIndex = maximumValueIndex(q,state);
	//check if there is another number same as the max value
	int n = howManyMax(q,state,maxIndex);
	//if there is,choose the action randomly between those same max values.
	if(n>1){
		int actionIndex[n];
		int m=0;
		for (int i = 0; i < 4; i++){
			if (q[state][i] == q[state][maxIndex]){
				actionIndex[m++]=i;
			}
		}	
		int act = rand()% n;	
		return actionIndex[act];
	}
	//if there is ony one maximum, index is also the action
	return maxIndex;
}
//sort probability from smallest to highest
void sortArraywithProbality(double dis[][2], int size){
	
	int tempAction;
	double tempProb;
	int min=0;
	for(int i=0 ; i<size; ++i){
		min = i;
		for(int j = i+1 ; j<size; ++j){
			if(dis[j][1] < dis[min][1])
				min = j;	
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
	int n = howManyMax(q,state,indexOut);;
	int size;
	if(n==4){
		size=4;
	}else{
		size = 4-n;
	}
	
	double softmax[size][2]; // {0.qvalue} (action,qvalue)
	int idx = 0;
	double t = 100; 		//give 100 to see close probs.. (tested) :)
 
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
	int probabilty = 1;
	for(int i=0; i<size; i++){
		distribution[i][action] = softmax[i][0]; 
		distribution[i][probabilty] = (exp(softmax[i][1]/t))/qALL; 
	}
	/*for(int i=0; i<size; i++){
		printf("action : %f, ",distribution[i][action]); 
		printf("p : %f\n",distribution[i][1]); 
	}*/
	int selectAction = -1;
	//sort distrubition
	sortArraywithProbality(distribution,size);
	
	//double selectRandom = 0.018;
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
double variance(int a[], int n){ 
    int sum = 0; 
    for (int i = 0; i < n; i++) 
        sum += a[i]; 
    double mean = (double)sum /  
                  (double)n; 
    double sqDiff = 0; 
    for (int i = 0; i < n; i++)  
        sqDiff += (a[i] - mean) *  
                  (a[i] - mean); 
    return sqDiff / n; 
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
		//printf("test %d\n",i);
		int grid[row][column];
		for(int i=0; i<row; ++i){
			for(int j=0; j<column; ++j){
				grid[i][j] = 0; 
			}
		}
		
		//mouse and cheese
		grid[0][0] = 1;
		grid[row-1][column-1] = 100;
			
		for(int i=0; i<row; ++i){
			for(int j=0; j<column; ++j){
				printf("%d ",grid[i][j]);
			}
			printf("\n");
		}
	
		int test;
		int randTest = rand()%10;
		if(randTest>0 && randTest<3){
			test = -1; 
		}else{
			test = 0;
		}
	
		int states = row*column;
		int actionNumber = 4;	//up-down-left-right
	
		//initliaze Q table
		double qtable[states][actionNumber];
		for(int i=0; i<states; ++i){
			for(int j=0; j<actionNumber; ++j){
				qtable[i][j] = 0; 			
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
		double epsilon =  0.35;	//discount factor
		double gamma =  0.9 ; //discount rate
		
		int stop = 0; //prevent infinite
		int state;
		int i,j;
		int steps=0;
		int lastNumbers = 20;
		int stepNumber[lastNumbers];
		int x=0;
	
		int episode = 0;
		double var = 0;
		int endLoop = 1;
	
		while(endLoop){ 
			//printf("*****************episode %d*************\n",episode+1);
			episode++;
			steps=0;
			//initiliaze s;	and grid[i][j] start point
			i=0;
			j=0;
			grid[row-1][column-1] = 100;
			state = i*column+j; 
			
		
			while(grid[row-1][column-1] != 1 ){	
				//printf("*****************steps %d*************\n",steps);
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
					//printf("take action\n:");
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
				int reward = test;
				if(newState == row*column-1){
					reward = 100;
				}
			
				qtable[state][act] = qtable[state][act] + alpha*(reward + gamma*(maximumValue(qtable,newState))- qtable[state][act]);
				state = newState;
				
				epsilon -= epsilon*(0.000001);			//decreased 0.001 percent
				if(epsilon < 0.01){
						epsilon = 0.01;
					}
				steps++;
			}
			stepNumber[x]=steps;
			x++;
			if(x-1 == lastNumbers){
				var = variance(stepNumber,lastNumbers);
				if((var >=0 && (var<0.1))){
					endLoop = 0;
				}
				x = 0;
			}
			stop++; 
		}
		/*for(int i=0; i<row; ++i){
			for(int j=0; j<column; ++j){
				printf("%d ",grid[i][j]);
			}
			printf("\n");
		}*/
		printf("\n****\n");
		for(int i=0; i<states; ++i){
			for(int j=0; j<actionNumber; ++j){
				printf("%.1f ",qtable[i][j]);
			}
			printf("\n");
		}
		for(int i=0; i<lastNumbers; ++i){
			printf("%d	",stepNumber[i]);
			
		}
		printf("\nreward: %d\n",test);
		printf("\n\nepisode: %d\n",episode);
	}
	
	return 0;
}
