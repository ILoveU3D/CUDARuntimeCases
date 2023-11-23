#include<stdio.h>
#include<time.h>
#define N 20*65536

void add(int *a,int *b,int *c){
	int t=0;
	while(t<N){
		c[t]=a[t]+b[t];
		t++;
	}
}

int main(){
	int a[N],b[N],c[N];
	//赋值
	for(int i=0;i<N;i++){
		a[i]=i-3;
		b[i]=i/2+1;
	}
	time_t start,end;
	start = time(NULL);
	add(a,b,c);
	end = time(NULL);
	printf("time=%fs\n",difftime(end,start));
}
