#include"helper_math.h"
#define DIM 512
#define MAX_SPEED 10
#define INF 2e10f
#define RAND(x) ((float)x*rand()/RAND_MAX)

struct Sphere{
	float3 position;
	float3 color;
	float3 velocity;
	float radius;
	__host__ void init(){
		position = make_float3(RAND(DIM)-DIM/2.0,RAND(DIM)-DIM/2.0,RAND(DIM/2)-DIM/4.0);
		color = make_float3(RAND(1),RAND(1),RAND(1));
		velocity = make_float3(RAND(MAX_SPEED)-MAX_SPEED/2.0,RAND(MAX_SPEED)-MAX_SPEED/2.0,0);
		radius = 25;
	}
	__device__ float hit(float2 camera,float3 *r,int time){
		float3 position_current = position + velocity*time;
		float distance = length(make_float2(camera.x-position_current.x,camera.y-position_current.y));
		if(distance < radius){
			float dz = sqrt(radius*radius - distance*distance);
			*r = color * dz / radius * 255;
			return position_current.z + dz; 
		}
		return -INF;
	}
};
