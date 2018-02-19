__kernel void hello(__global const int *a)
{	
	int gid = get_global_id(0);
	result[gid] = a[gid] + 1;
    
}
