__kernel void hello(__global  int *a, __global int *b)
{
	int gid = get_global_id(0);
	b[gid] = 1+a[gid];

}
