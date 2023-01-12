
__kernel void gamma_transform(__global uchar * img_src,
                              __global uchar * img_dst,
                              const int ncols,
                              __constant uchar * gamma_lut) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    int index = i * ncols + j;
    uchar map_val = gamma_lut[img_src[index]];
    uchar min_val = 0;
    uchar max_val = 255;
    img_dst[index] = clamp(map_val, min_val, max_val);

}