#include <torch/torch.h>
#include <iostream>

void nms_cuda_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh);
int nms_cuda(at::Tensor keep_out, at::Tensor num_out, at::Tensor boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh){
    nms_cuda_compute(keep_out.data<int>(), num_out.data<int>(), boxes_host.data<float>(), boxes_num,
          boxes_dim, nms_overlap_thresh);
    return 0;        
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_cuda", &nms_cuda, "NMS cuda");

}