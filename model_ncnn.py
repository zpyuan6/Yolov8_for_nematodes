import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        # net.load_param("runs\\uk_pest_01JAN_tiny\\weights\\best_ncnn_model\\model.ncnn.param")
        # net.load_model("runs\\uk_pest_01JAN_tiny\\weights\\best_ncnn_model\\model.ncnn.bin")
        # net.load_param("D:\\android\\PestMobi\\app\\src\\main\\assets\\yolov8n.param")
        # net.load_model("D:\\android\\PestMobi\\app\\src\\main\\assets\\yolov8n.bin")
        net.load_param("C:\\Users\\zhipeng\\Desktop\\new\\best-sim-opt-fp16.param")
        net.load_model("C:\\Users\\zhipeng\\Desktop\\new\\best-sim-opt-fp16.bin")


        with net.create_extractor() as ex:
            ex.input("images", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("output0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

            print(out[0].shape)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
