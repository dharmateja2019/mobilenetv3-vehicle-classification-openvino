import time

from test.test_classify import predict as classify_pytorch
from src.classify.openvino_infer import classify_openvino

IMAGE = "data/classify/4W/4w_1.jpg"
RUNS = 20


def benchmark(fn, name):
    times = []
    for _ in range(RUNS):
        start = time.time()
        fn(IMAGE)
        times.append((time.time() - start) * 1000)

    avg = sum(times) / len(times)
    fps = 1000 / avg
    print(f"{name}: {avg:.2f} ms | {fps:.2f} FPS")


if __name__ == "__main__":
    benchmark(classify_pytorch, "PyTorch")
    benchmark(classify_openvino, "OpenVINO")
