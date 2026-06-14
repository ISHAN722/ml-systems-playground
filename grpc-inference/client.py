import grpc
import inference_pb2
import inference_pb2_grpc

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"
    request = inference_pb2.GenerateRequest(prompt=prompt, max_new_tokens=30)

    response = stub.Generate(request)

    print("Response text:")
    print(response.text)
    print(f"\nLatency: {response.latency_ms:.2f} ms")

if __name__ == "__main__":
    run()