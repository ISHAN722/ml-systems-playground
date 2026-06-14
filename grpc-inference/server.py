import time
import grpc
from concurrent import futures
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import inference_pb2
import inference_pb2_grpc

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        print("Loading Model...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        print("Model Loaded.")

    def Generate(self, request, context):
        import traceback
        try:
            prompt = request.prompt
            max_new_tokens = request.max_new_tokens or 50

            inputs = self.tokenizer(prompt, return_tensors="pt")

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            end = time.perf_counter()

            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            latency_ms = (end - start) * 1000

            return inference_pb2.GenerateResponse(text=text, latency_ms=latency_ms)
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.GenerateResponse()
        

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server running on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()