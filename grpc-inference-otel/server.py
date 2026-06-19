import time
import grpc
from concurrent import futures
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from opentelemetry import trace

import inference_pb2
import inference_pb2_grpc
from telemetry import setup_telemetry

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.tracer = setup_telemetry("grpc-inference")
        print("Loading Model...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        print("Model Loaded.")

    def Generate(self, request, context):
        # root span for the entire request
        with self.tracer.start_as_current_span("grpc.generate") as span:
            span.set_attribute("request.prompt_length", len(request.prompt))
            span.set_attribute("request.max_new_tokens", request.max_new_tokens or 50) 

            prompt = request.prompt
            max_new_tokens = request.max_new_tokens or 50

            # tokenization span
            with self.tracer.start_as_current_span("tokenize") as tok_span:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_len = inputs["input_ids"].shape[1]
                tok_span.set_attribute("tokens.input_count", input_len)

            # inference span
            with self.tracer.start_as_current_span("model.generate") as inf_span:
                start = time.perf_counter()
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids = inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )

                latency_ms = (time.perf_counter() - start) * 1000
                output_len = output_ids.shape[1] - input_len
                inf_span.set_attribute("model.latency_ms", round(latency_ms, 2))
                inf_span.set_attribute("tokens.output_count", output_len)
                inf_span.set_attribute("tokens.per_second",
                    round(output_len / (latency_ms / 1000), 2))

            # decode span
            with self.tracer.start_as_current_span("decode"):
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens = True)

            span.set_attribute('response.length', len(text))
            return inference_pb2.GenerateResponse(text=text, latency_ms = latency_ms)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server running on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()