from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

def setup_telemetry(service_name: str = "grpc-inference"):
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource) 

    # export traces to console (in production this would go to Jaeger/Temp)
    exporter = ConsoleSpanExporter()
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

