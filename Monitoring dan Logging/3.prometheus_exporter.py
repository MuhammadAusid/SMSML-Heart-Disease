import time
import random
from prometheus_client import start_http_server, Counter, Histogram

# Metrik Prometheus
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total prediksi yang dilakukan')
LATENCY_HISTOGRAM = Histogram('model_prediction_latency_seconds', 'Waktu latensi prediksi')

def process_request():
    start_time = time.time()
    # Simulasi pemrosesan inferensi model
    time.sleep(random.uniform(0.1, 0.5))
    PREDICTION_COUNTER.inc()
    LATENCY_HISTOGRAM.observe(time.time() - start_time)

if __name__ == '__main__':
    # Jalankan server Prometheus Exporter di port 8000
    start_http_server(8000)
    print("Prometheus exporter berjalan di http://localhost:8000")
    while True:
        process_request()
        time.sleep(1)
